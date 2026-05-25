#!/usr/bin/env python3
"""
RAMS Event Argument Extraction (EAE) evaluation with vLLM + energy measurement + MLflow tracking.

Core pipeline: batch inference → telemetry collection → EAE evaluation → MLflow logging.
Telemetry utilities (energy scraping, vLLM metrics) live in ../xtreme/utils.py.

Dataset: RAMS 1.0c (local JSONL files, English only)
  - 9,124 events across 139 event types and 65 roles
  - 5-sentence document window per event
  - Single event trigger per document; arguments may span sentences
  - Splits: train=7,329 / dev=924 / test=871

Task: given document sentences + event trigger word + event type,
extract argument spans and map them to role labels.

Output format (prompt 1, JSON):
  {"killer": "Officer Caesar Goodson Jr.", "victim": "Freddie Gray", "place": "Baltimore"}

Evaluation metrics (span-level exact match, normalised):
  - Arg-C F1: role label AND argument text must match (argument classification)
  - Arg-I F1: argument text must match (argument identification, ignoring role)
"""

import argparse
import asyncio
import datetime
import glob
import json
import os
import re

# Reuse telemetry and HTTP utilities from the NER evaluation sibling package
import sys
import time

import aiohttp
import mlflow
import numpy as np
import pandas as pd
from aiohttp import ClientTimeout, TCPConnector
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "xtreme"))
from utils import (
    CHAT_URL,
    MLFLOW_TRACKING_URI,
    analyze_metrics_csv,
    compute_energy_corrected,
    job_id,
    metrics_csv_path,
    numpy_serializer,
    read_energy_joules,
    read_vllm_metrics,
)

print(CHAT_URL)
print(job_id)

# =====================================================================
# RAMS Dataset Loading
# =====================================================================

RAMS_SPLIT_FILES = {
    "train": "train.jsonlines",
    "dev": "dev.jsonlines",
    "test": "test.jsonlines",
}


def load_rams_split(data_dir: str, split: str = "dev") -> list:
    """Load a RAMS JSONL split from a local directory.

    Args:
        data_dir: path to directory containing train/dev/test.jsonlines
        split: one of "train", "dev", "test"

    Returns:
        list of dicts, one per document/event
    """
    fname = RAMS_SPLIT_FILES[split]
    path = os.path.join(data_dir, fname)
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


# =====================================================================
# RAMS Example Utilities
# =====================================================================

# Role names in RAMS look like "evt090arg01killer".
# We strip the numeric prefix to get the human-readable role (e.g. "killer").
_ROLE_PREFIX_RE = re.compile(r"^evt\d+arg\d+")


def clean_role(raw_role: str) -> str:
    """Strip the 'evt090arg01' prefix to get the readable role name ('killer')."""
    return _ROLE_PREFIX_RE.sub("", raw_role)


def flatten_tokens(example: dict) -> list:
    """Return all document tokens as a flat list (preserves global token indices)."""
    return [tok for sent in example["sentences"] for tok in sent]


def build_document_text(example: dict) -> str:
    """Return document as newline-separated sentences."""
    return "\n".join(" ".join(sent) for sent in example["sentences"])


def get_trigger_info(example: dict) -> tuple:
    """Return (trigger_word_string, raw_event_type_string)."""
    trig = example["evt_triggers"][0]
    flat = flatten_tokens(example)
    trigger_word = " ".join(flat[trig[0] : trig[1] + 1])
    event_type = trig[2][0][0]  # e.g. "life.die.deathcausedbyviolentevents"
    return trigger_word, event_type


def get_gold_args(example: dict) -> list:
    """Return list of (clean_role, arg_text) pairs for all gold event links."""
    flat = flatten_tokens(example)
    result = []
    for link in example["gold_evt_links"]:
        arg_span = link[1]
        arg_text = " ".join(flat[arg_span[0] : arg_span[1] + 1])
        role = clean_role(link[2])
        result.append((role, arg_text))
    return result


# =====================================================================
# Prompt Configuration (loaded from file; mirrors NER English_system_prompt.json)
# =====================================================================

_EE_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "prompts_in_all_languages")
_EE_SYSTEM_PROMPTS_FILE = os.path.join(_EE_PROMPTS_DIR, "ee_system_prompts.json")
_EE_FEW_SHOTS_FILE = os.path.join(_EE_PROMPTS_DIR, "ee_few_shots.json")

# Number of few-shot examples to prepend for each prompt style (mirrors NER pairs_for_choice)
EE_SHOTS_MAP = {1: 0, 2: 1, 3: 0, 4: 1, 5: 2, 6: 2, 7: 0, 8: 0}


def load_ee_system_prompt(choice: int) -> str:
    """Load system prompt text for a given choice from ee_system_prompts.json."""
    with open(_EE_SYSTEM_PROMPTS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    key = str(choice)
    if key not in data:
        raise ValueError(f"No EE system prompt found for choice: {choice}")
    val = data[key]
    return val if isinstance(val, str) else "".join(val)


def load_ee_few_shots() -> dict:
    """Load EE few-shot examples from ee_few_shots.json.

    Returns dict with keys 'shot1', 'shot2', each with 'user' and 'assistant' strings.
    """
    with open(_EE_FEW_SHOTS_FILE, encoding="utf-8") as f:
        return json.load(f)


def build_roles_by_event_type(train_jsonlines_path: str) -> dict:
    """Build mapping from event_type -> sorted list of clean role names from train split.

    Used to inject the valid role list into each inference prompt.
    """
    roles_by_type: dict = {}
    with open(train_jsonlines_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            etype = ex["evt_triggers"][0][2][0][0]
            if etype not in roles_by_type:
                roles_by_type[etype] = set()
            for link in ex["gold_evt_links"]:
                roles_by_type[etype].add(clean_role(link[2]))
    return {k: sorted(v) for k, v in roles_by_type.items()}


def build_ee_user_message(
    document_text: str,
    trigger_word: str,
    event_type: str,
    valid_roles_str: str = "",
) -> str:
    """Construct the user turn for a single EAE example."""
    roles_line = f"Valid roles: {valid_roles_str}\n" if valid_roles_str else ""
    return (
        f"Document:\n{document_text}\n\n"
        f'Event trigger: "{trigger_word}"\n'
        f"Event type: {event_type}\n"
        f"{roles_line}"
        "\nExtract the event arguments as a JSON object."
    )


def build_ee_messages(
    document_text: str,
    trigger_word: str,
    event_type: str,
    system_prompt_choice: int = 1,
    valid_roles_str: str = "",
) -> list:
    """Build the full chat message list for a single EAE example.

    For prompts with few-shot examples (2, 4, 5, 6), prepends the appropriate
    number of user/assistant shot pairs from ee_few_shots.json before the query.
    """
    system_content = load_ee_system_prompt(system_prompt_choice)
    user_content = build_ee_user_message(document_text, trigger_word, event_type, valid_roles_str)
    messages = [{"role": "system", "content": system_content}]
    n_shots = EE_SHOTS_MAP.get(system_prompt_choice, 0)
    if n_shots > 0:
        few_shots = load_ee_few_shots()
        for i in range(1, n_shots + 1):
            shot = few_shots.get(f"shot{i}", {})
            if shot:
                messages.append({"role": "user", "content": shot["user"]})
                messages.append({"role": "assistant", "content": shot["assistant"]})
    messages.append({"role": "user", "content": user_content})
    return messages


# =====================================================================
# Response Parsing
# =====================================================================

# Matches a JSON object anywhere in the model output (even wrapped in markdown fences)
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_ee_response(text: str) -> dict:
    """Parse model response text to a {role: arg_text} dict.

    Strategy:
    1. Strip markdown fences, try to parse the whole text as JSON.
    2. Extract the first {...} block and parse it.
    3. Return empty dict on failure.
    """
    if not text:
        return {}
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items() if isinstance(v, (str, int, float))}
    except json.JSONDecodeError:
        pass
    m = _JSON_OBJ_RE.search(cleaned)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict):
                return {str(k): str(v) for k, v in obj.items() if isinstance(v, (str, int, float))}
        except json.JSONDecodeError:
            pass
    return {}


# =====================================================================
# EAE Evaluation Metrics
# =====================================================================


def _normalise(text: str) -> str:
    """Lower-case and collapse whitespace for lenient text matching."""
    return " ".join(text.lower().split())


def compute_eae_metrics(gold_list: list, pred_list: list) -> dict:
    """Compute Arg-I and Arg-C F1 metrics over all examples.

    Args:
        gold_list: per-example list of (role, span_text) pairs
        pred_list: per-example dicts {role: span_text}

    Arg-I (Identification): span text must match (role ignored)
    Arg-C (Classification): role AND span text must both match

    Returns dict with keys: arg_i_f1, arg_c_f1, and supporting counts/scores.
    """
    tp_i = tp_c = 0
    total_gold = total_pred = 0

    for gold_pairs, pred_dict in zip(gold_list, pred_list):
        gold_texts = [_normalise(t) for _, t in gold_pairs]
        gold_pairs_norm = {(_normalise(t), r) for r, t in gold_pairs}  # (text, role) set
        pred_pairs_norm = [(r, _normalise(t)) for r, t in pred_dict.items()]

        total_gold += len(gold_pairs)
        total_pred += len(pred_pairs_norm)

        # Arg-I: span text appears in gold texts (any role)
        for _, pt in pred_pairs_norm:
            if pt in gold_texts:
                tp_i += 1

        # Arg-C: (text, role) pair matches a gold pair
        gold_set = {(t, r) for r, t in [(r, _normalise(t)) for r, t in gold_pairs]}
        for pr, pt in pred_pairs_norm:
            if (pt, pr) in gold_set:
                tp_c += 1

    prec_i = tp_i / total_pred if total_pred > 0 else 0.0
    rec_i = tp_i / total_gold if total_gold > 0 else 0.0
    f1_i = (2 * prec_i * rec_i / (prec_i + rec_i)) if (prec_i + rec_i) > 0 else 0.0

    prec_c = tp_c / total_pred if total_pred > 0 else 0.0
    rec_c = tp_c / total_gold if total_gold > 0 else 0.0
    f1_c = (2 * prec_c * rec_c / (prec_c + rec_c)) if (prec_c + rec_c) > 0 else 0.0

    return {
        "arg_i_precision": prec_i,
        "arg_i_recall": rec_i,
        "arg_i_f1": f1_i,
        "arg_c_precision": prec_c,
        "arg_c_recall": rec_c,
        "arg_c_f1": f1_c,
        "total_gold": total_gold,
        "total_pred": total_pred,
        "tp_arg_i": tp_i,
        "tp_arg_c": tp_c,
    }


# =====================================================================
# Batch Inference (chat-style models: Gemma, Mistral)
# =====================================================================


async def process_batch_chat(
    session,
    examples: list,
    max_tokens: int,
    model_name: str,
    system_prompt_choice: int = 1,
    semaphore=None,
    roles_by_event_type: dict = None,
):
    """Send a batch of RAMS examples to a chat-style model in parallel.

    Each example produces one chat completion request:
      - System message: EAE task instruction (from ee_system_prompts.json)
      - Few-shot turns: 0-2 user/assistant pairs depending on prompt style
      - User message: document + trigger + event type + valid roles

    Returns:
        list of response JSON objects (one per example), or Exception on failure
    """

    async def single_request(example: dict):
        doc_text = build_document_text(example)
        trigger_word, event_type = get_trigger_info(example)
        valid_roles = sorted(roles_by_event_type.get(event_type, [])) if roles_by_event_type else []
        valid_roles_str = ", ".join(valid_roles)
        messages = build_ee_messages(
            doc_text, trigger_word, event_type, system_prompt_choice, valid_roles_str
        )
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        async with semaphore, session.post(CHAT_URL, json=payload) as resp:
            return await resp.json()

    tasks = [single_request(ex) for ex in examples]
    return await asyncio.gather(*tasks, return_exceptions=True)


# =====================================================================
# Telemetry-Wrapped Batch Processing (identical structure to NER script)
# =====================================================================


async def process_and_measure(
    session,
    examples: list,
    max_tokens: int,
    model_name: str,
    system_prompt_choice: int = 1,
    semaphore=None,
    roles_by_event_type: dict = None,
):
    """Send batch to model while measuring energy and performance metrics.

    Measurement pattern:
      1. BEFORE: read GPU energy & vLLM metrics
      2. SEND: batch to model via process_batch_chat
      3. AFTER: read GPU energy & vLLM metrics again
      4. CALCULATE: differences → joules, tokens, latency
      5. DERIVE: throughput, energy-per-token, phase breakdown

    Returns:
        tuple of (responses, telemetry_dict)
    """
    # ===== BEFORE =====
    e0 = read_energy_joules()
    v0 = read_vllm_metrics()
    t0 = time.perf_counter()
    t0_epoch = time.time()
    print("Before batch request: ", datetime.datetime.now())

    responses = await process_batch_chat(
        session,
        examples,
        max_tokens,
        model_name,
        system_prompt_choice,
        semaphore=semaphore,
        roles_by_event_type=roles_by_event_type,
    )

    # ===== AFTER =====
    t1 = time.perf_counter()
    t1_epoch = time.time()
    print("After batch request: ", datetime.datetime.now())
    e1 = read_energy_joules()
    v1 = read_vllm_metrics()

    joules = max(e1 - e0, 0.0)
    latency = t1 - t0

    prompt_t = v1["prompt_tokens_total"] - v0["prompt_tokens_total"]
    gen_t = v1["generation_tokens_total"] - v0["generation_tokens_total"]
    total_t = prompt_t + gen_t

    e2e_count_diff = v1["e2e_latency_count"] - v0["e2e_latency_count"]
    e2e_latency_mean = (v1["e2e_latency_sum"] - v0["e2e_latency_sum"]) / max(e2e_count_diff, 1)

    ttft_count_diff = v1["time_to_first_count"] - v0["time_to_first_count"]
    ttft_mean = (v1["time_to_first_sum"] - v0["time_to_first_sum"]) / max(ttft_count_diff, 1)

    tpt_count_diff = v1["time_per_token_count"] - v0["time_per_token_count"]
    tpt_sum = v1["time_per_token_sum"] - v0["time_per_token_sum"]
    token_throughput = (tpt_count_diff / tpt_sum) if tpt_sum > 0 else 0.0
    time_per_token_mean = tpt_sum / max(tpt_count_diff, 1)

    prefill_total = v1["request_prefill_time_sum"] - v0["request_prefill_time_sum"]
    inference_total = v1["request_inference_time_sum"] - v0["request_inference_time_sum"]
    decode_total = v1["request_decode_time_sum"] - v0["request_decode_time_sum"]

    prefill_count_diff = v1["request_prefill_time_count"] - v0["request_prefill_time_count"]
    inference_count_diff = v1["request_inference_time_count"] - v0["request_inference_time_count"]
    decode_count_diff = v1["request_decode_time_count"] - v0["request_decode_time_count"]

    denom_prefill = prefill_count_diff if prefill_count_diff > 0 else max(len(examples), 1)
    denom_inference = inference_count_diff if inference_count_diff > 0 else max(len(examples), 1)
    denom_decode = decode_count_diff if decode_count_diff > 0 else max(len(examples), 1)

    prefill_avg_req = prefill_total / denom_prefill
    inference_avg_req = inference_total / denom_inference
    decode_avg_req = decode_total / denom_decode

    batch_elapsed = max(t1_epoch - t0_epoch, 1e-6)
    prefill_wall_candidate = (
        ttft_mean if (isinstance(ttft_mean, (int, float)) and ttft_mean > 0) else prefill_avg_req
    )
    prefill_wall_s = min(max(prefill_wall_candidate, 0.0), batch_elapsed)

    prompt_tps_prefill = (prompt_t / prefill_total) if prefill_total > 0 else 0.0
    decode_sum = v1["request_decode_time_sum"] - v0["request_decode_time_sum"]
    gen_tps_decode_time = (gen_t / decode_sum) if decode_sum > 0 else 0.0

    total_processing_time = max(e2e_latency_mean * len(examples), 1e-9)
    joules_prefill = (
        joules * (prefill_total / total_processing_time) if total_processing_time > 0 else 0.0
    )
    joules_inference = (
        joules * (inference_total / total_processing_time) if total_processing_time > 0 else 0.0
    )
    joules_decode = (
        joules * (decode_total / total_processing_time) if total_processing_time > 0 else 0.0
    )

    telemetry = {
        "batch_size": len(examples),
        "latency_s": latency,
        "energy_j": joules,
        "prompt_tokens": prompt_t,
        "generation_tokens": gen_t,
        "total_tokens": total_t,
        "e2e_latency_mean": e2e_latency_mean,
        "ttft_mean": ttft_mean,
        "time per token": time_per_token_mean,
        "token_throughput": token_throughput,
        "diff": latency - e2e_latency_mean,
        "prefill_avg_s": prefill_avg_req,
        "inference_avg_s": inference_avg_req,
        "decode_avg_s": decode_avg_req,
        "prefill_wall_s": prefill_wall_s,
        "prompt_tps_prefill": prompt_tps_prefill,
        "gen_tps_decode_time": gen_tps_decode_time,
        "J_total_per_token": joules / max(total_t, 1),
        "prefill_total_s": prefill_total,
        "inference_total_s": inference_total,
        "decode_total_s": decode_total,
        "joules_prefill": joules_prefill,
        "joules_inference": joules_inference,
        "joules_decode": joules_decode,
        "J_prefill_per_prompt_token": joules_prefill / max(prompt_t, 1),
        "J_inf_per_gen_token": joules_inference / max(gen_t, 1),
        "J_total_per_prompt_token": joules / max(prompt_t, 1),
        "J_total_per_gen_token": joules / max(gen_t, 1),
        "avg_power_draw": joules / max(e2e_latency_mean, 1e-6),
    }

    if len(examples) >= 32:
        try:
            csv_metrics = analyze_metrics_csv(t0_epoch, t1_epoch, prefill_wall_s, prompt_t, gen_t)
            if csv_metrics:
                telemetry.update(csv_metrics)
                csv_total = csv_metrics.get("csv_total_energy_j", 0.0)
                csv_pref = csv_metrics.get("csv_prefill_energy_j", 0.0)
                if csv_total > 0:
                    telemetry["csv_prefill_energy_ratio"] = csv_pref / csv_total
        except Exception as e:
            print(f"Error during CSV metrics analysis: {e}")

    return responses, telemetry


# =====================================================================
# EAE Evaluation Pipeline: Core Loop
# =====================================================================


async def evaluate_ee_pipeline_rams(
    dataset: list,
    batch_size: int,
    model_name: str,
    max_new_tokens: int = 200,
    system_prompt_choice: int = 1,
    max_concurrency: int = 256,
    roles_by_event_type: dict = None,
):
    """Run end-to-end EAE evaluation on RAMS examples.

    Process flow for each batch:
      1. Send batch to model via process_and_measure() (energy + latency)
      2. Parse model responses → {role: arg_text} dicts
      3. Accumulate gold and pred lists
    After all batches:
      4. Compute Arg-I / Arg-C F1

    Returns:
        (eae_metrics, generated_results, batch_telemetry)
    """
    all_gold = []  # list of list of (role, text) per example
    all_pred = []  # list of dict {role: text} per example
    generated_results = []
    batch_telemetry = []

    timeout = ClientTimeout(total=None, sock_connect=30, sock_read=900)
    connector = TCPConnector(limit=4096, enable_cleanup_closed=True, force_close=False)
    semaphore = asyncio.Semaphore(max_concurrency)
    print(f"[INFO] Semaphore: max {max_concurrency} concurrent HTTP requests")

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i : i + batch_size]

            try:
                responses, telemetry = await process_and_measure(
                    session,
                    batch,
                    max_new_tokens,
                    model_name,
                    system_prompt_choice,
                    semaphore=semaphore,
                    roles_by_event_type=roles_by_event_type,
                )
                batch_telemetry.append(telemetry)

                for example, response in zip(batch, responses):
                    trigger_word, event_type = get_trigger_info(example)
                    gold_pairs = get_gold_args(example)

                    # Extract model output text
                    if isinstance(response, Exception):
                        print(f"[WARN] Request failed: {type(response).__name__}: {response}")
                        text = ""
                    elif (
                        isinstance(response, dict) and "choices" in response and response["choices"]
                    ):
                        text = response["choices"][0].get("message", {}).get("content", "")
                    else:
                        text = str(response)

                    pred_dict = parse_ee_response(text)

                    # Debug: show first few examples
                    if len(generated_results) < 3:
                        print(f"\nDEBUG example {len(generated_results)}:")
                        print(f"  trigger: {trigger_word!r} | type: {event_type}")
                        print(f"  gold: {gold_pairs}")
                        print(f"  raw output[:200]: {text[:200]}")
                        print(f"  parsed pred: {pred_dict}")

                    generated_results.append(
                        {
                            "doc_key": example.get("doc_key", ""),
                            "trigger_word": trigger_word,
                            "event_type": event_type,
                            "gold_args": gold_pairs,
                            "raw_output": text,
                            "pred_args": pred_dict,
                        }
                    )
                    all_gold.append(gold_pairs)
                    all_pred.append(pred_dict)

            except Exception as e:
                print(f"Batch {i // batch_size} failed: {e}")
                for example in batch:
                    all_gold.append(get_gold_args(example))
                    all_pred.append({})
                continue

    if all_gold:
        eae_metrics = compute_eae_metrics(all_gold, all_pred)
    else:
        eae_metrics = {
            "arg_i_precision": 0.0,
            "arg_i_recall": 0.0,
            "arg_i_f1": 0.0,
            "arg_c_precision": 0.0,
            "arg_c_recall": 0.0,
            "arg_c_f1": 0.0,
            "total_gold": 0,
            "total_pred": 0,
            "tp_arg_i": 0,
            "tp_arg_c": 0,
        }

    return eae_metrics, generated_results, batch_telemetry


# =====================================================================
# MLflow Orchestration: Main Experiment Runner
# =====================================================================


async def run(args):
    # ===== STEP 1: Load Dataset =====
    rams_data_dir = os.path.join(args.rams_dir, "data")
    print(f"Loading RAMS split '{args.split}' from {rams_data_dir}")
    dataset = load_rams_split(rams_data_dir, split=args.split)

    if args.limit:
        dataset = dataset[: args.limit]
    print(f"Loaded {len(dataset)} RAMS examples (split={args.split})")

    # ===== STEP 1b: Build role ontology from train split =====
    train_jsonlines = os.path.join(args.rams_dir, "data", "train.jsonlines")
    roles_by_event_type = build_roles_by_event_type(train_jsonlines)
    print(f"[INFO] Role ontology: {len(roles_by_event_type)} event types loaded from train split")

    # ===== STEP 2: Setup Output Directories =====
    # Use ARTIFACTS_DIR from environment if set (by shell wrapper), otherwise build locally
    out_dir = os.environ.get("ARTIFACTS_DIR")
    if not out_dir:
        model_clean = args.model.lstrip("/").replace("/", "_")
        out_dir = (
            f"./inference_eval_artifacts/rams/"
            f"rams_{args.split}_B{args.batch_size}_{model_clean}_prompt{args.system_prompt_choice}_{job_id or 'local'}"
        )

    os.makedirs(os.path.join(out_dir, "generated_responses"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "ee_metrics"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "telemetry"), exist_ok=True)

    # ===== STEP 3: Setup MLflow =====
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", f"prompts_en_{args.model}_xtreme")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    timestamp_suffix = ""
    if getattr(args, "run_start_timestamp", None):
        try:
            dt = pd.to_datetime(args.run_start_timestamp)
            timestamp_suffix = f"_T{dt.strftime('%H%M%S')}"
        except Exception:
            pass

    prompt_label = f"prompt{args.system_prompt_choice}"

    with mlflow.start_run(
        run_name=f"rams_ee_b_{args.batch_size}_{args.split}_{prompt_label}{timestamp_suffix}_{job_id}"
    ):
        # ===== Log Configuration to MLflow =====
        tags = {
            "split": args.split,
            "batch_size": str(args.batch_size),
            "model": args.model,
            "dataset": "rams",
            "n_samples": str(len(dataset)),
            "max_concurrency": str(args.max_concurrency),
            "system_prompt_choice": str(args.system_prompt_choice),
        }
        params = {
            "split": args.split,
            "batch_size": args.batch_size,
            "model": args.model,
            "max_new_tokens": args.max_new_tokens,
            "max_concurrency": args.max_concurrency,
            "dataset": "rams",
            "n_samples": str(len(dataset)),
            "system_prompt_choice": args.system_prompt_choice,
        }
        mlflow.set_tags(tags)
        mlflow.log_params(params)

        # Log server config if available (written by one_to_rule_them_all.sh)
        host_base = os.environ.get("HOST_BASE")
        if not host_base:
            exp_result_base = os.environ.get("EXPERIMENT_RESULT_BASE")
            if exp_result_base:
                host_base = os.path.dirname(exp_result_base)
        if host_base and job_id:
            model_clean = args.model.lstrip("/")
            srv_pattern = os.path.join(
                host_base,
                "experiment_configs",
                f"*_job_{job_id}_m_{model_clean}_out_servconf.json",
            )
            srv_matches = glob.glob(srv_pattern)
            if srv_matches:
                srv_config_path = srv_matches[0]
                mlflow.log_artifact(srv_config_path)
                try:
                    with open(srv_config_path) as cf:
                        srv_cfg = json.load(cf)
                    cfg_params = {
                        k: srv_cfg[k]
                        for k in [
                            "max_model_len",
                            "max_num_seqs",
                            "max_num_batched_tokens",
                            "tensor_parallel_size",
                            "dtype",
                            "kv_cache_dtype",
                            "block_size",
                            "gpu_memory_utilization",
                            "enable_prefix_caching",
                            "enable_chunked_prefill",
                            "node_name",
                            "number_of_allocated_gpus",
                            "metrics_interval_ms",
                            "model_path",
                        ]
                        if k in srv_cfg
                    }
                    if cfg_params:
                        mlflow.log_params(cfg_params)
                except Exception as _e:
                    print(f"Warning: could not parse server config: {_e}")

        # Log sample prompt for audit / reproducibility
        sample_ex = dataset[0]
        sample_doc_text = build_document_text(sample_ex)
        sample_trigger, sample_type = get_trigger_info(sample_ex)
        sample_valid_roles = sorted(roles_by_event_type.get(sample_type, []))
        sample_roles_str = ", ".join(sample_valid_roles)
        sample_prompt = {
            "prompt_type": "ee_chat",
            "system_prompt_choice": args.system_prompt_choice,
            "n_shots": EE_SHOTS_MAP.get(args.system_prompt_choice, 0),
            "valid_roles_for_sample": sample_valid_roles,
            "system": load_ee_system_prompt(args.system_prompt_choice),
            "user": build_ee_user_message(
                sample_doc_text, sample_trigger, sample_type, sample_roles_str
            ),
        }
        sample_prompt_path = os.path.join(out_dir, f"sample_prompt_{prompt_label}.json")
        with open(sample_prompt_path, "w") as f:
            json.dump(sample_prompt, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(sample_prompt_path)

        # ===== STEP 4: Run EAE Evaluation =====
        eae_metrics, gen_responses, batch_telemetry = await evaluate_ee_pipeline_rams(
            dataset,
            batch_size=args.batch_size,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            system_prompt_choice=args.system_prompt_choice,
            max_concurrency=args.max_concurrency,
            roles_by_event_type=roles_by_event_type,
        )

        # ===== STEP 5: Save Results to JSON Files =====
        responses_path = os.path.join(
            out_dir,
            "generated_responses",
            f"responses_B{args.batch_size}_{prompt_label}_{job_id}.json",
        )
        ee_metrics_path = os.path.join(
            out_dir,
            "ee_metrics",
            f"ee_metrics_B_{args.batch_size}_{prompt_label}_{job_id}.json",
        )
        telemetry_path = os.path.join(
            out_dir,
            "telemetry",
            f"telemetry_B_{args.batch_size}_{prompt_label}_{job_id}.json",
        )

        with open(responses_path, "w") as f:
            json.dump(gen_responses, f, indent=2, default=numpy_serializer, ensure_ascii=False)
        with open(ee_metrics_path, "w") as f:
            json.dump(eae_metrics, f, indent=2, default=numpy_serializer, ensure_ascii=False)
        with open(telemetry_path, "w") as f:
            json.dump(batch_telemetry, f, indent=2, default=numpy_serializer, ensure_ascii=False)

        mlflow.log_artifact(responses_path)
        mlflow.log_artifact(ee_metrics_path)
        mlflow.log_artifact(telemetry_path)

        # ===== STEP 6: Log EAE Metrics to MLflow =====
        mlflow.log_metric("arg_i_precision", float(eae_metrics.get("arg_i_precision", 0.0)))
        mlflow.log_metric("arg_i_recall", float(eae_metrics.get("arg_i_recall", 0.0)))
        mlflow.log_metric("arg_i_f1", float(eae_metrics.get("arg_i_f1", 0.0)))
        mlflow.log_metric("arg_c_precision", float(eae_metrics.get("arg_c_precision", 0.0)))
        mlflow.log_metric("arg_c_recall", float(eae_metrics.get("arg_c_recall", 0.0)))
        mlflow.log_metric("arg_c_f1", float(eae_metrics.get("arg_c_f1", 0.0)))

        # ===== STEP 7: Aggregate Telemetry and Efficiency Metrics =====
        mean_metrics = {}

        if batch_telemetry:
            all_keys = set()
            for t in batch_telemetry:
                all_keys.update(t.keys())
            for key in sorted(all_keys):
                vals = [
                    t[key]
                    for t in batch_telemetry
                    if isinstance(t.get(key), (int, float)) and not isinstance(t.get(key), bool)
                ]
                if vals:
                    mean_metrics[key] = float(np.mean(vals))

            energy_j_sum = float(
                sum(
                    t.get("energy_j", 0.0)
                    for t in batch_telemetry
                    if isinstance(t.get("energy_j"), (int, float))
                )
            )
            mean_metrics["energy_j_sum"] = energy_j_sum

            csv_prefill_energy_sum = float(
                sum(
                    t.get("csv_prefill_energy_j", 0.0)
                    for t in batch_telemetry
                    if isinstance(t.get("csv_prefill_energy_j"), (int, float))
                )
            )
            csv_generation_energy_sum = float(
                sum(
                    t.get("csv_generation_energy_j", 0.0)
                    for t in batch_telemetry
                    if isinstance(t.get("csv_generation_energy_j"), (int, float))
                )
            )
            csv_total_energy_sum = csv_prefill_energy_sum + csv_generation_energy_sum
            mean_metrics["csv_prefill_energy_j_sum"] = csv_prefill_energy_sum
            mean_metrics["csv_generation_energy_j_sum"] = csv_generation_energy_sum
            mean_metrics["csv_total_energy_j_sum"] = csv_total_energy_sum
            if csv_total_energy_sum > 0:
                mean_metrics["csv_prefill_energy_ratio"] = (
                    csv_prefill_energy_sum / csv_total_energy_sum
                )
                mean_metrics["csv_generation_energy_ratio"] = (
                    csv_generation_energy_sum / csv_total_energy_sum
                )

            total_gen_tokens = float(
                sum(
                    t.get("generation_tokens", 0)
                    for t in batch_telemetry
                    if isinstance(t.get("generation_tokens"), (int, float))
                )
            )
            total_examples = sum(
                t.get("batch_size", 0)
                for t in batch_telemetry
                if isinstance(t.get("batch_size"), (int, float))
            )
            mean_metrics["avg_gen_tokens_per_example"] = total_gen_tokens / max(total_examples, 1)

            arg_c_f1 = float(eae_metrics.get("arg_c_f1", 0.0))
            total_pred = int(eae_metrics.get("total_pred", 0))
            tp_c = int(eae_metrics.get("tp_arg_c", 0))
            total_seconds = float(sum(t.get("latency_s", 0.0) for t in batch_telemetry))

            mean_metrics.update(
                {
                    "ee_arg_i_f1": float(eae_metrics.get("arg_i_f1", 0.0)),
                    "ee_arg_c_f1": arg_c_f1,
                    "ee_arg_i_precision": float(eae_metrics.get("arg_i_precision", 0.0)),
                    "ee_arg_i_recall": float(eae_metrics.get("arg_i_recall", 0.0)),
                    "ee_arg_c_precision": float(eae_metrics.get("arg_c_precision", 0.0)),
                    "ee_arg_c_recall": float(eae_metrics.get("arg_c_recall", 0.0)),
                    "whole_energy": energy_j_sum,
                    "total_seconds": total_seconds,
                    "total_pred_args": total_pred,
                    "total_gold_args": int(eae_metrics.get("total_gold", 0)),
                }
            )

            # Energy-efficiency metrics (mirror NER script conventions; F1 = Arg-C F1)
            if arg_c_f1 > 0 and energy_j_sum > 0:
                mean_metrics["J_per_F1"] = energy_j_sum / arg_c_f1
                mean_metrics["F1_per_J"] = arg_c_f1 / energy_j_sum

            if total_pred > 0 and energy_j_sum > 0:
                mean_metrics["J_per_entity"] = energy_j_sum / total_pred
                mean_metrics["entities_per_J"] = total_pred / energy_j_sum

            if tp_c > 0 and energy_j_sum > 0:
                mean_metrics["J_per_TP"] = energy_j_sum / tp_c
                mean_metrics["TP_per_J"] = tp_c / energy_j_sum
                mean_metrics["TP_per_s"] = tp_c / total_seconds if total_seconds > 0 else 0.0
                mean_metrics["s_per_TP"] = total_seconds / tp_c if total_seconds > 0 else 0.0

            loggable = {
                k: float(v)
                for k, v in mean_metrics.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            mlflow.log_metrics(loggable)

            mean_metrics_path = os.path.join(
                out_dir, f"mean_metrics_B_{args.batch_size}_{prompt_label}_{job_id}.json"
            )
            with open(mean_metrics_path, "w") as f:
                json.dump(mean_metrics, f, indent=2, default=numpy_serializer)
            mlflow.log_artifact(mean_metrics_path)

            for cfg in glob.glob(os.path.join(out_dir, "*out_inference.json")):
                mlflow.log_artifact(cfg)

        # ===== STEP 8: Corrected Energy from CSV =====
        time.sleep(10)
        run_start_timestamp = getattr(args, "run_start_timestamp", None)
        res = compute_energy_corrected(metrics_csv_path, start_timestamp=run_start_timestamp)
        if res.get("ok"):
            mlflow.log_metric("energy_corrected", float(res["energy_corrected"]))
            mlflow.log_metric("latency_corrected", float(res["latency_corrected"]))
            mean_metrics.update(
                {
                    "energy_corrected": float(res["energy_corrected"]),
                    "latency_corrected": float(res["latency_corrected"]),
                }
            )
            mlflow.log_params(
                {
                    "energy_window_start": res["t_start"],
                    "energy_window_end": res["t_end"],
                    "energy_start_util": res["start_util"],
                    "energy_end_util": res["end_util"],
                    "run_start_timestamp": run_start_timestamp,
                }
            )
        else:
            mlflow.set_tag("energy_window_note", f"skipped: {res.get('reason')}")

        if metrics_csv_path and os.path.isfile(metrics_csv_path):
            mlflow.log_artifact(metrics_csv_path)

        print(
            f"Completed: Arg-C F1={eae_metrics.get('arg_c_f1', 0.0):.4f}, "
            f"Arg-I F1={eae_metrics.get('arg_i_f1', 0.0):.4f}, "
            f"Energy Sum={mean_metrics.get('energy_j_sum', 0.0):.4f} J"
        )


# =====================================================================
# Command-Line Interface
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="RAMS EAE evaluation with vLLM + energy measurement + MLflow"
    )

    parser.add_argument(
        "--rams-dir",
        type=str,
        required=True,
        help="Path to the RAMS_1.0c root directory (must contain data/train.jsonlines etc.).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["train", "dev", "test"],
        help="Which RAMS split to evaluate (default: dev).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["/gemma-3-4b-it", "/gemma-3-12b-it", "/gemma-3-27b-it", "/mistral", "/llama-3-3-70B-it"],
        help="Which vLLM-served model to use.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of examples per batch.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate per example (longer than NER due to multi-argument output).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=256,
        help="Max concurrent HTTP requests per batch (semaphore limit).",
    )
    parser.add_argument(
        "--system-prompt-choice",
        type=int,
        choices=range(1, 11),
        default=1,
        help="Which system prompt variant to use (1 = minimal; more prompts to be added).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Sample only first N examples (for quick testing).",
    )
    parser.add_argument(
        "--run-start-timestamp",
        type=str,
        default=None,
        help="ISO timestamp for energy window start (for CSV alignment). Format: YYYY-MM-DDTHH:MM:SS.ssssss",
    )

    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
