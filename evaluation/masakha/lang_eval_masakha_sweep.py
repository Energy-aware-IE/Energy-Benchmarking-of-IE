#!/usr/bin/env python3
"""
MasakhaNER2 NER evaluation sweep script — mirrors the XTREME sweep script structure.

Pipeline: batch inference → telemetry collection → NER evaluation → MLflow logging.
Shares telemetry/energy utilities with the XTREME sweep via ../xtreme/utils.py.

Key differences from XTREME:
  - Dataset: masakhane/masakhaner2 (African NER, 20 languages)
  - Language codes: 3-letter ISO 639-3 (bam, ewe, hau, swa, yor, …)
  - Few-shots: prompts_in_all_languages/ner_few_shots_masakha.json
  - Gold labels may include B-DATE/I-DATE → projected to O (only PER/ORG/LOC evaluated)
  - Prompts 1-8 (no DSPy / style 9)
  - Run name prefix: masakha_ner_b_
"""

import argparse
import asyncio
import datetime
import glob
import json
import os
import re
import sys
import time

import aiohttp
import mlflow
from aiohttp import ClientTimeout, TCPConnector
try:
    from datasets import load_dataset as _hf_load_dataset
except ImportError:
    _hf_load_dataset = None
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm

# ── Import shared utilities from the XTREME eval directory ────────────────────
_xtreme_dir = os.path.join(os.path.dirname(__file__), "../xtreme")
sys.path.insert(0, os.path.abspath(_xtreme_dir))

from utils import (
    CHAT_URL,
    MLFLOW_TRACKING_URI,
    analyze_metrics_csv,
    compute_energy_corrected,
    get_bio_tags_language_aware,
    job_id,
    metrics_csv_path,
    numpy_serializer,
    parse_response_json_like,
    read_energy_joules,
    read_vllm_metrics,
    robust_gemma_entity_extraction,
)

print(CHAT_URL)
print(job_id)

# =============================================================================
# MasakhaNER2 local data loader (datasets 4.x no longer supports loading scripts)
# =============================================================================

# Label names as defined in the original masakhaner2.py loading script
_MASAKHANER2_LABEL_NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-DATE", "I-DATE"]

# Default local data directory (pre-downloaded from GitHub)
_MASAKHANER2_LOCAL_DIR = os.path.join(
    os.path.dirname(__file__), "../../../../data/masakhaner2"
)


def _load_masakhaner2_from_txt(filepath: str) -> list:
    """Parse a CoNLL-style BIO txt file into a list of dicts with id/tokens/ner_tags."""
    label2id = {label: i for i, label in enumerate(_MASAKHANER2_LABEL_NAMES)}
    examples = []
    tokens, tags = [], []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line == "" or line.startswith("-DOCSTART-"):
                if tokens:
                    examples.append({
                        "id": str(len(examples)),
                        "tokens": tokens,
                        "ner_tags": [label2id.get(t, 0) for t in tags],
                    })
                    tokens, tags = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    tags.append(parts[1])
    if tokens:
        examples.append({
            "id": str(len(examples)),
            "tokens": tokens,
            "ner_tags": [label2id.get(t, 0) for t in tags],
        })
    return examples


class _FakeFeature:
    """Minimal stand-in for ClassLabel.names to match the HF dataset API."""
    names = _MASAKHANER2_LABEL_NAMES


class _FakeFeatures:
    ner_tags = type("_Seq", (), {"feature": _FakeFeature()})()

    def __getitem__(self, key):
        return getattr(self, key)


class _FakeSplit:
    """Minimal stand-in for an HF dataset split."""
    def __init__(self, data: list):
        self._data = data
        self.features = _FakeFeatures()

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def select(self, indices):
        return _FakeSplit([self._data[i] for i in indices])


def load_masakhaner2_local(language: str, split: str = "test") -> "_FakeSplit":
    """Load MasakhaNER2 from pre-downloaded local txt files."""
    txt_path = os.path.join(_MASAKHANER2_LOCAL_DIR, language, f"{split}.txt")
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(
            f"MasakhaNER2 local data not found: {txt_path}\n"
            f"Pre-download from https://github.com/masakhane-io/masakhane-ner/raw/main/MasakhaNER2.0/data/"
        )
    data = _load_masakhaner2_from_txt(txt_path)
    return _FakeSplit(data)


def load_masakhaner2(language: str) -> dict:
    """
    Load MasakhaNER2 dataset, preferring local pre-downloaded txt files.
    Falls back to HuggingFace datasets if local files are unavailable.
    """
    local_dir = os.path.abspath(_MASAKHANER2_LOCAL_DIR)
    local_test = os.path.join(local_dir, language, "test.txt")
    if os.path.isfile(local_test):
        print(f"[INFO] Loading MasakhaNER2 from local files: {local_dir}/{language}/")
        return {
            "train": load_masakhaner2_local(language, "train"),
            "validation": load_masakhaner2_local(language, "dev"),
            "test": load_masakhaner2_local(language, "test"),
        }
    # Fallback: HuggingFace (requires older datasets version)
    if _hf_load_dataset is None:
        raise RuntimeError("datasets library not available and no local data found.")
    print(f"[WARN] Local data not found for {language}, falling back to HuggingFace load_dataset")
    return _hf_load_dataset("masakhane/masakhaner2", language, trust_remote_code=True)


# =============================================================================
# MasakhaNER2 constants
# =============================================================================

# Full language name lookup (ISO 639-3 code → English name)
MASAKHANER2_LANGS = {
    "bam": "Bambara",
    "ewe": "Ewe",
    "fon": "Fon",
    "hau": "Hausa",
    "ibo": "Igbo",
    "kin": "Kinyarwanda",
    "lug": "Luganda",
    "luo": "Dholuo",
    "mos": "Mossi",
    "pcm": "Nigerian Pidgin",
    "sna": "Shona",
    "swa": "Swahili",
    "tsn": "Setswana",
    "twi": "Twi",
    "wol": "Wolof",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "zul": "Zulu",
    "bbj": "Ghomala",
    "nya": "Chichewa",
}

# Only evaluate PER, ORG, LOC — DATE tags (present in MasakhaNER2) are projected to O
TARGET_TYPES = {"PER", "ORG", "LOC"}

# Masakha-specific few-shots file
_script_dir = os.path.dirname(__file__)
_serve_vllm_dir = os.path.abspath(os.path.join(_script_dir, "../../"))
MASAKHA_FEW_SHOTS_PATH = os.path.join(
    _serve_vllm_dir, "prompts_in_all_languages/ner_few_shots_masakha.json"
)
TEMPLATE_PROMPT_PATH = os.path.join(
    _serve_vllm_dir, "prompts_in_all_languages/Template_system_prompt.json"
)

# =============================================================================
# Masakha-specific prompt helpers
# =============================================================================


def masakha_lang_name(language: str) -> str:
    return MASAKHANER2_LANGS.get(language, language)


def load_masakha_few_shots(language: str) -> list:
    with open(MASAKHA_FEW_SHOTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if language not in data:
        # Swahili is a good fallback — high quality, related to many target languages
        fallback = "swa" if "swa" in data else next(iter(data))
        print(f"[WARN] No few-shots for {language}, using '{fallback}' as fallback")
        return data[fallback]
    return data[language]


def load_masakha_system_template(language: str, system_prompt_choice: int) -> str:
    """Load Template_system_prompt.json (prompts 1-8) with {language} substituted."""
    with open(TEMPLATE_PROMPT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    key = str(system_prompt_choice)
    if key not in data:
        raise ValueError(f"No system prompt for choice: {system_prompt_choice}")
    prompt = data[key]
    if isinstance(prompt, list):
        prompt = "\n".join(prompt)
    return prompt.replace("{language}", masakha_lang_name(language))


def build_masakha_gemma_messages(
    sentence: str,
    language: str,
    system_prompt_choice: int,
    examples_for_lang: list,
) -> list:
    """Build chat messages for Gemma/Mistral using MasakhaNER2 system prompts."""
    sys_msg = load_masakha_system_template(language, system_prompt_choice)
    messages = [{"role": "system", "content": sys_msg}]

    pairs_for_choice = {1: 0, 2: 1, 3: 0, 4: 1, 5: 2, 6: 3, 7: 0, 8: 0}
    n_pairs = pairs_for_choice.get(system_prompt_choice, 0)
    if n_pairs > 0:
        for i in range(min(n_pairs * 2, len(examples_for_lang))):
            messages.append(examples_for_lang[i])

    messages.append({"role": "user", "content": sentence})
    return messages


def project_to_targets(tags: list, target_types: set = TARGET_TYPES) -> list:
    """Project non-PER/ORG/LOC tags (e.g. B-DATE, I-DATE) to O."""
    result = []
    for tag in tags:
        if tag == "O":
            result.append("O")
        else:
            parts = tag.split("-", 1)
            if len(parts) == 2 and parts[1] in target_types:
                result.append(tag)
            else:
                result.append("O")
    return result


def build_masakha_artifacts_dir(language: str, batch_size: int, model_name: str) -> str:
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_name)
    return f"masakha_{language}_B{batch_size}_{safe_model}_{job_id}"


# =============================================================================
# Batch inference (shared pattern with XTREME eval script)
# =============================================================================


async def process_batch_chat(
    session,
    sentences,
    max_tokens,
    model_name,
    language,
    system_prompt_choice=None,
    examples_for_lang=None,
    semaphore=None,
):
    """Send batch of sentences to chat-style model (Gemma, Mistral) in parallel."""

    async def single_request(sentence):
        messages = build_masakha_gemma_messages(
            sentence,
            language,
            system_prompt_choice or 1,
            examples_for_lang or [],
        )
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stop": ["}"],
        }
        try:
            async with semaphore, session.post(CHAT_URL, json=payload) as resp:
                if resp.status != 200:
                    return f"HTTP {resp.status}"
                result = await resp.json()
                choices = result.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    return msg.get("content", "") or choices[0].get("text", "")
                return ""
        except Exception as exc:
            return exc

    tasks = [single_request(s) for s in sentences]
    return await asyncio.gather(*tasks, return_exceptions=True)


# =============================================================================
# Telemetry-wrapped batch processing (identical to XTREME)
# =============================================================================


async def process_and_measure(
    session,
    prompts,
    max_tokens,
    model_name,
    language,
    system_prompt_choice=None,
    examples_for_lang=None,
    semaphore=None,
):
    e0 = read_energy_joules()
    v0 = read_vllm_metrics()
    t0 = time.perf_counter()
    t0_epoch = time.time()
    print("Before batch request: ", datetime.datetime.now())

    responses = await process_batch_chat(
        session,
        prompts,
        max_tokens,
        model_name,
        language,
        system_prompt_choice=system_prompt_choice,
        examples_for_lang=examples_for_lang,
        semaphore=semaphore,
    )

    t1 = time.perf_counter()
    t1_epoch = time.time()
    print("After batch request: ", datetime.datetime.now())
    e1 = read_energy_joules()
    v1 = read_vllm_metrics()

    joules = max(e1 - e0, 0.0)
    latency = t1 - t0

    prompt_t = max(v1["prompt_tokens_total"] - v0["prompt_tokens_total"], 0)
    gen_t = max(v1["generation_tokens_total"] - v0["generation_tokens_total"], 0)
    total_t = prompt_t + gen_t

    e2e_count_diff = max(v1["e2e_latency_count"] - v0["e2e_latency_count"], 0)
    e2e_latency_mean = max(
        (v1["e2e_latency_sum"] - v0["e2e_latency_sum"]) / max(e2e_count_diff, 1), 0.0
    )

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

    denom_prefill = prefill_count_diff if prefill_count_diff > 0 else max(len(prompts), 1)
    denom_inference = inference_count_diff if inference_count_diff > 0 else max(len(prompts), 1)
    denom_decode = decode_count_diff if decode_count_diff > 0 else max(len(prompts), 1)

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

    total_processing_time = max(e2e_latency_mean * len(prompts), 1e-9)
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
        "batch_size": len(prompts),
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
        "J_gen_per_token": joules / max(gen_t, 1),
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

    if len(prompts) >= 32:
        try:
            csv_metrics = analyze_metrics_csv(metrics_csv_path, t0_epoch, t1_epoch)
            if csv_metrics:
                telemetry.update({f"csv_{k}": v for k, v in csv_metrics.items()})
        except Exception as exc:
            print(f"[WARN] CSV analysis failed: {exc}")

    return responses, telemetry


# =============================================================================
# NER evaluation pipeline for MasakhaNER2
# =============================================================================


async def evaluate_ner_pipeline_masakha(
    test_dataset,
    label_list,
    batch_size,
    model_name,
    max_new_tokens=150,
    language="hau",
    system_prompt_choice=None,
    max_concurrency=1024,
):
    """
    Run end-to-end NER evaluation on MasakhaNER2.

    Gold tags come directly from the dataset (with DATE projected to O).
    Predicted tags are generated by the model and aligned to the gold token sequence.
    """
    all_gold, all_pred = [], []
    generated_results = []
    batch_telemetry = []

    examples_for_lang = load_masakha_few_shots(language)

    timeout = ClientTimeout(total=None, sock_connect=30, sock_read=900)
    connector = TCPConnector(limit=4096, enable_cleanup_closed=True, force_close=False)
    semaphore = asyncio.Semaphore(max_concurrency)
    print(f"[INFO] Semaphore: max {max_concurrency} concurrent HTTP requests")

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for batch_start in tqdm(range(0, len(test_dataset), batch_size)):
            batch = test_dataset.select(
                range(batch_start, min(batch_start + batch_size, len(test_dataset)))
            )

            # Build sentences and gold BIO tags from dataset tokens
            sentences = [" ".join(ex["tokens"]) for ex in batch]
            gold_tags_batch = []
            for ex in batch:
                raw_tags = [label_list[t] for t in ex["ner_tags"]]
                projected = project_to_targets(raw_tags)
                gold_tags_batch.append(projected)

            # Run inference + measure energy
            responses, telemetry = await process_and_measure(
                session,
                sentences,
                max_new_tokens,
                model_name,
                language,
                system_prompt_choice=system_prompt_choice,
                examples_for_lang=examples_for_lang,
                semaphore=semaphore,
            )
            batch_telemetry.append(telemetry)

            # Parse responses → entity dicts → predicted BIO tags
            for i, (sentence, gold_tags, response) in enumerate(
                zip(sentences, gold_tags_batch, responses)
            ):
                if isinstance(response, Exception):
                    entities = {"PER": [], "ORG": [], "LOC": []}
                    raw_text = ""
                else:
                    raw_text = str(response) if response else ""
                    entities = robust_gemma_entity_extraction(raw_text)
                    if not any(entities.values()):
                        entities = parse_response_json_like(raw_text)

                # Keep only PER/ORG/LOC keys
                entities = {k: v for k, v in entities.items() if k in TARGET_TYPES}
                for k in TARGET_TYPES:
                    entities.setdefault(k, [])

                # Generate predicted BIO tags aligned to dataset tokens
                tokens = batch[i]["tokens"]
                token_sentence = " ".join(tokens)
                pred_tokens, pred_tags = get_bio_tags_language_aware(
                    token_sentence, entities, language
                )

                # Align lengths: truncate or pad to match gold
                n_gold = len(gold_tags)
                pred_tags = pred_tags[:n_gold] + ["O"] * max(0, n_gold - len(pred_tags))

                all_gold.append(gold_tags)
                all_pred.append(pred_tags)
                generated_results.append(
                    {
                        "sentence": sentence,
                        "tokens": tokens,
                        "raw_output": raw_text,
                        "entities": entities,
                        "gold_tags": gold_tags,
                        "pred_tags": pred_tags,
                    }
                )

    # Compute final NER metrics (entity-span level via seqeval)
    if all_gold and all_pred:
        ner_metrics = {
            "f1": f1_score(all_gold, all_pred, zero_division=0),
            "precision": precision_score(all_gold, all_pred, zero_division=0),
            "recall": recall_score(all_gold, all_pred, zero_division=0),
            "classification_report": classification_report(all_gold, all_pred, zero_division=0),
            "n_predicted_entities": sum(
                1 for seq in all_pred for tag in seq if tag.startswith("B-")
            ),
            "n_true_entities": sum(1 for seq in all_gold for tag in seq if tag.startswith("B-")),
        }
    else:
        ner_metrics = {
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "classification_report": "",
            "n_predicted_entities": 0,
            "n_true_entities": 0,
        }

    return ner_metrics, generated_results, batch_telemetry


# =============================================================================
# Main orchestration: dataset load → eval → MLflow logging
# =============================================================================


async def run(args):
    language = args.language
    model_name = args.model
    batch_size = args.batch_size
    max_new_tokens = args.max_new_tokens
    system_prompt_choice = args.system_prompt_choice
    max_concurrency = args.max_concurrency
    limit = args.limit
    run_start_timestamp = args.run_start_timestamp

    lang_full = masakha_lang_name(language)
    print(f"[INFO] MasakhaNER2 language: {language} ({lang_full})")

    # ── Load MasakhaNER2 dataset ───────────────────────────────────────────────
    print(f"[INFO] Loading masakhane/masakhaner2 config={language} ...")
    try:
        ds = load_masakhaner2(language)
    except Exception as exc:
        print(f"[ERROR] Failed to load MasakhaNER2 for {language}: {exc}")
        raise

    test_data = ds["test"]
    label_list = test_data.features["ner_tags"].feature.names
    print(f"[INFO] Test split: {len(test_data)} examples | Labels: {label_list}")

    if limit and limit > 0:
        test_data = test_data.select(range(min(limit, len(test_data))))
        print(f"[INFO] Limiting to {len(test_data)} examples")

    # ── Set up output directories ──────────────────────────────────────────────
    artifacts_dir = os.environ.get("ARTIFACTS_DIR", ".")
    subdir_name = build_masakha_artifacts_dir(language, batch_size, model_name)
    per_run_dir = os.path.join(artifacts_dir, subdir_name)
    os.makedirs(per_run_dir, exist_ok=True)
    os.makedirs(os.path.join(per_run_dir, "telemetry"), exist_ok=True)

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow_exp = os.environ.get(
        "MLFLOW_EXPERIMENT_NAME",
        f"masakha_{language}_{model_name.replace('/', '')}",
    )
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(mlflow_exp)

    sweep_param_changed = os.environ.get("SWEEP_PARAM_CHANGED", "BASELINE")
    sweep_param_value = os.environ.get("SWEEP_PARAM_VALUE", "BASELINE")
    format_mode = os.environ.get("SWEEP_FORMAT_MODE", "json__false")

    # Construct run name (same pattern as XTREME for analysis script compatibility)
    ts = datetime.datetime.now().strftime("%H%M%S")
    run_name = f"masakha_ner_b_{batch_size}_{language}_prompt{system_prompt_choice}_T{ts}_{job_id}"
    if format_mode != "json__false":
        dm_tag = format_mode.replace("__", "_")
        run_name = (
            f"masakha_ner_b_{batch_size}_{language}_prompt{system_prompt_choice}"
            f"_{dm_tag}_T{ts}_{job_id}"
        )

    print(f"[INFO] MLflow experiment : {mlflow_exp}")
    print(f"[INFO] MLflow run name   : {run_name}")
    print(f"[INFO] Sweep param       : {sweep_param_changed}={sweep_param_value}")

    # ── Run NER evaluation ─────────────────────────────────────────────────────
    print(f"[INFO] Starting NER evaluation (prompt style {system_prompt_choice}) ...")
    ner_metrics, generated_results, batch_telemetry = await evaluate_ner_pipeline_masakha(
        test_data,
        label_list,
        batch_size=batch_size,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        language=language,
        system_prompt_choice=system_prompt_choice,
        max_concurrency=max_concurrency,
    )

    print(
        f"[INFO] F1={ner_metrics['f1']:.4f}  P={ner_metrics['precision']:.4f}  R={ner_metrics['recall']:.4f}"
    )

    # ── Save generated results JSON ────────────────────────────────────────────
    results_path = os.path.join(
        per_run_dir,
        f"generated_results_{language}_prompt{system_prompt_choice}_{job_id}.json",
    )
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(generated_results, f, ensure_ascii=False, indent=2, default=numpy_serializer)

    # ── Compute energy-corrected metrics ──────────────────────────────────────
    _corrected_res = {}
    if run_start_timestamp and metrics_csv_path and os.path.exists(metrics_csv_path or ""):
        try:
            _corrected_res = compute_energy_corrected(metrics_csv_path, start_timestamp=run_start_timestamp)
        except Exception as exc:
            print(f"[WARN] compute_energy_corrected failed: {exc}")

    # ── Aggregate telemetry ────────────────────────────────────────────────────
    def _agg(key):
        vals = [t[key] for t in batch_telemetry if key in t and isinstance(t[key], (int, float))]
        return sum(vals) if vals else 0.0

    total_gen_tokens = _agg("generation_tokens")
    n_sentences = len(test_data)
    avg_gen_tokens = total_gen_tokens / max(n_sentences, 1)

    # ── Energy-efficiency metrics ──────────────────────────────────────────────
    energy_j_sum = _agg("energy_j")
    total_seconds = _agg("latency_s")
    f1 = float(ner_metrics.get("f1", 0.0))
    pred_entities = int(ner_metrics.get("n_predicted_entities", 0))
    tp = float(ner_metrics.get("precision", 0.0)) * float(pred_entities)

    efficiency_metrics: dict = {}
    if f1 > 0 and energy_j_sum > 0:
        efficiency_metrics["J_per_F1"] = energy_j_sum / f1
        efficiency_metrics["F1_per_J"] = f1 / energy_j_sum
    if pred_entities > 0 and energy_j_sum > 0:
        efficiency_metrics["J_per_entity"] = energy_j_sum / pred_entities
        efficiency_metrics["entities_per_J"] = pred_entities / energy_j_sum
    if tp > 0 and energy_j_sum > 0:
        efficiency_metrics["J_per_TP"] = energy_j_sum / tp
        efficiency_metrics["TP_per_J"] = tp / energy_j_sum
        efficiency_metrics["s_per_TP"] = total_seconds / tp if total_seconds > 0 else 0.0
        efficiency_metrics["TP_per_s"] = tp / total_seconds if total_seconds > 0 else 0.0

    corrected_metrics: dict = {}
    if _corrected_res.get("ok"):
        energy_corrected = float(_corrected_res["energy_corrected"])
        latency_corrected = float(_corrected_res["latency_corrected"])
        corrected_metrics["energy_corrected"] = energy_corrected
        corrected_metrics["latency_corrected"] = latency_corrected
        if f1 > 0 and energy_corrected > 0:
            corrected_metrics["J_per_F1_corrected"] = energy_corrected / f1
            corrected_metrics["F1_per_J_corrected"] = f1 / energy_corrected
        if pred_entities > 0 and energy_corrected > 0:
            corrected_metrics["J_per_entity_corrected"] = energy_corrected / pred_entities
            corrected_metrics["entities_per_J_corrected"] = pred_entities / energy_corrected
        if tp > 0 and energy_corrected > 0:
            corrected_metrics["J_per_TP_corrected"] = energy_corrected / tp
            corrected_metrics["TP_per_J_corrected"] = tp / energy_corrected
            corrected_metrics["TP_per_s_corrected"] = tp / latency_corrected if latency_corrected > 0 else 0.0
            corrected_metrics["s_per_TP_corrected"] = latency_corrected / tp

    # ── Write mean_metrics JSON (used by inference standalone for resume detection) ──
    mean_metrics = {
        "language": language,
        "model": model_name,
        "system_prompt_choice": system_prompt_choice,
        "batch_size": batch_size,
        "f1": ner_metrics["f1"],
        "precision": ner_metrics["precision"],
        "recall": ner_metrics["recall"],
        "n_predicted_entities": ner_metrics["n_predicted_entities"],
        "n_true_entities": ner_metrics["n_true_entities"],
        "n_sentences": n_sentences,
        "avg_gen_tokens_per_sentence": avg_gen_tokens,
        "energy_j_sum": energy_j_sum,
        "latency_s_sum": total_seconds,
        "generation_tokens_sum": total_gen_tokens,
        "prompt_tokens_sum": _agg("prompt_tokens"),
        "job_id": job_id,
        "sweep_param_changed": sweep_param_changed,
        "sweep_param_value": sweep_param_value,
        **efficiency_metrics,
        **corrected_metrics,
    }
    mean_metrics_path = os.path.join(
        artifacts_dir,
        f"mean_metrics_B_{batch_size}_prompt{system_prompt_choice}_{job_id}.json",
    )
    with open(mean_metrics_path, "w", encoding="utf-8") as f:
        json.dump(mean_metrics, f, indent=2, default=numpy_serializer)
    print(f"[INFO] Saved mean_metrics: {mean_metrics_path}")

    # ── Log to MLflow ──────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=run_name):
        # Tags for analysis script lookups
        mlflow.set_tag("language", language)
        mlflow.set_tag("task", "masakha_ner")
        mlflow.set_tag("model", model_name)
        mlflow.set_tag("sweep_param_changed", sweep_param_changed)
        mlflow.set_tag("sweep_param_value", sweep_param_value)
        mlflow.set_tag("system_prompt_choice", str(system_prompt_choice))
        mlflow.set_tag("format_mode", format_mode)
        mlflow.set_tag("job_id", str(job_id))

        # Parameters
        mlflow.log_param("language", language)
        mlflow.log_param("language_full", lang_full)
        mlflow.log_param("model", model_name)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("max_new_tokens", max_new_tokens)
        mlflow.log_param("system_prompt_choice", system_prompt_choice)
        mlflow.log_param("max_concurrency", max_concurrency)
        mlflow.log_param("n_examples", n_sentences)
        mlflow.log_param("sweep_param_changed", sweep_param_changed)
        mlflow.log_param("sweep_param_value", sweep_param_value)
        mlflow.log_param("format_mode", format_mode)

        # Core NER metrics
        mlflow.log_metric("f1", ner_metrics["f1"])
        mlflow.log_metric("precision", ner_metrics["precision"])
        mlflow.log_metric("recall", ner_metrics["recall"])
        mlflow.log_metric("n_predicted_entities", ner_metrics["n_predicted_entities"])
        mlflow.log_metric("n_true_entities", ner_metrics["n_true_entities"])

        # Telemetry metrics
        mlflow.log_metric("energy_j_sum", _agg("energy_j"))
        mlflow.log_metric("latency_s_sum", _agg("latency_s"))
        mlflow.log_metric("generation_tokens_sum", total_gen_tokens)
        mlflow.log_metric("prompt_tokens_sum", _agg("prompt_tokens"))
        mlflow.log_metric("avg_gen_tokens_per_sentence", avg_gen_tokens)

        # Energy-efficiency and corrected metrics
        for k, v in {**efficiency_metrics, **corrected_metrics}.items():
            if isinstance(v, (int, float)) and not (v != v):  # skip NaN
                try:
                    mlflow.log_metric(k, v)
                except Exception:
                    pass

        # Sample prompt artifact
        try:
            sample_ex = test_data[0]
            sample_sentence = " ".join(sample_ex["tokens"])
            sample_msgs = build_masakha_gemma_messages(
                sample_sentence, language, system_prompt_choice, []
            )
            sample_prompt = {
                "prompt_type": "masakha_ner",
                "system_prompt_choice": system_prompt_choice,
                "language": language,
                "language_full": lang_full,
                "system": sample_msgs[0]["content"] if sample_msgs else "",
                "user": sample_msgs[-1]["content"] if sample_msgs else sample_sentence,
            }
            sample_prompt_path = os.path.join(
                per_run_dir,
                f"sample_prompt_prompt{system_prompt_choice}.json",
            )
            with open(sample_prompt_path, "w", encoding="utf-8") as f:
                json.dump(sample_prompt, f, indent=2, ensure_ascii=False)
            mlflow.log_artifact(sample_prompt_path)
        except Exception as _e:
            print(f"[WARN] Could not write sample_prompt: {_e}")

        # Artifacts
        mlflow.log_artifact(mean_metrics_path)
        mlflow.log_artifact(results_path)

        # Telemetry artifact
        telemetry_path = os.path.join(
            per_run_dir,
            "telemetry",
            f"telemetry_B_{batch_size}_prompt{system_prompt_choice}_{job_id}.json",
        )
        with open(telemetry_path, "w", encoding="utf-8") as f:
            json.dump(batch_telemetry, f, indent=2, default=numpy_serializer)
        mlflow.log_artifact(telemetry_path)

        # GPU metrics CSV (DCGM energy data)
        if metrics_csv_path and os.path.isfile(metrics_csv_path):
            try:
                mlflow.log_artifact(metrics_csv_path)
            except Exception as _e:
                print(f"[WARN] Could not log gpu_metrics CSV: {_e}")

        if os.path.exists(artifacts_dir):
            try:
                srv_config_path = os.path.join(artifacts_dir, f"server_config_{job_id}.json")
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
            except Exception:
                pass
        for inf_cfg in glob.glob(os.path.join(artifacts_dir, "*out_inference.json")):
            try:
                mlflow.log_artifact(inf_cfg)
            except Exception:
                pass

        # Classification report as text artifact
        cls_report_path = os.path.join(
            per_run_dir,
            f"cls_report_{language}_prompt{system_prompt_choice}_{job_id}.txt",
        )
        with open(cls_report_path, "w") as f:
            f.write(ner_metrics.get("classification_report", ""))
        mlflow.log_artifact(cls_report_path)

    print(f"[INFO] MLflow run logged: {run_name}")


# =============================================================================
# CLI
# =============================================================================

SUPPORTED_LANGUAGES = sorted(MASAKHANER2_LANGS.keys())

# Languages known to work well with Gemma-3 / Mistral (good web presence in training data)
RECOMMENDED_LANGUAGES = [
    "ewe",
    "hau",
    "ibo",
    "kin",
    "pcm",
    "sna",
    "swa",
    "wol",
    "xho",
    "yor",
    "zul",
    "nya",
    "tsn",
    "lug",
    "twi",
]

# Languages with limited model support (low-resource, may produce poor results)
RISKY_LANGUAGES = ["bam", "fon", "luo", "mos", "bbj"]


def main():
    parser = argparse.ArgumentParser(
        description="MasakhaNER2 NER sweep evaluation with vLLM + energy + MLflow"
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=SUPPORTED_LANGUAGES,
        help=f"MasakhaNER2 language code. Recommended: {', '.join(RECOMMENDED_LANGUAGES)}. "
        f"Risky (low-resource): {', '.join(RISKY_LANGUAGES)}",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["/gemma-3-4b-it", "/gemma-3-12b-it", "/gemma-3-27b-it", "/mistral", "/llama-3-3-70B-it"],
        help="Model identifier",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument(
        "--system-prompt-choice",
        type=int,
        choices=range(1, 9),
        default=1,
        help="System prompt variant 1-8 (all use Template_system_prompt.json with language substitution)",
    )
    parser.add_argument("--max-concurrency", type=int, default=256)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of test examples (default: use full test split)",
    )
    parser.add_argument(
        "--run-start-timestamp",
        type=str,
        default=None,
        help="ISO timestamp for energy CSV alignment (YYYY-MM-DDTHH:MM:SS.ffffff)",
    )

    args = parser.parse_args()

    if args.language in RISKY_LANGUAGES:
        print(
            f"[WARN] Language '{args.language}' ({masakha_lang_name(args.language)}) is low-resource "
            f"and may produce poor NER results with Gemma/Mistral models."
        )

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
