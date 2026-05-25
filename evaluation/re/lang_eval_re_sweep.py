#!/usr/bin/env python3
"""
DocRED Relation Extraction (RE) evaluation with vLLM + energy measurement + MLflow tracking.

Core pipeline: batch inference → telemetry collection → RE evaluation → MLflow logging.
Telemetry utilities (energy scraping, vLLM metrics) live in ../xtreme/utils.py.

Dataset: DocRED (thunlp/docred on HuggingFace, MIT license)
  - Document-level RE from Wikipedia + Wikidata
  - 96 Wikidata relation types (Wikidata property IDs P6–P937)
  - Splits: train_annotated=3,053 / validation=998 / test=1,000 (test has no labels)
  - Loaded via huggingface_hub.hf_hub_download (avoids deprecated loading-script API)

Task: given a full document and all named entities in it, extract all factual
(head_entity, relation, tail_entity) triples that are supported by the text.
One inference call per document; entities are pre-listed from the gold annotation.

Output format (JSON array):
  [{"head": "Skai TV", "relation": "country", "tail": "Greece"}, ...]

Evaluation metrics (micro-averaged over all triples):
  - F1 / Precision / Recall: standard micro-F1 over (head, relation, tail) triples
  - Ign F1: same but excluding triples that appear in the training set
    (standard DocRED metric; isolates inter-document generalisation)
"""

import argparse
import asyncio
import datetime
import glob
import gzip
import json
import os
import re
import sys
import time

import aiohttp
import mlflow
import numpy as np
import pandas as pd
from aiohttp import ClientTimeout, TCPConnector
from huggingface_hub import hf_hub_download
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
# DocRED Dataset Loading
# =====================================================================

# Map CLI split name → HuggingFace DocRED file name
_DOCRED_SPLIT_FILES = {
    "validation": "data/dev.json.gz",
    "dev": "data/dev.json.gz",  # alias
    "test": "data/test.json.gz",
}

_DOCRED_REPO_ID = "thunlp/docred"
_REL_INFO_FILE = "data/rel_info.json.gz"

# Maximum document word count before truncation.
# DocRED averages ~200 words; truncating to 220 keeps 90 %+ of documents intact
# while protecting the 1024-token context budget.
_MAX_DOC_WORDS = 220


def build_valid_relations_str(rel_info: dict) -> str:
    """Return a comma-separated string of all DocRED relation names.

    Relations are sorted alphabetically so the injected list is deterministic
    regardless of dict insertion order.
    """
    return ", ".join(sorted(rel_info.values()))


def _hf_download(filename: str) -> str:
    """Download a DocRED file from HuggingFace (uses local cache if present)."""
    return hf_hub_download(repo_id=_DOCRED_REPO_ID, filename=filename, repo_type="dataset")


def load_rel_info() -> dict:
    """Return {Wikidata_PID: relation_text} mapping (96 entries)."""
    path = _hf_download(_REL_INFO_FILE)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_docred_split(split: str) -> list:
    """Load a DocRED split.  Returns list of document dicts.

    Each document has:
      - 'sents': list of sentences, each a list of word strings
      - 'vertexSet': list of entity clusters (each cluster is a list of mention dicts)
      - 'labels': list of {h, t, r} dicts (h/t = entity indices, r = Wikidata PID)
                  NB: 'labels' may be absent or empty for the test split.
      - 'title': string
    """
    fname = _DOCRED_SPLIT_FILES.get(split)
    if fname is None:
        raise ValueError(f"Unknown split '{split}'. Choose from: {list(_DOCRED_SPLIT_FILES)}")
    path = _hf_download(fname)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# =====================================================================
# DocRED Example Utilities
# =====================================================================


def build_doc_text(doc: dict, max_words: int = _MAX_DOC_WORDS) -> str:
    """Concatenate all sentences into a single space-separated string.

    Truncates to *max_words* words to stay within the context budget.
    Truncation happens at the sentence boundary just before the limit.
    """
    words_so_far = 0
    kept_sentences = []
    for sent in doc["sents"]:
        if words_so_far + len(sent) > max_words:
            break
        kept_sentences.append(sent)
        words_so_far += len(sent)
    if not kept_sentences:
        # Fall back: include at least the first sentence (avoids empty prompts)
        kept_sentences = doc["sents"][:1]
    return " ".join(" ".join(s) for s in kept_sentences)


def get_entity_canonical_names(doc: dict) -> list:
    """Return the canonical (first-mention) name for each entity cluster."""
    return [cluster[0]["name"] for cluster in doc["vertexSet"]]


def get_doc_gold_triples(doc: dict, rel_info: dict) -> set:
    """Return gold (head_name, relation_text, tail_name) triples for this document.

    Uses the first mention (index 0) of each entity cluster as the canonical name.
    Returns empty set if the document has no 'labels' key (e.g. test split).
    """
    triples = set()
    for lbl in doc.get("labels", []):
        h_name = doc["vertexSet"][lbl["h"]][0]["name"]
        t_name = doc["vertexSet"][lbl["t"]][0]["name"]
        rel_text = rel_info[lbl["r"]]
        triples.add((_normalise(h_name), _normalise(rel_text), _normalise(t_name)))
    return triples


def build_training_triples_set(rel_info: dict) -> set:
    """Build the set of all (head, relation, tail) triples from DocRED train_annotated.

    Used to compute Ign F1: triples in this set are excluded when evaluating
    the validation/test split, following the standard DocRED evaluation protocol.
    """
    path = _hf_download("data/train_annotated.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        train = json.load(f)
    triples: set = set()
    for doc in train:
        for lbl in doc.get("labels", []):
            h_name = doc["vertexSet"][lbl["h"]][0]["name"]
            t_name = doc["vertexSet"][lbl["t"]][0]["name"]
            rel_text = rel_info[lbl["r"]]
            triples.add((_normalise(h_name), _normalise(rel_text), _normalise(t_name)))
    return triples


# =====================================================================
# Prompt Configuration (loaded from file; mirrors EE / NER prompt files)
# =====================================================================

_RE_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "prompts_in_all_languages")
_RE_SYSTEM_PROMPTS_FILE = os.path.join(_RE_PROMPTS_DIR, "re_system_prompts.json")
_RE_FEW_SHOTS_FILE = os.path.join(_RE_PROMPTS_DIR, "re_few_shots.json")

# Number of few-shot examples per prompt style (mirrors EE EE_SHOTS_MAP)
RE_SHOTS_MAP = {1: 0, 2: 1, 3: 0, 4: 1, 5: 2, 6: 2, 7: 0, 8: 0}


def load_re_system_prompt(choice: int, valid_relations_str: str = None) -> str:
    """Load system prompt text for a given choice from re_system_prompts.json.

    If the template contains the literal placeholder ``{valid_relations}`` and
    *valid_relations_str* is provided, the placeholder is replaced at runtime.
    This allows prompts 3, 7, and 8 to stay clean in the JSON file while
    receiving the full DocRED relation list dynamically.
    """
    with open(_RE_SYSTEM_PROMPTS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    key = str(choice)
    if key not in data:
        raise ValueError(f"No RE system prompt found for choice: {choice}")
    val = data[key]
    text = val if isinstance(val, str) else "".join(val)
    if valid_relations_str and "{valid_relations}" in text:
        text = text.replace("{valid_relations}", valid_relations_str)
    return text


def load_re_few_shots() -> dict:
    """Load RE few-shot examples from re_few_shots.json.

    Returns dict with keys 'shot1', 'shot2', each with 'user' and 'assistant' strings.
    """
    with open(_RE_FEW_SHOTS_FILE, encoding="utf-8") as f:
        return json.load(f)


def build_re_user_message(document_text: str, entity_names: list) -> str:
    """Construct the user turn for a single RE document."""
    entities_str = ", ".join(entity_names)
    return (
        f"Document:\n{document_text}\n\n"
        f"Entities: {entities_str}\n\n"
        "Extract all relation triples as a JSON array."
    )


def build_re_messages(
    document_text: str,
    entity_names: list,
    system_prompt_choice: int = 1,
    rel_info: dict = None,
) -> list:
    """Build the full chat message list for a single RE document.

    For prompts with few-shot examples (2, 4, 5, 6), prepends the appropriate
    number of user/assistant shot pairs from re_few_shots.json before the query.

    If *rel_info* is provided and the selected prompt template contains the
    ``{valid_relations}`` placeholder (prompts 3, 7, 8), the full DocRED
    relation list is injected at runtime.
    """
    valid_relations_str = build_valid_relations_str(rel_info) if rel_info else None
    system_content = load_re_system_prompt(system_prompt_choice, valid_relations_str)
    user_content = build_re_user_message(document_text, entity_names)
    messages = [{"role": "system", "content": system_content}]
    n_shots = RE_SHOTS_MAP.get(system_prompt_choice, 0)
    if n_shots > 0:
        few_shots = load_re_few_shots()
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

# Matches a JSON array anywhere in the model output (including markdown fences)
_JSON_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)


def _normalise(text: str) -> str:
    """Lower-case and collapse whitespace for lenient matching."""
    return " ".join(text.lower().split())


def parse_re_response(text: str) -> list:
    """Parse model response to a list of (head, relation, tail) normalised string triples.

    Strategy:
    1. Strip markdown fences and try to parse the whole text as a JSON array.
    2. Extract the first [...] block and parse it.
    3. Return empty list on failure.

    Each valid object must have 'head', 'relation', 'tail' string fields.
    """
    if not text:
        return []
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()

    def _extract_triples(obj):
        """Convert a parsed JSON object to (head, relation, tail) tuple or None."""
        if not isinstance(obj, dict):
            return None
        h = obj.get("head", "")
        r = obj.get("relation", "")
        t = obj.get("tail", "")
        if h and r and t:
            return (_normalise(str(h)), _normalise(str(r)), _normalise(str(t)))
        return None

    # Try full parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            result = [_extract_triples(o) for o in parsed]
            return [x for x in result if x is not None]
        if isinstance(parsed, dict):
            triple = _extract_triples(parsed)
            return [triple] if triple else []
    except json.JSONDecodeError:
        pass

    # Try to extract first [...] block
    m = _JSON_ARRAY_RE.search(cleaned)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                result = [_extract_triples(o) for o in parsed]
                return [x for x in result if x is not None]
        except json.JSONDecodeError:
            pass

    # Partial-JSON recovery: model output may be truncated mid-array.
    # Find the last complete JSON object and close the array after it.
    array_start = cleaned.find("[")
    if array_start != -1:
        fragment = cleaned[array_start:]
        # Find last position where a complete object ends ("}" or "},")
        last_close = max(fragment.rfind("},"), fragment.rfind("}"))
        if last_close != -1:
            partial = fragment[: last_close + 1] + "]"
            try:
                parsed = json.loads(partial)
                if isinstance(parsed, list):
                    result = [_extract_triples(o) for o in parsed]
                    return [x for x in result if x is not None]
            except json.JSONDecodeError:
                pass

    return []


# =====================================================================
# RE Evaluation Metrics
# =====================================================================


def compute_re_metrics(gold_list: list, pred_list: list, training_triples: set = None) -> dict:
    """Compute micro-F1 and Ign F1 over DocRED RE triples.

    Args:
        gold_list: per-document set of (head_norm, relation_norm, tail_norm) gold triples
        pred_list: per-document list of (head_norm, relation_norm, tail_norm) predicted triples
        training_triples: set of all triples from train_annotated (for Ign F1 computation)

    Standard F1:   TP / (TP+FP) and TP / (TP+FN) over all triples
    Ign F1:        same, but triples in training_triples are excluded from
                   both gold and predicted sets before counting (DocRED convention)
    """
    tp = fp = fn = 0
    tp_ign = fp_ign = fn_ign = 0

    for gold_set, pred_items in zip(gold_list, pred_list):
        pred_set = set(pred_items)

        # Standard micro-F1
        for p in pred_set:
            if p in gold_set:
                tp += 1
            else:
                fp += 1
        for g in gold_set:
            if g not in pred_set:
                fn += 1

        # Ign F1 (exclude training-set triples)
        if training_triples is not None:
            gold_ign = gold_set - training_triples
            pred_ign = pred_set - training_triples
            for p in pred_ign:
                if p in gold_ign:
                    tp_ign += 1
                else:
                    fp_ign += 1
            for g in gold_ign:
                if g not in pred_ign:
                    fn_ign += 1

    def _f1(tp_val, fp_val, fn_val):
        prec = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0.0
        rec = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    prec, rec, f1 = _f1(tp, fp, fn)
    prec_ign, rec_ign, f1_ign = _f1(tp_ign, fp_ign, fn_ign)

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "ign_precision": prec_ign,
        "ign_recall": rec_ign,
        "ign_f1": f1_ign,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_gold": tp + fn,
        "total_pred": tp + fp,
        "tp_ign": tp_ign,
        "total_gold_ign": tp_ign + fn_ign,
        "total_pred_ign": tp_ign + fp_ign,
    }


# =====================================================================
# Batch Inference (chat-style model)
# =====================================================================


async def process_batch_chat(
    session,
    docs: list,
    rel_info: dict,
    max_tokens: int,
    model_name: str,
    system_prompt_choice: int = 1,
    semaphore=None,
):
    """Send a batch of DocRED documents to a chat-style model in parallel.

    Each document produces one chat completion request:
      - System message: RE task instruction (from re_system_prompts.json)
      - Few-shot turns: 0-2 user/assistant pairs depending on prompt style
      - User message: document text + entity list

    Returns:
        list of response JSON objects (one per document), or Exception on failure
    """

    async def single_request(doc: dict):
        doc_text = build_doc_text(doc)
        entity_names = get_entity_canonical_names(doc)
        messages = build_re_messages(doc_text, entity_names, system_prompt_choice, rel_info)
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        async with semaphore, session.post(CHAT_URL, json=payload) as resp:
            return await resp.json()

    tasks = [single_request(doc) for doc in docs]
    return await asyncio.gather(*tasks, return_exceptions=True)


# =====================================================================
# Telemetry-Wrapped Batch Processing (identical structure to EE/NER scripts)
# =====================================================================


async def process_and_measure(
    session,
    docs: list,
    rel_info: dict,
    max_tokens: int,
    model_name: str,
    system_prompt_choice: int = 1,
    semaphore=None,
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
        docs,
        rel_info,
        max_tokens,
        model_name,
        system_prompt_choice,
        semaphore=semaphore,
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

    denom_prefill = prefill_count_diff if prefill_count_diff > 0 else max(len(docs), 1)
    denom_inference = inference_count_diff if inference_count_diff > 0 else max(len(docs), 1)
    denom_decode = decode_count_diff if decode_count_diff > 0 else max(len(docs), 1)

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

    total_processing_time = max(e2e_latency_mean * len(docs), 1e-9)
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
        "batch_size": len(docs),
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

    if len(docs) >= 32:
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
# RE Evaluation Pipeline: Core Loop
# =====================================================================


async def evaluate_re_pipeline_docred(
    dataset: list,
    rel_info: dict,
    training_triples: set,
    batch_size: int,
    model_name: str,
    max_new_tokens: int = 300,
    system_prompt_choice: int = 1,
    max_concurrency: int = 256,
):
    """Run end-to-end RE evaluation on DocRED documents.

    Process flow for each batch:
      1. Send batch to model via process_and_measure() (energy + latency)
      2. Parse model responses → list of (head, relation, tail) triples per doc
      3. Accumulate gold and pred triple sets
    After all batches:
      4. Compute micro-F1 and Ign F1

    Returns:
        (re_metrics, generated_results, batch_telemetry)
    """
    all_gold: list = []  # per-document set of (h, r, t) gold triples
    all_pred: list = []  # per-document list of (h, r, t) predicted triples
    generated_results: list = []
    batch_telemetry: list = []

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
                    rel_info,
                    max_new_tokens,
                    model_name,
                    system_prompt_choice,
                    semaphore=semaphore,
                )
                batch_telemetry.append(telemetry)

                for doc, response in zip(batch, responses):
                    gold_triples = get_doc_gold_triples(doc, rel_info)
                    entity_names = get_entity_canonical_names(doc)

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

                    pred_triples = parse_re_response(text)

                    # Debug: show first few examples
                    if len(generated_results) < 3:
                        print(f"\nDEBUG example {len(generated_results)}:")
                        print(f"  title: {doc.get('title', '?')!r}")
                        print(f"  entities[:5]: {entity_names[:5]}")
                        print(f"  gold triples[:3]: {list(gold_triples)[:3]}")
                        print(f"  raw output[:200]: {text[:200]}")
                        print(f"  parsed pred[:3]: {pred_triples[:3]}")

                    generated_results.append(
                        {
                            "title": doc.get("title", ""),
                            "n_entities": len(entity_names),
                            "gold_triples": [list(t) for t in gold_triples],
                            "raw_output": text,
                            "pred_triples": [list(t) for t in pred_triples],
                        }
                    )
                    all_gold.append(gold_triples)
                    all_pred.append(pred_triples)

            except Exception as e:
                print(f"Batch {i // batch_size} failed: {e}")
                for doc in batch:
                    all_gold.append(get_doc_gold_triples(doc, rel_info))
                    all_pred.append([])
                continue

    if all_gold:
        re_metrics = compute_re_metrics(all_gold, all_pred, training_triples)
    else:
        re_metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "ign_precision": 0.0,
            "ign_recall": 0.0,
            "ign_f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "total_gold": 0,
            "total_pred": 0,
            "tp_ign": 0,
            "total_gold_ign": 0,
            "total_pred_ign": 0,
        }

    return re_metrics, generated_results, batch_telemetry


# =====================================================================
# MLflow Orchestration: Main Experiment Runner
# =====================================================================


async def run(args):
    # ===== STEP 1: Load Dataset =====
    print(f"Loading DocRED split '{args.split}' ...")
    dataset = load_docred_split(args.split)
    rel_info = load_rel_info()

    if args.limit:
        dataset = dataset[: args.limit]
    print(f"Loaded {len(dataset)} DocRED documents (split={args.split})")

    # ===== STEP 1b: Build training triples for Ign F1 =====
    print("Building training triples set for Ign F1 ...")
    training_triples = build_training_triples_set(rel_info)
    print(f"[INFO] Training triples: {len(training_triples):,} unique (h, r, t) tuples")

    # ===== STEP 2: Setup Output Directories =====
    out_dir = os.environ.get("ARTIFACTS_DIR")
    if not out_dir:
        model_clean = args.model.lstrip("/").replace("/", "_")
        out_dir = (
            f"./inference_eval_artifacts/re/"
            f"re_docred_{args.split}_B{args.batch_size}_{model_clean}_prompt{args.system_prompt_choice}_{job_id or 'local'}"
        )

    os.makedirs(os.path.join(out_dir, "generated_responses"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "re_metrics"), exist_ok=True)
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
        run_name=f"docred_re_updated_prompts_b_{args.batch_size}_{args.split}_{prompt_label}{timestamp_suffix}_{job_id}"
    ):
        # ===== Log Configuration to MLflow =====
        tags = {
            "split": args.split,
            "batch_size": str(args.batch_size),
            "model": args.model,
            "dataset": "docred",
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
            "dataset": "docred",
            "n_samples": str(len(dataset)),
            "system_prompt_choice": args.system_prompt_choice,
            "n_training_triples": len(training_triples),
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
        sample_doc = dataset[0]
        sample_doc_text = build_doc_text(sample_doc)
        sample_entities = get_entity_canonical_names(sample_doc)
        sample_prompt = {
            "prompt_type": "re_chat",
            "system_prompt_choice": args.system_prompt_choice,
            "n_shots": RE_SHOTS_MAP.get(args.system_prompt_choice, 0),
            "system": load_re_system_prompt(
                args.system_prompt_choice,
                build_valid_relations_str(rel_info),
            ),
            "user": build_re_user_message(sample_doc_text, sample_entities),
        }
        sample_prompt_path = os.path.join(out_dir, f"sample_prompt_{prompt_label}.json")
        with open(sample_prompt_path, "w") as f:
            json.dump(sample_prompt, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(sample_prompt_path)

        # ===== STEP 4: Run RE Evaluation =====
        re_metrics, gen_responses, batch_telemetry = await evaluate_re_pipeline_docred(
            dataset,
            rel_info,
            training_triples,
            batch_size=args.batch_size,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            system_prompt_choice=args.system_prompt_choice,
            max_concurrency=args.max_concurrency,
        )

        # ===== STEP 5: Save Results to JSON Files =====
        responses_path = os.path.join(
            out_dir,
            "generated_responses",
            f"responses_B{args.batch_size}_{prompt_label}_{job_id}.json",
        )
        re_metrics_path = os.path.join(
            out_dir,
            "re_metrics",
            f"re_metrics_B_{args.batch_size}_{prompt_label}_{job_id}.json",
        )
        telemetry_path = os.path.join(
            out_dir,
            "telemetry",
            f"telemetry_B_{args.batch_size}_{prompt_label}_{job_id}.json",
        )

        with open(responses_path, "w") as f:
            json.dump(gen_responses, f, indent=2, default=numpy_serializer, ensure_ascii=False)
        with open(re_metrics_path, "w") as f:
            json.dump(re_metrics, f, indent=2, default=numpy_serializer, ensure_ascii=False)
        with open(telemetry_path, "w") as f:
            json.dump(batch_telemetry, f, indent=2, default=numpy_serializer, ensure_ascii=False)

        mlflow.log_artifact(responses_path)
        mlflow.log_artifact(re_metrics_path)
        mlflow.log_artifact(telemetry_path)

        # ===== STEP 6: Log RE Metrics to MLflow =====
        mlflow.log_metric("re_precision", float(re_metrics.get("precision", 0.0)))
        mlflow.log_metric("re_recall", float(re_metrics.get("recall", 0.0)))
        mlflow.log_metric("re_f1", float(re_metrics.get("f1", 0.0)))
        mlflow.log_metric("re_ign_precision", float(re_metrics.get("ign_precision", 0.0)))
        mlflow.log_metric("re_ign_recall", float(re_metrics.get("ign_recall", 0.0)))
        mlflow.log_metric("re_ign_f1", float(re_metrics.get("ign_f1", 0.0)))

        # ===== STEP 7: Aggregate Telemetry and Efficiency Metrics =====
        mean_metrics: dict = {}

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

            re_f1 = float(re_metrics.get("f1", 0.0))
            re_ign_f1 = float(re_metrics.get("ign_f1", 0.0))
            total_pred = int(re_metrics.get("total_pred", 0))
            tp = int(re_metrics.get("tp", 0))
            total_seconds = float(sum(t.get("latency_s", 0.0) for t in batch_telemetry))

            mean_metrics.update(
                {
                    "re_f1": re_f1,
                    "re_ign_f1": re_ign_f1,
                    "re_precision": float(re_metrics.get("precision", 0.0)),
                    "re_recall": float(re_metrics.get("recall", 0.0)),
                    "re_ign_precision": float(re_metrics.get("ign_precision", 0.0)),
                    "re_ign_recall": float(re_metrics.get("ign_recall", 0.0)),
                    "whole_energy": energy_j_sum,
                    "total_seconds": total_seconds,
                    "total_pred_triples": total_pred,
                    "total_gold_triples": int(re_metrics.get("total_gold", 0)),
                    "tp_triples": tp,
                }
            )

            # Energy-efficiency metrics (F1 = standard micro-F1)
            if re_f1 > 0 and energy_j_sum > 0:
                mean_metrics["J_per_F1"] = energy_j_sum / re_f1
                mean_metrics["F1_per_J"] = re_f1 / energy_j_sum

            if total_pred > 0 and energy_j_sum > 0:
                mean_metrics["J_per_triple"] = energy_j_sum / total_pred
                mean_metrics["triples_per_J"] = total_pred / energy_j_sum

            if tp > 0 and energy_j_sum > 0:
                mean_metrics["J_per_TP"] = energy_j_sum / tp
                mean_metrics["TP_per_J"] = tp / energy_j_sum
                mean_metrics["TP_per_s"] = tp / total_seconds if total_seconds > 0 else 0.0
                mean_metrics["s_per_TP"] = total_seconds / tp if total_seconds > 0 else 0.0

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
            f"Completed: F1={re_metrics.get('f1', 0.0):.4f}, "
            f"Ign F1={re_metrics.get('ign_f1', 0.0):.4f}, "
            f"Precision={re_metrics.get('precision', 0.0):.4f}, "
            f"Recall={re_metrics.get('recall', 0.0):.4f}, "
            f"Energy Sum={mean_metrics.get('energy_j_sum', 0.0):.4f} J"
        )


# =====================================================================
# Command-Line Interface
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="DocRED RE evaluation with vLLM + energy measurement + MLflow"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "dev", "test"],
        help=(
            "Which DocRED split to evaluate. "
            "'validation'/'dev' = dev.json.gz (998 docs, with labels). "
            "'test' = test.json.gz (1000 docs, no gold labels → F1 will be 0). "
            "Default: validation."
        ),
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
        help="Number of documents per batch.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=300,
        help=(
            "Maximum tokens to generate per document. "
            "RE output is a JSON array of triples; 300 covers ~15 triples."
        ),
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
        choices=range(1, 9),
        default=1,
        help="Which system prompt variant to use (1–8).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Sample only first N documents (for quick testing).",
    )
    parser.add_argument(
        "--run-start-timestamp",
        type=str,
        default=None,
        help="ISO timestamp for energy window start. Format: YYYY-MM-DDTHH:MM:SS.ssssss",
    )

    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
