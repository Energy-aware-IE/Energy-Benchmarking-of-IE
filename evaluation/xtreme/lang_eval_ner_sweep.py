#!/usr/bin/env python3
"""
XTREME NER evaluation with vLLM + energy measurement + MLflow tracking.

Core pipeline: batch inference → telemetry collection → NER evaluation → MLflow logging.
Utility functions (prompt building, parsing, telemetry scraping, etc.) live in utils.py.
"""

import argparse
import asyncio
import contextlib
import datetime
import glob
import json
import os
import subprocess
import time

import aiohttp
import mlflow
import numpy as np
import pandas as pd
from aiohttp import ClientTimeout, TCPConnector
from datasets import load_dataset
from dspy_ner import create_ner_program, setup_dspy
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm
from utils import (
    CHAT_URL,
    COMPLETIONS_URL,
    MLFLOW_TRACKING_URI,
    analyze_metrics_csv,
    build_artifacts_dir,
    build_gemma_messages,
    build_gollie_prompt,
    compute_energy_corrected,
    create_sample_prompt,
    full_lang_name,
    get_bio_tags_language_aware,
    job_id,
    load_few_shots,
    make_messages_for,
    metrics_csv_path,
    numpy_serializer,
    parse_gollie_output_to_entities,
    parse_response_json_like,
    read_energy_joules,
    read_vllm_metrics,
    resolve_xtreme_subset,
    robust_gemma_entity_extraction,
)

print(CHAT_URL)
print(job_id)


def load_dspy_instructions(program_path: str):
    """Extract the optimized instruction string and field descriptions from a saved DSPy program JSON."""
    try:
        with open(program_path) as f:
            data = json.load(f)
        instructions = data["predict"]["signature"]["instructions"]
        fields = data["predict"]["signature"].get("fields", [])

        # Append output field descriptions (skip the first field which is the input "Sentence:")
        output_fields = [f for f in fields if f.get("prefix", "").rstrip(":").lower() != "sentence"]
        if output_fields:
            field_lines = "\n\nOutput fields:"
            for field in output_fields:
                prefix = field.get("prefix", "").rstrip(":")
                desc = field.get("description", "")
                field_lines += f"\n- {prefix}: {desc}"
            instructions += field_lines

        return instructions
    except Exception as e:
        print(f"[WARNING] Could not load DSPy instructions from {program_path}: {e}")
        return None


# =====================================================================
# Batch Inference: Two model-specific functions
# =====================================================================
# These functions handle different model input formats:
# - Chat models (Gemma, Mistral): use /v1/chat/completions endpoint with message format
# - Completion models (GoLLIE): use /v1/completions endpoint with prompt text format
# Both use async/await for parallel processing of entire batches simultaneously.


async def process_batch_chat(
    session,
    sentences,
    max_tokens,
    model_name,
    language,
    system_prompt_choice=None,
    use_generic_template=False,
    custom_system_prompt=None,
    semaphore=None,
):
    """
    Send batch of sentences to chat-style model (Gemma, Mistral) in parallel.

    For each sentence, constructs a message with:
    - System message: NER task instruction (from utils.build_system_msg or load_gemma_system_template)
    - Few-shot examples: language-specific examples (from utils.load_few_shots)
    - User message: the sentence to analyze

    Uses asyncio.gather() to send all sentences simultaneously to /v1/chat/completions endpoint.
    This enables parallel inference → much faster than serial processing.

    Args:
        session: aiohttp ClientSession for HTTP requests
        sentences: list of input sentences to process
        max_tokens: max tokens to generate per sentence
        model_name: /gemma-3-4b-it, /gemma-3-12b-it, or /mistral
        language: language code (en, de, zh, ar, etc.) for language-specific prompting
        system_prompt_choice: which system prompt variant to use (1-8) for Gemma models
        use_generic_template: whether to use language-agnostic prompt template

    Returns:
        list of responses from model (one per sentence), or Exception if request fails

    See also: utils.build_gemma_messages(), utils.load_few_shots()
    """

    async def single_request(sentence):
        """Process one sentence: build messages and send to chat endpoint."""
        # Load few-shot examples for this language (helps model understand task)
        few_shots = load_few_shots(language)

        # Choose prompt builder based on model and whether system_prompt_choice is set
        if custom_system_prompt is not None:
            # Style 9: use the optimized instructions extracted from DSPy JSON.
            # Override output format to JSON so robust_gemma_entity_extraction can parse it.
            json_format_override = (
                '\n\nReturn your answer as a JSON object with keys "PER", "ORG", "LOC", '
                "each mapping to a list of entity mention strings exactly as they appear in the sentence. "
                'Example: {"PER": ["John"], "ORG": ["Microsoft"], "LOC": ["Berlin"]}'
            )
            messages = [
                {"role": "system", "content": custom_system_prompt + json_format_override},
                {"role": "user", "content": sentence},
            ]
        elif (
            model_name in ["/gemma-3-4b-it", "/gemma-3-12b-it", "/gemma-3-27b-it", "/mistral", "/llama-3-3-70B-it"]
            and system_prompt_choice is not None
        ):
            # Gemma/Llama: use language-specific system prompt with few-shot examples
            messages = build_gemma_messages(
                sentence,
                language,
                system_prompt_choice,
                few_shots,
                use_generic_template=use_generic_template,
            )
        else:
            # Fallback: use generic NER system prompt with few-shot examples
            messages = make_messages_for(sentence, language, few_shots)

        # Construct OpenAI-compatible chat completion payload
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,  # Deterministic: no randomness
            "max_tokens": max_tokens,
            "stop": ["}"],  # Stop at closing brace (end of JSON)
        }

        # Send request to chat endpoint and return response JSON
        # Semaphore limits concurrent HTTP requests to avoid TCP flooding
        async with semaphore, session.post(CHAT_URL, json=payload) as resp:
            result = await resp.json()
            return result

    # Create async tasks for all sentences (one per sentence)
    tasks = [single_request(s) for s in sentences]

    # Execute all tasks in parallel using asyncio.gather()
    # return_exceptions=True: if one fails, continue with others
    return await asyncio.gather(*tasks, return_exceptions=True)


async def process_batch_completion(
    session,
    sentences,
    max_tokens,
    model_name,
    language,
    system_prompt_choice=None,
    use_generic_template=False,
    semaphore=None,
):
    """
    Send batch of sentences to completion-style model (GoLLIE) in parallel.

    GoLLIE models use a different format: they complete code-like prompts instead of
    responding to chat messages. Each prompt includes:
    - Language header: defines the task in structured pseudo-code format
    - Text literal: the sentence to analyze in JSON string format
    - Result prefix: 'result = [' to guide model toward structured output

    Uses asyncio.gather() to send all sentences simultaneously to /v1/completions endpoint.

    Args:
        session: aiohttp ClientSession for HTTP requests
        sentences: list of input sentences to process
        max_tokens: max tokens to generate per sentence
        model_name: /gollie
        language: language code for language-specific prompting
        system_prompt_choice, use_generic_template: not used for GoLLIE (included for compatibility)

    Returns:
        list of responses from model (one per sentence), or Exception if request fails

    See also: utils.build_gollie_prompt(), utils.LANG2HEADER
    """

    async def single_request(sentence):
        """Process one sentence: build GoLLIE prompt and send to completion endpoint."""
        # Build language-specific prompt with structured format
        prompt = build_gollie_prompt(sentence, language, use_generic_template)

        # Construct OpenAI-compatible completion payload
        payload = {
            "model": model_name,
            "prompt": prompt,
            "temperature": 0.0,  # Deterministic
            "max_tokens": max_tokens,
            "stop": ["]", "\n"],  # Stop at end of list or newline
        }

        # Send request to completion endpoint with 60s timeout
        # Semaphore limits concurrent HTTP requests to avoid TCP flooding
        async with (
            semaphore,
            session.post(
                COMPLETIONS_URL, json=payload, timeout=aiohttp.ClientTimeout(total=60)
            ) as resp,
        ):
            # Handle HTTP errors gracefully
            if resp.status != 200:
                return {"error": f"HTTP {resp.status}", "text": await resp.text()}
            return await resp.json()

    # Create async tasks for all sentences
    tasks = [single_request(s) for s in sentences]

    # Execute all tasks in parallel
    return await asyncio.gather(*tasks, return_exceptions=True)


# =====================================================================
# Telemetry-Wrapped Batch Processing
# =====================================================================
# This is the core measurement function: measures energy consumption and performance metrics
# around batch inference.
#
# Measurement pattern:
#   1. BEFORE: read GPU energy & vLLM metrics
#   2. SEND: batch to model (via process_batch_chat or process_batch_completion)
#   3. AFTER: read GPU energy & vLLM metrics again
#   4. CALCULATE: differences to get Joules, tokens, latency
#   5. DERIVE: higher-level metrics (throughput, energy per token, power draw, etc.)
#
# This allows answering: "How much energy was used for this NER batch?"


async def process_and_measure(
    session,
    prompts,
    max_tokens,
    model_name,
    language,
    system_prompt_choice=None,
    use_generic_template=False,
    custom_system_prompt=None,
    semaphore=None,
):
    """
    Send batch to model while measuring energy and performance metrics.

    Key measurements:
    - Energy (joules): from GPU via DCGM metrics
    - Latency (seconds): wall-clock time for batch
    - Token counts: prompt tokens + generation tokens
    - Throughput: tokens per second
    - Power draw: joules per second
    - Prefill vs. generation breakdown: energy split by inference phase

    Returns:
        tuple of (responses, telemetry_dict)
        - responses: raw model outputs (one per sentence)
        - telemetry_dict: 20+ metrics about energy, tokens, latency, throughput

    See also: utils.read_energy_joules(), utils.read_vllm_metrics()
    """
    # ===== BEFORE: Capture baseline measurements =====
    e0 = read_energy_joules()  # GPU energy (joules)
    v0 = read_vllm_metrics()  # vLLM metrics dict (14 metrics)
    t0 = time.perf_counter()  # High-precision wall-clock time
    t0_epoch = time.time()  # Unix epoch time (for CSV alignment)
    print("Before batch request: ", datetime.datetime.now())

    # ===== SEND: Choose model type and execute batch inference =====
    if model_name in ["/mistral", "/gemma-3-4b-it", "/gemma-3-12b-it", "/gemma-3-27b-it", "/llama-3-3-70B-it"]:
        # Chat-style models: send batch to chat completions endpoint
        responses = await process_batch_chat(
            session,
            prompts,
            max_tokens,
            model_name,
            language,
            system_prompt_choice,
            use_generic_template,
            custom_system_prompt,
            semaphore=semaphore,
        )
    else:  # GoLLIE
        # Completion-style models: send batch to completions endpoint
        responses = await process_batch_completion(
            session,
            prompts,
            max_tokens,
            model_name,
            language,
            use_generic_template=use_generic_template,
            semaphore=semaphore,
        )

    # ===== AFTER: Capture final measurements =====
    t1 = time.perf_counter()  # Wall-clock time
    t1_epoch = time.time()  # Unix epoch time
    print("After batch request: ", datetime.datetime.now())
    e1 = read_energy_joules()  # GPU energy (joules)
    v1 = read_vllm_metrics()  # vLLM metrics dict

    # ===== CALCULATE: Compute basic differences =====
    joules = max(e1 - e0, 0.0)  # Energy consumed in joules
    latency = t1 - t0  # Wall-clock latency in seconds

    # Token count deltas (vLLM tracks prompt tokens and generation tokens separately)
    # Clamped to 0: if read_vllm_metrics() failed for v1, all values are 0 → diff would be negative
    prompt_t = max(v1["prompt_tokens_total"] - v0["prompt_tokens_total"], 0)
    gen_t = max(v1["generation_tokens_total"] - v0["generation_tokens_total"], 0)
    total_t = prompt_t + gen_t

    # End-to-end latency per request (average across batch)
    e2e_count_diff = max(v1["e2e_latency_count"] - v0["e2e_latency_count"], 0)
    e2e_latency_mean = max(
        (v1["e2e_latency_sum"] - v0["e2e_latency_sum"]) / max(e2e_count_diff, 1), 0.0
    )

    # Time to first token (TTFT): latency before first token generated
    ttft_count_diff = v1["time_to_first_count"] - v0["time_to_first_count"]
    ttft_mean = (v1["time_to_first_sum"] - v0["time_to_first_sum"]) / max(ttft_count_diff, 1)

    # Token throughput: tokens generated per second during decode phase
    tpt_count_diff = v1["time_per_token_count"] - v0["time_per_token_count"]
    tpt_sum = v1["time_per_token_sum"] - v0["time_per_token_sum"]
    token_throughput = (tpt_count_diff / tpt_sum) if tpt_sum > 0 else 0.0
    time_per_token_mean = tpt_sum / max(tpt_count_diff, 1)

    # Inference phases (vLLM tracks time breakdown):
    #   - Prefill: processing prompt tokens (once per batch)
    #   - Inference: iterative token generation
    #   - Decode: converting logits to tokens
    prefill_total = v1["request_prefill_time_sum"] - v0["request_prefill_time_sum"]
    inference_total = v1["request_inference_time_sum"] - v0["request_inference_time_sum"]
    decode_total = v1["request_decode_time_sum"] - v0["request_decode_time_sum"]

    prefill_count_diff = v1["request_prefill_time_count"] - v0["request_prefill_time_count"]
    inference_count_diff = v1["request_inference_time_count"] - v0["request_inference_time_count"]
    decode_count_diff = v1["request_decode_time_count"] - v0["request_decode_time_count"]

    # Average per-request durations (fallback to batch size if counts missing)
    denom_prefill = prefill_count_diff if prefill_count_diff > 0 else max(len(prompts), 1)
    denom_inference = inference_count_diff if inference_count_diff > 0 else max(len(prompts), 1)
    denom_decode = decode_count_diff if decode_count_diff > 0 else max(len(prompts), 1)

    prefill_avg_req = prefill_total / denom_prefill
    inference_avg_req = inference_total / denom_inference
    decode_avg_req = decode_total / denom_decode

    # Approximate wall-clock prefill duration for the batch (bounded by batch elapsed)
    batch_elapsed = max(t1_epoch - t0_epoch, 1e-6)
    prefill_wall_candidate = (
        ttft_mean if (isinstance(ttft_mean, (int, float)) and ttft_mean > 0) else prefill_avg_req
    )
    prefill_wall_s = min(max(prefill_wall_candidate, 0.0), batch_elapsed)

    diff = latency - e2e_latency_mean

    # prefill throughput (prompt tokens per second of prefill time)
    prompt_tps_prefill = (prompt_t / prefill_total) if prefill_total > 0 else 0.0

    # decode-time-based throughput
    decode_sum = v1["request_decode_time_sum"] - v0["request_decode_time_sum"]
    gen_tps_decode_time = (gen_t / decode_sum) if decode_sum > 0 else 0.0

    # ===== DERIVE: Higher-level metrics =====

    # Energy breakdown by phase: allocate total energy proportionally to time spent in each phase
    # Assumption: energy consumption proportional to time spent
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

    # ===== BUILD TELEMETRY DICT =====
    # Stores 20+ metrics about this batch: energy, tokens, latency, throughput, efficiency
    # These metrics are accumulated across batches and logged to MLflow
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
        "diff": diff,
        "prefill_avg_s": prefill_avg_req,
        "inference_avg_s": inference_avg_req,
        "decode_avg_s": decode_avg_req,
        "prefill_wall_s": prefill_wall_s,
        "prompt_tps_prefill": prompt_tps_prefill,
        "gen_tps_decode_time": gen_tps_decode_time,
        "J_total_per_token": joules / max(total_t, 1),
        # Additional metrics (kept for MLflow continuity)
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

    # Optional: For large batches (>=32 sentences), analyze metrics CSV file
    # CSV contains detailed timestamped energy readings throughout the run
    # This allows validating energy calculations against recorded data
    if len(prompts) >= 32:
        try:
            csv_metrics = analyze_metrics_csv(t0_epoch, t1_epoch, prefill_wall_s, prompt_t, gen_t)
            if csv_metrics:
                telemetry.update(csv_metrics)  # Add CSV-derived metrics to telemetry
                # Prefill energy as fraction of total batch energy (CSV-measured)
                csv_total = csv_metrics.get("csv_total_energy_j", 0.0)
                csv_pref = csv_metrics.get("csv_prefill_energy_j", 0.0)
                if csv_total > 0:
                    telemetry["csv_prefill_energy_ratio"] = csv_pref / csv_total
        except Exception as e:
            print(f"Error during CSV metrics analysis: {e}")

    return responses, telemetry


# =====================================================================
# DSPy NER Evaluation Pipeline
# =====================================================================
# When --use-dspy is set, this function replaces the raw HTTP pipeline.
# DSPy handles prompt construction and LM calls via Signature + Module.
# Energy telemetry is still captured per batch.
#
# Compatible with GEPA-optimised programs loaded via --dspy-program-path.


def evaluate_ner_pipeline_dspy(
    test_dataset,
    label_list,
    batch_size,
    program,
    language="de",
):
    """
    Run NER evaluation using a DSPy program instead of raw HTTP calls.

    The DSPy program (NERPredictor) outputs structured per/org/loc lists
    directly, so no response text parsing is needed.

    Returns the same tuple as evaluate_ner_pipeline_xtreme for MLflow
    compatibility: (ner_metrics, generated_results, batch_telemetry).
    """
    all_gold, all_pred = [], []
    generated_results = []
    batch_telemetry = []

    for i in tqdm(range(0, len(test_dataset), batch_size)):
        end = min(i + batch_size, len(test_dataset))
        batch = test_dataset.select(range(i, end))
        sentences = []
        gold_batch = []

        for example in batch:
            tokens = example["tokens"]
            ner_tags = example["ner_tags"]
            sentence = "".join(tokens) if language in ["zh", "ar"] else " ".join(tokens)
            gold_tags = [label_list[tag] for tag in ner_tags]
            sentences.append(sentence)
            gold_batch.append(gold_tags)

        # ===== Energy measurement around batch =====
        e0 = read_energy_joules()
        v0 = read_vllm_metrics()
        t0 = time.perf_counter()
        t0_epoch = time.time()

        batch_preds = []
        for sentence in sentences:
            try:
                result = program(sentence=sentence)
                entities = {
                    "PER": result.per if isinstance(result.per, list) else [],
                    "ORG": result.org if isinstance(result.org, list) else [],
                    "LOC": result.loc if isinstance(result.loc, list) else [],
                }
            except Exception as e:
                print(f"DSPy prediction failed for sentence: {sentence[:60]}... -> {e}")
                entities = {"PER": [], "ORG": [], "LOC": []}
            batch_preds.append(entities)

        t1 = time.perf_counter()
        t1_epoch = time.time()
        e1 = read_energy_joules()
        v1 = read_vllm_metrics()

        # ===== Compute telemetry =====
        joules = max(e1 - e0, 0.0)
        latency = t1 - t0
        prompt_t = max(v1["prompt_tokens_total"] - v0["prompt_tokens_total"], 0)
        gen_t = max(v1["generation_tokens_total"] - v0["generation_tokens_total"], 0)
        total_t = prompt_t + gen_t

        e2e_count_diff = max(v1["e2e_latency_count"] - v0["e2e_latency_count"], 0)
        e2e_latency_mean = max(
            (v1["e2e_latency_sum"] - v0["e2e_latency_sum"]) / max(e2e_count_diff, 1), 0.0
        )

        telemetry = {
            "batch_size": len(sentences),
            "latency_s": latency,
            "energy_j": joules,
            "prompt_tokens": prompt_t,
            "generation_tokens": gen_t,
            "total_tokens": total_t,
            "e2e_latency_mean": e2e_latency_mean,
            "J_total_per_token": joules / max(total_t, 1),
            "avg_power_draw": joules / max(e2e_latency_mean, 1e-6),
        }

        if len(sentences) >= 32:
            try:
                csv_metrics = analyze_metrics_csv(t0_epoch, t1_epoch, latency, prompt_t, gen_t)
                if csv_metrics:
                    telemetry.update(csv_metrics)
            except Exception as e:
                print(f"Error during CSV metrics analysis: {e}")

        batch_telemetry.append(telemetry)

        # ===== BIO tagging + alignment =====
        for entities, gold_tags, sentence in zip(batch_preds, gold_batch, sentences, strict=True):
            _, pred_tags = get_bio_tags_language_aware(sentence, entities, language)

            if len(pred_tags) < len(gold_tags):
                pred_tags += ["O"] * (len(gold_tags) - len(pred_tags))
            elif len(pred_tags) > len(gold_tags):
                pred_tags = pred_tags[: len(gold_tags)]

            generated_results.append(
                {
                    "sentence": sentence,
                    "raw_output": str(entities),
                    "entities": entities,
                    "gold_tags": gold_tags,
                    "pred_tags": pred_tags,
                }
            )
            all_gold.append(gold_tags)
            all_pred.append(pred_tags)

    # ===== Compute final NER metrics =====
    if all_gold and all_pred:
        ner_metrics = {
            "precision": precision_score(all_gold, all_pred),
            "recall": recall_score(all_gold, all_pred),
            "f1": f1_score(all_gold, all_pred),
            "cls_report": classification_report(all_gold, all_pred, output_dict=True),
        }
        total_pred = sum(1 for seq in all_pred for tag in seq if tag.startswith("B-"))
        total_true = sum(1 for seq in all_gold for tag in seq if tag.startswith("B-"))
        ner_metrics["total_pred_entities"] = total_pred
        ner_metrics["total_true_entities"] = total_true
    else:
        ner_metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "cls_report": {},
            "total_pred_entities": 0,
            "total_true_entities": 0,
        }

    return ner_metrics, generated_results, batch_telemetry


# =====================================================================
# NER Evaluation Pipeline: Core Loop
# =====================================================================
# This function implements the complete NER evaluation:
#   1. For each batch: send to model + measure energy
#   2. Parse model outputs: messy text → clean entity lists
#   3. Convert to BIO tags: entity mentions → token-by-token labels (language-aware)
#   4. Accumulate: collect gold and predicted tags across all sentences
#   5. Calculate final NER metrics: precision, recall, F1 using seqeval library


async def evaluate_ner_pipeline_xtreme(
    test_dataset,
    label_list,
    batch_size,
    model_name,
    max_new_tokens=150,
    language="de",
    system_prompt_choice=None,
    use_generic_template=False,
    custom_system_prompt=None,
    max_concurrency=1024,
):
    """
    Run end-to-end NER evaluation on XTREME dataset.

    Process flow for each batch:
    - Extract sentences and gold BIO tags from test data
    - Send batch to model via process_and_measure() (measure energy + latency)
    - Parse model responses (utils.parse_gollie_output_to_entities or utils.robust_gemma_entity_extraction)
    - Convert entity mentions to BIO tags (utils.get_bio_tags_language_aware)
    - Align prediction length with gold tags
    - Accumulate into all_gold and all_pred lists

    After all batches:
    - Calculate NER metrics using seqeval (precision, recall, F1)
    - Count total predicted and true entities

    Args:
        test_dataset: HuggingFace dataset with token sequences and NER tag IDs
        label_list: list of label names (e.g., ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'O'])
        batch_size: number of sentences per batch (32-128 typical)
        model_name: /mistral, /gollie, /gemma-3-4b-it, /gemma-3-12b-it
        max_new_tokens: max tokens to generate (150 typical)
        language: language code (en, de, zh, ar, etc.)
        system_prompt_choice: which system prompt variant (1-8)
        use_generic_template: use language-agnostic or language-specific prompt

    Returns:
        tuple of (ner_metrics, generated_results, batch_telemetry)
        - ner_metrics: dict with precision, recall, f1, classification_report, entity counts
        - generated_results: list of dicts with sentence, raw output, entities, tags
        - batch_telemetry: list of telemetry dicts (one per batch)
    """
    # Accumulators for metrics calculation
    all_gold, all_pred = [], []  # Gold and predicted BIO tags for all sentences
    generated_results = []  # Debug: raw outputs, entities, tags for each sentence
    batch_telemetry = []  # Energy, latency, tokens for each batch

    # Configure async HTTP session with generous timeouts
    timeout = ClientTimeout(total=None, sock_connect=30, sock_read=900)

    # HTTP_TRANSPORT_MODE selects the connection-pooling strategy (reviewer
    # question: does keep-alive/connection pooling vs. per-request connections
    # affect TIME_WAIT churn, recall, and energy?). Defaults to "pooled",
    # the configuration used for all published sweep numbers.
    #   pooled (default): shared pool, keep-alive on, limit=4096
    #   close:            keep-alive off, one fresh TCP connection per request
    #   tight:            keep-alive on, pool size capped to max_concurrency
    http_transport_mode = os.environ.get("HTTP_TRANSPORT_MODE", "pooled").lower()
    if http_transport_mode == "close":
        connector = TCPConnector(limit=4096, enable_cleanup_closed=True, force_close=True)
    elif http_transport_mode == "tight":
        connector = TCPConnector(limit=max_concurrency, enable_cleanup_closed=True, force_close=False)
    else:
        connector = TCPConnector(limit=4096, enable_cleanup_closed=True, force_close=False)
    print(f"[INFO] HTTP_TRANSPORT_MODE={http_transport_mode}")

    # Semaphore limits concurrent HTTP requests to avoid TCP flooding / TIME_WAIT exhaustion
    semaphore = asyncio.Semaphore(max_concurrency)
    print(f"[INFO] Semaphore: max {max_concurrency} concurrent HTTP requests")

    # Optional TIME_WAIT socket diagnostic (reviewer question: does connection
    # pooling/keep-alive actually reduce TIME_WAIT churn?). Opt-in via
    # SOCKET_MONITOR=1 so it never runs during normal/published sweeps.
    socket_monitor_task = None
    socket_log_path = None
    if os.environ.get("SOCKET_MONITOR", "0") == "1":
        socket_log_path = os.path.join(
            os.environ.get("ARTIFACTS_DIR", "."),
            f"socket_timewait_{http_transport_mode}.csv",
        )

        async def _poll_time_wait(log_path):
            with open(log_path, "w") as f:
                f.write("timestamp,time_wait_count\n")
                while True:
                    try:
                        out = subprocess.run(
                            ["ss", "-tan", "state", "time-wait"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        count = max(0, len(out.stdout.splitlines()) - 1)
                    except Exception:
                        count = -1
                    f.write(f"{time.time()},{count}\n")
                    f.flush()
                    await asyncio.sleep(2)

        socket_monitor_task = asyncio.create_task(_poll_time_wait(socket_log_path))

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Process dataset in batches (32-128 sentences per batch)
        for i in tqdm(range(0, len(test_dataset), batch_size)):
            end = min(i + batch_size, len(test_dataset))
            batch = test_dataset.select(range(i, end))
            sentences = []
            gold_batch = []

            # Extract sentences and gold tags from batch examples
            for example in batch:
                tokens = example["tokens"]
                ner_tags = example["ner_tags"]

                # Language-aware sentence reconstruction:
                #   - Chinese, Arabic: no spaces (characters compound)
                #   - Other languages: join with spaces
                if language in ["zh", "ar"]:
                    sentence = "".join(tokens)  # Character-level
                else:
                    sentence = " ".join(tokens)  # Token-level

                # Decode label IDs to BIO tag strings (e.g., [0, 1, 4] → ['B-PER', 'I-PER', 'O'])
                gold_tags = [label_list[tag] for tag in ner_tags]
                sentences.append(sentence)
                gold_batch.append(gold_tags)

            try:
                # Send batch and measure: energy, latency, tokens, throughput
                responses, telemetry = await process_and_measure(
                    session,
                    sentences,
                    max_new_tokens,
                    model_name,
                    language,
                    system_prompt_choice,
                    use_generic_template,
                    custom_system_prompt,
                    semaphore=semaphore,
                )
                batch_telemetry.append(telemetry)

                # Process each sentence's response: extract entities → BIO tags
                for response, gold_tags, sentence in zip(responses, gold_batch, sentences):
                    # ===== STEP 1: Extract text from model response =====
                    if isinstance(response, Exception):
                        print(f"[WARN] Request failed: {type(response).__name__}: {response}")
                        text = ""
                    elif (
                        isinstance(response, dict)
                        and "choices" in response
                        and len(response["choices"]) > 0
                    ):
                        # Handle different response formats (chat vs. completion)
                        if model_name in ["/mistral", "/gemma-3-4b-it", "/gemma-3-12b-it", "/gemma-3-27b-it", "/llama-3-3-70B-it"]:
                            # Chat responses: look for message.content
                            text = (
                                response["choices"][0]["message"]["content"]
                                if "message" in response["choices"][0]
                                else response["choices"][0].get("text", "")
                            )
                        else:
                            # Completion responses: look for text field
                            text = response["choices"][0].get("text", "")
                    else:
                        text = str(response)

                    # ===== STEP 2: Parse entities from messy model text =====
                    # Different parsing strategies for different models
                    if model_name == "/mistral":
                        # Mistral: expects JSON-like output
                        entities = parse_response_json_like(text)
                    elif model_name in ["/gemma-3-4b-it", "/gemma-3-12b-it", "/gemma-3-27b-it", "/llama-3-3-70B-it"]:
                        # Gemma/Llama: may wrap JSON in markdown, has fallback regex extraction
                        if i < batch_size and len(generated_results) < 3:
                            print(
                                f"\nDEBUG Gemma Entity Extraction for example {len(generated_results)}:"
                            )
                            print(f"Raw text starts with: {text[:100]}...")
                        entities = robust_gemma_entity_extraction(text)
                        if i < batch_size and len(generated_results) < 3:
                            print(f"Extracted entities: {entities}")
                    else:
                        # GoLLIE: special pseudo-code format with PER(mention="...") patterns
                        entities = parse_gollie_output_to_entities(text)

                    # Debug output for Arabic (language with special handling)
                    if language == "ar" and entities:
                        print(f"DEBUG: Sentence: {sentence[:50]}...")
                        print(f"DEBUG: Raw output: {text[:100]}...")
                        print(f"DEBUG: Extracted entities: {entities}")

                    # ===== STEP 3: Convert entity mentions to BIO tags =====
                    # Language-aware: Chinese/Arabic use char-level, others use token-level
                    _, pred_tags = get_bio_tags_language_aware(sentence, entities, language)

                    if language == "ar" and entities:
                        print(f"DEBUG: Generated pred_tags length: {len(pred_tags)}")
                        print(f"DEBUG: Gold tags length: {len(gold_tags)}")
                        print(f"DEBUG: First 10 pred_tags: {pred_tags[:10]}")

                    # ===== STEP 4: Align prediction length with gold tags =====
                    # seqeval expects sequences of same length → pad or truncate
                    if len(pred_tags) < len(gold_tags):
                        pred_tags += ["O"] * (len(gold_tags) - len(pred_tags))
                    elif len(pred_tags) > len(gold_tags):
                        pred_tags = pred_tags[: len(gold_tags)]

                    # ===== STEP 5: Store results for analysis =====
                    generated_results.append(
                        {
                            "sentence": sentence,
                            "raw_output": text,
                            "entities": entities,
                            "gold_tags": gold_tags,
                            "pred_tags": pred_tags,
                        }
                    )
                    # Accumulate for final metrics calculation
                    all_gold.append(gold_tags)
                    all_pred.append(pred_tags)

            except Exception as e:
                # Graceful degradation: if batch fails, append empty predictions
                print(f"Batch {i // batch_size} failed:", e)
                for gold_tags in gold_batch:
                    all_gold.append(gold_tags)
                    all_pred.append(["O"] * len(gold_tags))  # All "Outside" tags
                continue

    if socket_monitor_task is not None:
        socket_monitor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await socket_monitor_task
        print(f"[INFO] TIME_WAIT socket log written to {socket_log_path}")

    # ===== CALCULATE FINAL NER METRICS =====
    # Use seqeval library to compute precision, recall, F1 at entity span level

    if len(all_gold) > 0 and len(all_pred) > 0:
        # Calculate NER metrics using seqeval library
        # seqeval counts correct entity spans (boundaries + type must match)
        ner_metrics = {
            "precision": precision_score(
                all_gold, all_pred
            ),  # Of predicted entities, how many correct?
            "recall": recall_score(all_gold, all_pred),  # Of true entities, how many detected?
            "f1": f1_score(all_gold, all_pred),  # Harmonic mean of precision and recall
            "cls_report": classification_report(
                all_gold, all_pred, output_dict=True
            ),  # Per-entity breakdown
        }

        # Count total entity mentions (B- tags mark entity boundaries)
        total_pred = sum(1 for seq in all_pred for tag in seq if tag.startswith("B-"))
        total_true = sum(1 for seq in all_gold for tag in seq if tag.startswith("B-"))
        ner_metrics["total_pred_entities"] = total_pred
        ner_metrics["total_true_entities"] = total_true
    else:
        # No predictions: return zero metrics
        ner_metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "cls_report": {},
            "total_pred_entities": 0,
            "total_true_entities": 0,
        }

    return ner_metrics, generated_results, batch_telemetry


# =====================================================================
# MLflow Orchestration: Main Experiment Runner
# =====================================================================
# This function orchestrates the entire experiment:
#   1. Load XTREME dataset for specified language
#   2. Setup output directories and MLflow experiment
#   3. Run NER evaluation pipeline (calls evaluate_ner_pipeline_xtreme)
#   4. Save results to JSON files
#   5. Log NER metrics and telemetry to MLflow
#   6. Compute and log energy-efficiency metrics
#
# After run completes:
#   - Results are in JSON files (for offline analysis)
#   - Metrics are in MLflow UI (for interactive exploration and comparison)


async def run(args):

    # ===== STEP 1: Load Dataset =====
    # Resolve language code to XTREME PAN-X subset name (e.g., "de" → "PAN-X.de")
    subset = resolve_xtreme_subset(args.language)
    print(f"Loading XTREME subset: {subset}")

    # Load test split from HuggingFace Datasets library
    test_ds = load_dataset("google/xtreme", subset, split="test", trust_remote_code=True)

    # Optionally limit to first N samples (for quick testing or small experiments)
    # Default: use all samples (up to 10,000)
    test_subset = test_ds.select(range(min(args.limit if args.limit else 10000, len(test_ds))))

    # Extract label names from dataset metadata (e.g., ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'O'])
    labels = test_ds.features["ner_tags"].feature.names
    print(
        f"Loaded {args.language} dataset with {len(test_subset)} samples "
        f"and {len(labels)} labels: {labels}"
    )

    # ===== STEP 2: Setup Output Directories =====
    # Use ARTIFACTS_DIR from environment if set (by shell wrapper)
    # Otherwise create directory based on language, batch size, and model name
    out_dir = os.environ.get("ARTIFACTS_DIR")
    if not out_dir:
        artifacts_dir = build_artifacts_dir(args.language, args.batch_size, args.model)
        base_dir = "./inference_eval_artifacts/xtreme"
        out_dir = os.path.join(base_dir, artifacts_dir)

    # Create subdirectories for different artifact types
    os.makedirs(os.path.join(out_dir, "generated_responses"), exist_ok=True)  # Raw model outputs
    os.makedirs(os.path.join(out_dir, "ner_metrics"), exist_ok=True)  # NER evaluation results
    os.makedirs(os.path.join(out_dir, "telemetry"), exist_ok=True)  # Energy and latency data

    # ===== STEP 3: Setup MLflow Tracking =====
    # MLflow allows comparing experiments across different configurations
    # Experiment: logical grouping of related runs
    # MLFLOW_EXPERIMENT_NAME env var overrides the default name (used by sweep orchestrator)
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", f"prompts_en_{args.model}_xtreme")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    # Optional: include timestamp in run name for easier identification
    timestamp_suffix = ""
    if getattr(args, "run_start_timestamp", None):
        try:
            dt = pd.to_datetime(args.run_start_timestamp)
            timestamp_suffix = f"_T{dt.strftime('%H%M%S')}"
        except Exception:
            pass

    # Start MLflow run: tracks all metrics, params, and artifacts for this experiment
    _sweep_param = os.environ.get("SWEEP_PARAM_CHANGED", "")
    _sweep_value = os.environ.get("SWEEP_PARAM_VALUE", "")
    _run_base = f"xtreme_ner_b_{str(args.batch_size)}_{args.language}_prompt{args.system_prompt_choice}{timestamp_suffix}_{job_id}"
    _run_name = f"{_sweep_param}_{_run_base}" if _sweep_param else _run_base
    with mlflow.start_run(run_name=_run_name):
        # ===== Log Experiment Configuration to MLflow =====
        # Tags: categorical labels for filtering/grouping runs
        tags = {
            "language": args.language,
            "batch_size": str(args.batch_size),
            "model": args.model,
            "dataset": "xtreme",
            "n_samples": str(len(test_subset)),
            "max_concurrency": str(args.max_concurrency),
        }
        if _sweep_param:
            tags["sweep_param_changed"] = _sweep_param
            tags["sweep_param_value"] = _sweep_value

        # Params: numeric/string hyperparameters that affect model behavior
        params = {
            "language": args.language,
            "batch_size": args.batch_size,
            "model": args.model,
            "max_new_tokens": args.max_new_tokens,
            "max_concurrency": args.max_concurrency,
            "dataset": "xtreme",
            "n_samples": str(len(test_subset)),
        }

        # Add system prompt choice if applicable
        if (
            args.model in ["/gemma-3-4b-it", "/gemma-3-12b-it", "/gemma-3-27b-it", "/mistral", "/llama-3-3-70B-it"]
            and args.system_prompt_choice is not None
        ):
            tags["system_prompt_choice"] = str(args.system_prompt_choice)
            params["system_prompt_choice"] = args.system_prompt_choice

        # Add generic template flag if used
        if args.use_generic_template:
            tags["use_generic_template"] = "True"
            params["use_generic_template"] = True

        if _sweep_param:
            tags["sweep_param_changed"] = _sweep_param
            tags["sweep_param_value"] = _sweep_value

        # Publish tags and params to MLflow
        mlflow.set_tags(tags)
        mlflow.log_params(params)

        # ---- Find and log server config + key params ----
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
                    with open(srv_config_path) as _cf:
                        srv_cfg = json.load(_cf)
                    cfg_params = {}
                    for _k in [
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
                    ]:
                        if _k in srv_cfg:
                            cfg_params[_k] = srv_cfg[_k]
                    if cfg_params:
                        mlflow.log_params(cfg_params)
                except Exception as _e:
                    print(f"Warning: could not parse server config: {_e}")

        # ---- Sample prompt ----
        # For styles 9 and 10 (DSPy optimized), load instructions first and build sample manually
        custom_system_prompt = None
        if args.system_prompt_choice in (9, 10) and args.dspy_program_path:
            custom_system_prompt = load_dspy_instructions(args.dspy_program_path)
            if custom_system_prompt is None:
                print(f"[ERROR] Could not load DSPy instructions from {args.dspy_program_path}")
                raise SystemExit(1)
            print(
                f"[INFO] Loaded optimized DSPy instructions for style {args.system_prompt_choice} from {args.dspy_program_path}"
            )

        if custom_system_prompt is not None:
            sample_sentence = "John Smith from Microsoft visited Berlin last week."
            sample_prompt = {
                "prompt_type": "dspy_optimized_chat",
                "messages": [
                    {"role": "system", "content": custom_system_prompt},
                    {"role": "user", "content": sample_sentence},
                ],
                "system_prompt_choice": args.system_prompt_choice,
                "dspy_program_path": args.dspy_program_path,
            }
        else:
            sample_prompt = create_sample_prompt(
                args.model,
                args.language,
                args.system_prompt_choice
                if args.model in ["/gemma-3-4b-it", "/gemma-3-12b-it", "/gemma-3-27b-it", "/mistral", "/llama-3-3-70B-it"]
                else None,
                use_generic_template=args.use_generic_template,
            )
        # Build a prompt label for per-prompt output files
        prompt_label = (
            f"prompt{args.system_prompt_choice}" if args.system_prompt_choice else "default"
        )

        sample_prompt_path = os.path.join(out_dir, f"sample_prompt_{prompt_label}.json")
        with open(sample_prompt_path, "w") as f:
            json.dump(sample_prompt, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(sample_prompt_path)

        # ===== STEP 4: Run NER Evaluation =====
        # This is the main pipeline: processes entire test set in batches
        # Returns: NER metrics (P/R/F1), raw responses, and energy/latency telemetry
        if args.use_dspy:
            # DSPy path: uses structured Signature + Module for NER prediction.
            # Optionally loads a GEPA-optimised program from disk.
            setup_dspy(args.model)
            program = create_ner_program(
                program_path=args.dspy_program_path,
                use_chain_of_thought=args.dspy_chain_of_thought,
            )
            print(
                f"DSPy mode: program={'loaded from ' + args.dspy_program_path if args.dspy_program_path else 'baseline'}, "
                f"CoT={args.dspy_chain_of_thought}"
            )

            # Log DSPy-specific params to MLflow
            mlflow.log_params(
                {
                    "use_dspy": True,
                    "dspy_chain_of_thought": args.dspy_chain_of_thought,
                    "dspy_program_path": args.dspy_program_path or "baseline",
                }
            )

            ner_metrics, gen_responses, batch_telemetry = evaluate_ner_pipeline_dspy(
                test_subset,
                labels,
                batch_size=args.batch_size,
                program=program,
                language=args.language,
            )
        else:
            ner_metrics, gen_responses, batch_telemetry = await evaluate_ner_pipeline_xtreme(
                test_subset,
                labels,
                batch_size=args.batch_size,
                model_name=args.model,
                max_new_tokens=args.max_new_tokens,
                language=args.language,
                system_prompt_choice=args.system_prompt_choice
                if args.model in ["/gemma-3-4b-it", "/gemma-3-12b-it", "/gemma-3-27b-it", "/mistral", "/llama-3-3-70B-it"]
                else None,
                use_generic_template=args.use_generic_template,
                custom_system_prompt=custom_system_prompt,
                max_concurrency=args.max_concurrency,
            )

        # ===== STEP 5: Save Results to JSON Files =====
        # JSON files enable offline analysis and version control
        responses_path = os.path.join(
            out_dir,
            "generated_responses",
            f"responses_B{args.batch_size}_{prompt_label}_{job_id}.json",
        )
        ner_metrics_path = os.path.join(
            out_dir, "ner_metrics", f"ner_metrics_B_{args.batch_size}_{prompt_label}_{job_id}.json"
        )
        telemetry_path = os.path.join(
            out_dir, "telemetry", f"telemetry_B_{args.batch_size}_{prompt_label}_{job_id}.json"
        )

        # Save generated responses, NER metrics, and telemetry
        with open(responses_path, "w") as f:
            json.dump(gen_responses, f, indent=2, default=numpy_serializer, ensure_ascii=False)
        with open(ner_metrics_path, "w") as f:
            json.dump(ner_metrics, f, indent=2, default=numpy_serializer, ensure_ascii=False)
        with open(telemetry_path, "w") as f:
            json.dump(batch_telemetry, f, indent=2, default=numpy_serializer, ensure_ascii=False)

        # ===== STEP 6: Log NER Metrics to MLflow =====
        # These metrics appear in MLflow UI for quick visualization and comparison
        mlflow.log_artifact(responses_path)
        mlflow.log_artifact(ner_metrics_path)
        mlflow.log_artifact(telemetry_path)

        # Log primary NER metrics
        mlflow.log_metric("precision", float(ner_metrics.get("precision", 0.0)))
        mlflow.log_metric("recall", float(ner_metrics.get("recall", 0.0)))
        mlflow.log_metric("f1", float(ner_metrics.get("f1", 0.0)))

        # ===== STEP 7: Aggregate and Compute Energy-Efficiency Metrics =====
        # Aggregate metrics across all batches: compute means and efficiency metrics
        mean_metrics = {}

        if batch_telemetry:
            # Aggregate each metric type across batches (compute means)
            all_keys = set()
            for t in batch_telemetry:
                all_keys.update(t.keys())
            for key in sorted(all_keys):
                vals = []
                for t in batch_telemetry:
                    v = t.get(key)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        vals.append(v)
                if vals:
                    mean_metrics[key] = float(np.mean(vals))

            # Sum total energy across all batches (not mean, but sum)
            energy_j_sum = float(
                sum(
                    t.get("energy_j", 0.0)
                    for t in batch_telemetry
                    if isinstance(t.get("energy_j", None), (int, float))
                    and not isinstance(t.get("energy_j"), bool)
                )
            )
            mean_metrics["energy_j_sum"] = energy_j_sum

            # Sum CSV-measured prefill and generation energy across all batches
            csv_prefill_energy_sum = float(
                sum(
                    t.get("csv_prefill_energy_j", 0.0)
                    for t in batch_telemetry
                    if isinstance(t.get("csv_prefill_energy_j", None), (int, float))
                )
            )
            csv_generation_energy_sum = float(
                sum(
                    t.get("csv_generation_energy_j", 0.0)
                    for t in batch_telemetry
                    if isinstance(t.get("csv_generation_energy_j", None), (int, float))
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

            # Compute avg generation tokens per sentence (for IE vs non-IE comparison)
            total_gen_tokens = float(
                sum(
                    t.get("generation_tokens", 0)
                    for t in batch_telemetry
                    if isinstance(t.get("generation_tokens", None), (int, float))
                )
            )
            total_sentences = sum(
                t.get("batch_size", 0)
                for t in batch_telemetry
                if isinstance(t.get("batch_size", None), (int, float))
            )
            mean_metrics["avg_gen_tokens_per_sentence"] = total_gen_tokens / max(total_sentences, 1)

            # Store NER metrics alongside telemetry metrics
            whole_energy = energy_j_sum
            mean_metrics.update(
                {
                    "ner_precision": float(ner_metrics["precision"]),
                    "ner_recall": float(ner_metrics["recall"]),
                    "ner_f1": float(ner_metrics["f1"]),
                    "whole_energy": whole_energy,
                }
            )

            # ===== COMPUTE ENERGY-EFFICIENCY METRICS =====
            # Answer research questions: "What's the energy cost of good NER?"
            f1 = float(ner_metrics.get("f1", 0.0))
            pred_entities = int(ner_metrics.get("total_pred_entities", 0))
            true_entities = int(ner_metrics.get("total_true_entities", 0))
            total_seconds = float(sum(t.get("latency_s", 0.0) for t in batch_telemetry))

            # True positives: count predicted entities that matched gold entities
            # Approximated as: (precision * pred_entities)
            tp = float(ner_metrics.get("precision", 0.0)) * float(pred_entities)

            mean_metrics["total_seconds"] = total_seconds
            mean_metrics["total_pred_entities"] = pred_entities
            mean_metrics["total_true_entities"] = true_entities

            # Energy-to-quality ratios:
            # - J_per_F1: energy cost to achieve unit F1 score
            # - F1_per_J: F1 score per joule (efficiency)
            if f1 > 0 and energy_j_sum > 0:
                mean_metrics["J_per_F1"] = energy_j_sum / f1
                mean_metrics["F1_per_J"] = f1 / energy_j_sum

            # Entity-to-energy ratios:
            # - J_per_entity: energy cost per predicted entity
            # - entities_per_J: entity extraction efficiency
            if pred_entities > 0 and energy_j_sum > 0:
                mean_metrics["J_per_entity"] = energy_j_sum / pred_entities
                mean_metrics["entities_per_J"] = pred_entities / energy_j_sum

            # True-positive-to-energy ratios:
            # - J_per_TP: energy cost per correct entity (best measure of value)
            # - TP_per_J: correct entities per joule (efficiency)
            # - s_per_TP: seconds per correct entity
            # - TP_per_s: correct entities per second (throughput)
            if tp > 0 and energy_j_sum > 0:
                mean_metrics["J_per_TP"] = energy_j_sum / tp
                mean_metrics["TP_per_J"] = tp / energy_j_sum
                mean_metrics["s_per_TP"] = total_seconds / tp if total_seconds > 0 else 0.0
                mean_metrics["TP_per_s"] = tp / total_seconds if total_seconds > 0 else 0.0

            # ===== Log All Metrics to MLflow =====
            # Filter out non-numeric values before logging
            loggable_metrics = {
                k: float(v)
                for k, v in mean_metrics.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            mlflow.log_metrics(loggable_metrics)

            mean_metrics_path = os.path.join(
                out_dir, f"mean_metrics_B_{args.batch_size}_{prompt_label}_{job_id}.json"
            )
            with open(mean_metrics_path, "w") as f:
                json.dump(mean_metrics, f, indent=2, default=numpy_serializer)
            mlflow.log_artifact(mean_metrics_path)

            # Log inference config files from artifacts dir
            config_files = glob.glob(os.path.join(out_dir, "*out_inference.json"))
            for cfg in config_files:
                mlflow.log_artifact(cfg)
            env_artifacts = os.environ.get("ARTIFACTS_DIR", "")
            if env_artifacts and os.path.abspath(env_artifacts) != os.path.abspath(out_dir):
                for cfg in glob.glob(os.path.join(env_artifacts, "*out_inference.json")):
                    mlflow.log_artifact(cfg)

        # ===== STEP 8: Compute Corrected Energy from Detailed CSV =====
        # For validation: query the metrics CSV file with detailed timestamped energy readings
        # This provides an alternative energy measurement independent of vLLM metrics
        time.sleep(10)  # Wait for CSV to be written to disk
        run_start_timestamp = getattr(args, "run_start_timestamp", None)

        # Compute energy by examining GPU utilization in CSV:
        # Only count energy when GPU is actively working (not idle periods)
        res = compute_energy_corrected(metrics_csv_path, start_timestamp=run_start_timestamp)

        # Log corrected energy metrics if CSV analysis succeeded
        if res.get("ok"):
            energy_corrected = float(res["energy_corrected"])
            latency_corrected = float(res["latency_corrected"])
            mlflow.log_metric("energy_corrected", energy_corrected)
            mlflow.log_metric("latency_corrected", latency_corrected)
            mean_metrics.update(
                {
                    "energy_corrected": energy_corrected,
                    "latency_corrected": latency_corrected,
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

            # ===== RECOMPUTE EFFICIENCY METRICS USING energy_corrected =====
            # energy_corrected (GPU util > 0 window from CSV) is the preferred denominator
            # for J/TP, J/F1 etc. because:
            #   - energy_j_sum = sum of per-batch DCGM deltas (includes minor TCP/scheduling
            #     overhead at batch boundaries, misses between-batch GPU idle)
            #   - energy_corrected = NVML-measured energy only when GPU is actively computing,
            #     independent of client-side HTTP timing → task-attributable energy
            if batch_telemetry:
                corrected_efficiency: dict = {}
                if f1 > 0 and energy_corrected > 0:
                    corrected_efficiency["J_per_F1_corrected"] = energy_corrected / f1
                    corrected_efficiency["F1_per_J_corrected"] = f1 / energy_corrected
                if pred_entities > 0 and energy_corrected > 0:
                    corrected_efficiency["J_per_entity_corrected"] = (
                        energy_corrected / pred_entities
                    )
                    corrected_efficiency["entities_per_J_corrected"] = (
                        pred_entities / energy_corrected
                    )
                if tp > 0 and energy_corrected > 0:
                    corrected_efficiency["J_per_TP_corrected"] = energy_corrected / tp
                    corrected_efficiency["TP_per_J_corrected"] = tp / energy_corrected
                    corrected_efficiency["TP_per_s_corrected"] = (
                        tp / latency_corrected if latency_corrected > 0 else 0.0
                    )
                    corrected_efficiency["s_per_TP_corrected"] = (
                        latency_corrected / tp if tp > 0 else 0.0
                    )

                if corrected_efficiency:
                    mean_metrics.update(corrected_efficiency)
                    mlflow.log_metrics({k: float(v) for k, v in corrected_efficiency.items()})

                # Re-save mean_metrics JSON with corrected efficiency metrics included
                with open(mean_metrics_path, "w") as f:
                    json.dump(mean_metrics, f, indent=2, default=numpy_serializer)
                mlflow.log_artifact(mean_metrics_path)
        else:
            # If CSV analysis failed, log the reason
            mlflow.set_tag("energy_window_note", f"skipped: {res.get('reason')}")

        # Log the CSV file itself to MLflow for audit trail
        if metrics_csv_path and os.path.isfile(metrics_csv_path):
            mlflow.log_artifact(metrics_csv_path)

        # Print final summary
        print(
            f"Completed: F1={ner_metrics['f1']:.4f}, "
            f"Energy Sum={mean_metrics.get('energy_j_sum', 0.0):.4f}J, "
            f"Whole Energy={mean_metrics.get('whole_energy', 0.0):.4f}J"
        )


# =====================================================================
# Command-Line Interface (CLI)
# =====================================================================
# User-facing command-line arguments for the evaluation script.
# Allows flexible experiment configuration without code changes.
#
# Example usage:
#   python lang_eval_mlflow_mi_gol.py --language de --model /gemma-3-4b-it --batch-size 128
#   python lang_eval_mlflow_mi_gol.py --language all --model /mistral --batch-size 64

# All supported languages in XTREME PAN-X dataset
SUPPORTED_LANGUAGES = sorted(
    [
        "ar",
        "bg",
        "de",
        "en",
        "es",
        "fr",
        "el",
        "hi",
        "id",
        "it",
        "ja",
        "ko",
        "nl",
        "pt",
        "ru",
        "th",
        "tr",
        "ur",
        "vi",
        "zh",
        "yo",
    ]
)


def main():
    """Parse command-line arguments and execute NER evaluation."""
    parser = argparse.ArgumentParser(
        description="XTREME NER evaluation with vLLM + energy & MLflow"
    )

    # ===== Language Selection =====
    # Which language from XTREME PAN-X to evaluate.
    # Use 'all' to run evaluation across all supported languages sequentially.
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=SUPPORTED_LANGUAGES + ["all"],
        help="Language subset from XTREME PAN-X. Use 'all' to run all.",
    )

    # ===== Model Selection =====
    # Which pre-trained LLM to use for NER evaluation.
    # Different models have different inference endpoints and prompt requirements.
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Which vLLM-served model to use for NER predictions.",
    )

    # ===== Batch Configuration =====
    # batch-size: Number of sentences to send to model in a single HTTP request.
    #   Higher batch sizes improve throughput but increase latency and memory.
    #   Default: 128 sentences per batch
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of sentences per batch (affects energy and latency measurements).",
    )

    # max-new-tokens: Maximum output length for each sentence.
    #   Limits generation to avoid runaway outputs. Typically 150 tokens captures
    #   entity-rich BIO tag sequences.
    #   Default: 150 tokens
    parser.add_argument(
        "--max-new-tokens", type=int, default=150, help="Maximum tokens to generate per sentence."
    )

    # ===== Concurrency Control =====
    # max-concurrency: Semaphore limit for concurrent HTTP requests per batch.
    #   Prevents TCP flooding and TIME_WAIT port exhaustion at high batch sizes.
    #   Should be <= max_num_seqs in vLLM config.
    #   Default: 1024 (aggressive: matches max_num_seqs=1024)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1024,
        help="Max concurrent HTTP requests per batch (semaphore limit). Default: 1024.",
    )

    # ===== Prompt Configuration =====
    # system-prompt-choice: For Gemma models, selects which of 8 system prompts to use.
    #   Different variants may emphasize different aspects (accuracy, conciseness, format).
    #   Range 1-8. Ignored for Mistral and GoLLIE models.
    parser.add_argument(
        "--system-prompt-choice",
        type=int,
        choices=range(1, 11),
        default=1,
        help="Which system prompt variant to use (1-8 for Gemma, 9 for DSPy optimised, 10 for matched DSPy optimised).",
    )

    # use-generic-template: Use language-agnostic instructions instead of
    #   language-specific system prompts. This evaluates model robustness across
    #   languages without language-tailored guidance.
    parser.add_argument(
        "--use-generic-template",
        action="store_true",
        help="Use generic prompt template instead of language-specific templates.",
    )

    # ===== DSPy Configuration =====
    # use-dspy: Enable DSPy-based NER evaluation (structured Signature + Module).
    #   Replaces raw HTTP prompt construction with DSPy's Predict/ChainOfThought.
    #   Compatible with GEPA-optimised programs (load via --dspy-program-path).
    parser.add_argument(
        "--use-dspy",
        action="store_true",
        help="Use DSPy program for NER instead of raw HTTP prompts.",
    )

    # dspy-program-path: Path to a saved GEPA-optimised DSPy program.
    #   Created by program.save() after GEPA optimisation.
    #   If not set, uses baseline (un-optimised) NER program.
    parser.add_argument(
        "--dspy-program-path",
        type=str,
        default=None,
        help="Path to a saved DSPy program JSON (from GEPA optimisation).",
    )

    # dspy-chain-of-thought: Use ChainOfThought instead of Predict.
    #   Adds a rationale step before entity extraction (slower but may improve accuracy).
    parser.add_argument(
        "--dspy-chain-of-thought",
        action="store_true",
        help="Use ChainOfThought (adds rationale step) instead of Predict.",
    )

    # ===== Data Sampling =====
    # limit: Sample only first N examples from the test set.
    #   Useful for quick testing without waiting for full evaluation.
    #   Default: None (use all test data)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Sample only first N examples from dataset (for quick testing).",
    )

    # ===== Energy Measurement Window =====
    # run-start-timestamp: ISO-formatted timestamp marking the start of the benchmark run.
    #   Allows alignment of MLflow run with detailed timestamped energy logs (CSV).
    #   The evaluation script measures energy. The CSV file on the GPU node provides
    #   an independent energy measurement for validation.
    #   Format: YYYY-MM-DDTHH:MM:SS.ssssss (with microseconds)
    #   Default: None (uses current time)
    parser.add_argument(
        "--run-start-timestamp",
        type=str,
        default=None,
        help="ISO timestamp for energy window start. Format: YYYY-MM-DDTHH:MM:SS.ssssss. Used to align with CSV energy metrics.",
    )
    args = parser.parse_args()

    # ===== Execute: Run Evaluation =====
    # After parsing arguments, launch the main evaluation pipeline.
    # Handles two modes:
    #   1. Single language: Run evaluate once for specified language
    #   2. All languages: Loop through all supported languages sequentially

    if args.language == "all":
        # ===== Multi-Language Mode =====
        # Run evaluation for all 21 supported languages sequentially in a loop.
        # Each language gets its own MLflow run with separate metrics.
        # Yoruba (yo) is skipped due to potential data quality issues.
        for lang in [l for l in SUPPORTED_LANGUAGES if l != "yo"]:
            try:
                print(f"\n\n===== Running for language: {lang} ({full_lang_name(lang)}) =====\n")
                # Switch args.language to current loop language
                args.language = lang
                # Call async run() function which orchestrates:
                #   1. Load language's test set from XTREME
                #   2. Create output directories
                #   3. Initialize MLflow tracking
                #   4. Run NER evaluation pipeline (batch inference → parsing → tagging → metrics)
                #   5. Save results (responses, metrics, telemetry)
                #   6. Log metrics to MLflow
                #   7. Compute energy-efficiency metrics (J/F1, F1/J, J/TP, etc.)
                #   8. Compute corrected energy from CSV file (validates vLLM measurements)
                asyncio.run(run(args))
            except Exception as e:
                # Log error but continue with next language (graceful degradation)
                print(f"Error processing language {lang}: {e}")
                continue
    else:
        # ===== Single Language Mode =====
        # Run evaluation once for the specified language (default: English).
        # Same pipeline as above, but single run.
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
