#!/bin/bash
# =============================================================================
# Sweep Orchestrator — RE (DocRED Relation Extraction)
# =============================================================================
# Submits one SLURM job per parameter variation (ablation study) for the RE task.
# Design: one parameter changes at a time; all others stay at baseline.
#
# Usage:
#   bash sweep_orchestrator_re.sh [--dry-run]
#
# Output:
#   - One SLURM job per (model, param, value) combination
#   - Job registry CSV: sweep_registry/re_job_registry_<timestamp>.csv
#   - Per-run env files: sweep_configs/re_run_<id>.env  (sourced by SLURM job)
#
# To monitor after submission:
#   squeue -u $USER
#   column -t -s, sweep_registry/re_job_registry_<timestamp>.csv | less -S
#
# To add a new model: edit MODELS array below.
# To add a new parameter variation: edit BASELINE and PARAM_VARIATIONS below.
#
# NOTE: calculate_kv_scales=true is intentionally excluded (vLLM 0.8.4 bug:
#   AttributeError in attention/layer.py when used with kv-cache-dtype fp8).
#
# NOTE: Supported RE models (as of re_lang_eval_mlflow_mi_gol_all9_semaphore.py):
#   /gemma-3-4b-it, /gemma-3-12b-it, /gemma-3-27b-it, /mistral, /llama-3-3-70B-it
# =============================================================================

set -euo pipefail

DRY_RUN=false
for _arg in "$@"; do
    [[ "${_arg}" == "--dry-run" ]] && DRY_RUN=true
done

REPO_ROOT="/path/to/repo"
HOST_BASE="${REPO_ROOT}/src/serve_vllm"
SWEEP_SLURM_SCRIPT="${HOST_BASE}/sweep_single_run_slurm_re.sh"
SWEEP_CONFIGS_DIR="${HOST_BASE}/sweep_configs"
REGISTRY_DIR="${HOST_BASE}/sweep_registry"
RESULTS_BASE="${HOST_BASE}/experiment_sweep_results/RE"

mkdir -p "${SWEEP_CONFIGS_DIR}" "${REGISTRY_DIR}" "${RESULTS_BASE}"
mkdir -p "${HOST_BASE}/slurm_out/slurm_sweep_re"

# =============================================================================
# ── Experiment dimensions ─────────────────────────────────────────────────────
# =============================================================================
MODELS=(
    "llama-3-3-70B-it"
    # "gemma-3-27b-it"
    # "gemma-3-12b-it"
    # "mistral"
)

# =============================================================================
# ── Baseline configuration ────────────────────────────────────────────────────
# =============================================================================
declare -A BASELINE=(
    # Inference script parameters
    [prompt_styles]="1,2,3,4,5,6,7,8"
    [max_concurrency]="256"
    [batch_size]="128"
    [max_new_tokens]="300"
    [sample_limit]="10000"
    # vLLM server parameters
    [max_model_len]="4096"
    [max_num_seqs]="512"
    [max_num_batched_tokens]="8192"
    [kv_cache_dtype]="fp8"
    [block_size]="16"
    [dtype]="bfloat16"
    [enable_prefix_caching]="true"
    [enable_chunked_prefill]="true"
    [calculate_kv_scales]="false"
    [tensor_parallel_size]="4"                                      # 27b fits on 1 H100
    [gpu_memory_util]="0.95"
    # Output format / decoding mode
    [format_mode]="json__false"
)

# =============================================================================
# ── Parameter variation lists ─────────────────────────────────────────────────
# =============================================================================
# One entry per parameter. Space-separated list of values to try.
# The baseline value should be INCLUDED so the baseline run covers it.
# Non-baseline values generate ablation runs.
#
# NOTE: calculate_kv_scales has only "false" — "true" is broken in vLLM 0.8.4.

declare -A PARAM_VARIATIONS=(
    [prompt_styles]="1,2,3,4,5,6,7,8"                          # always same; no ablation
    [max_concurrency]="256 512 1024"
    [batch_size]="64 128 256"
    [max_new_tokens]="150 200 300 400"
    [sample_limit]="10000"                                       # no ablation
    [max_model_len]="2048 4096 8192"
    [max_num_seqs]="256 512 1024"
    [max_num_batched_tokens]="4096 8192 16384"
    [kv_cache_dtype]="fp8 auto"
    [block_size]="16 32 64 128"
    [dtype]="bfloat16"                                           # no ablation
    [enable_prefix_caching]="true false"
    [enable_chunked_prefill]="true false"
    [calculate_kv_scales]="false"                                # no ablation (true is broken)
    [tensor_parallel_size]="2 4"                                  # 27b: baseline=1, ablate 2/4
    [gpu_memory_util]="0.95"                                     # no ablation
    # Format variants: json__false (base), yaml__false, dst__false, json__json_xgrammar
    [format_mode]="json__false yaml__false dst__false json__json_xgrammar"
)

# Ordered list of parameters to sweep (controls order in registry and loop)
SWEEP_PARAMS=(
    prompt_styles
    max_concurrency
    batch_size
    max_new_tokens
    max_model_len
    max_num_seqs
    max_num_batched_tokens
    kv_cache_dtype
    block_size
    dtype
    enable_prefix_caching
    enable_chunked_prefill
    calculate_kv_scales
    tensor_parallel_size
    gpu_memory_util
    sample_limit
    format_mode
)

# =============================================================================
# ── Helper: get value for a param (baseline or override) ─────────────────────
# =============================================================================
param_val() {
    local param="$1"
    local override_param="$2"
    local override_val="$3"
    if [[ "$param" == "$override_param" ]]; then
        echo "$override_val"
    else
        echo "${BASELINE[$param]}"
    fi
}

# =============================================================================
# ── Helper: write per-run env file ───────────────────────────────────────────
# =============================================================================
write_env_file() {
    local env_file="$1"
    local model="$2"
    local override_param="$3"
    local override_val="$4"
    local mlflow_exp="$5"
    local artifacts_base="$6"

    cat > "${env_file}" <<EOF
# Auto-generated by sweep_orchestrator_re.sh — do not edit manually.
# (model=${model}, task=re, param_changed=${override_param}, val=${override_val})

SWEEP_MODEL="${model}"
SWEEP_TASK="re"
SWEEP_SPLIT="validation"
SWEEP_PARAM_CHANGED="${override_param}"
SWEEP_PARAM_VALUE="${override_val}"
SWEEP_MLFLOW_EXPERIMENT="${mlflow_exp}"
SWEEP_ARTIFACTS_BASE="${artifacts_base}"

# Inference script parameters
SWEEP_PROMPT_STYLES="$(param_val prompt_styles "$override_param" "$override_val")"
SWEEP_MAX_CONCURRENCY="$(param_val max_concurrency "$override_param" "$override_val")"
SWEEP_BATCH_SIZE="$(param_val batch_size "$override_param" "$override_val")"
SWEEP_MAX_NEW_TOKENS="$(param_val max_new_tokens "$override_param" "$override_val")"
SWEEP_SAMPLE_LIMIT="$(param_val sample_limit "$override_param" "$override_val")"

# vLLM server parameters
SWEEP_MAX_MODEL_LEN="$(param_val max_model_len "$override_param" "$override_val")"
SWEEP_MAX_NUM_SEQS="$(param_val max_num_seqs "$override_param" "$override_val")"
SWEEP_MAX_NUM_BATCHED_TOKENS="$(param_val max_num_batched_tokens "$override_param" "$override_val")"
SWEEP_KV_CACHE_DTYPE="$(param_val kv_cache_dtype "$override_param" "$override_val")"
SWEEP_BLOCK_SIZE="$(param_val block_size "$override_param" "$override_val")"
SWEEP_DTYPE="$(param_val dtype "$override_param" "$override_val")"
SWEEP_ENABLE_PREFIX_CACHING="$(param_val enable_prefix_caching "$override_param" "$override_val")"
SWEEP_ENABLE_CHUNKED_PREFILL="$(param_val enable_chunked_prefill "$override_param" "$override_val")"
SWEEP_CALCULATE_KV_SCALES="$(param_val calculate_kv_scales "$override_param" "$override_val")"
SWEEP_TENSOR_PARALLEL_SIZE="$(param_val tensor_parallel_size "$override_param" "$override_val")"
SWEEP_GPU_MEMORY_UTIL="$(param_val gpu_memory_util "$override_param" "$override_val")"
SWEEP_FORMAT_MODE="$(param_val format_mode "$override_param" "$override_val")"
EOF
}

# =============================================================================
# ── Submit a single job ───────────────────────────────────────────────────────
# =============================================================================
submit_job() {
    local env_file="$1"
    local job_name="$2"

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "[DRY-RUN] Would submit: ${job_name} (env: ${env_file})"
        echo "DRY_RUN_FAKE_ID"
        return
    fi

    sbatch \
        --job-name="${job_name}" \
        --output="${HOST_BASE}/slurm_out/slurm_sweep_re/slurm_%x_%j.out" \
        --error="${HOST_BASE}/slurm_out/slurm_sweep_re/slurm_%x_%j.err" \
        --export="ALL,SWEEP_ENV_FILE=${env_file}" \
        "${SWEEP_SLURM_SCRIPT}" \
    | awk '{print $NF}'
}

# =============================================================================
# ── Registry setup ────────────────────────────────────────────────────────────
# =============================================================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REGISTRY_FILE="${REGISTRY_DIR}/re_job_registry_${TIMESTAMP}.csv"

printf '%s\n' \
    "job_id,submitted_at,model,task,split,param_changed,param_value,prompt_styles,max_concurrency,batch_size,max_new_tokens,max_model_len,max_num_seqs,max_num_batched_tokens,kv_cache_dtype,block_size,dtype,enable_prefix_caching,enable_chunked_prefill,calculate_kv_scales,tensor_parallel_size,gpu_memory_util,mlflow_experiment,artifacts_base,env_file" \
    > "${REGISTRY_FILE}"

echo "[INFO] RE Sweep started: ${TIMESTAMP}"
echo "[INFO] Registry: ${REGISTRY_FILE}"
echo "[INFO] Dry-run: ${DRY_RUN}"
echo ""

# =============================================================================
# ── Main sweep loop ───────────────────────────────────────────────────────────
# =============================================================================
for model in "${MODELS[@]}"; do
    MLFLOW_EXP="sweep_re_${model}_${TIMESTAMP}"
    ARTIFACTS_BASE="${RESULTS_BASE}/sweep_re_${model}_${TIMESTAMP}"
    mkdir -p "${ARTIFACTS_BASE}"

    echo "[INFO] ==========================================================="
    echo "[INFO] RE Sweep: model=${model}  task=re  split=validation"
    echo "[INFO] MLflow experiment: ${MLFLOW_EXP}"
    echo "[INFO] Artifacts base: ${ARTIFACTS_BASE}"
    echo "[INFO] ==========================================================="

    # ── 1. Baseline run ──────────────────────────────────────────────────────
    BL_ENV="${SWEEP_CONFIGS_DIR}/re_run_${model}_BASELINE_${TIMESTAMP}.env"
    write_env_file "${BL_ENV}" "${model}" \
        "BASELINE" "BASELINE" "${MLFLOW_EXP}" "${ARTIFACTS_BASE}"

    JOB_NAME="sw_re_bl_${model:0:8}"
    JOB_ID=$(submit_job "${BL_ENV}" "${JOB_NAME}")
    SUBMITTED_AT=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "[INFO] Baseline job submitted: ${JOB_ID}"

    printf '%s\n' \
        "${JOB_ID},${SUBMITTED_AT},${model},re,validation,BASELINE,BASELINE,\"${BASELINE[prompt_styles]}\",${BASELINE[max_concurrency]},${BASELINE[batch_size]},${BASELINE[max_new_tokens]},${BASELINE[max_model_len]},${BASELINE[max_num_seqs]},${BASELINE[max_num_batched_tokens]},${BASELINE[kv_cache_dtype]},${BASELINE[block_size]},${BASELINE[dtype]},${BASELINE[enable_prefix_caching]},${BASELINE[enable_chunked_prefill]},${BASELINE[calculate_kv_scales]},${BASELINE[tensor_parallel_size]},${BASELINE[gpu_memory_util]},${MLFLOW_EXP},${ARTIFACTS_BASE},${BL_ENV}" \
        >> "${REGISTRY_FILE}"

    # ── 2. Ablation runs ─────────────────────────────────────────────────────
    for param in "${SWEEP_PARAMS[@]}"; do
        IFS=' ' read -ra values <<< "${PARAM_VARIATIONS[${param}]}"

        for val in "${values[@]}"; do
            # Skip baseline value (already covered by the baseline run above)
            if [[ "${val}" == "${BASELINE[${param}]}" ]]; then
                continue
            fi

            # Sanitize value for use in file/job names
            val_clean="${val//,/_}"
            val_clean="${val_clean//__/_}"
            val_clean="${val_clean:0:20}"

            ENV_FILE="${SWEEP_CONFIGS_DIR}/re_run_${model}_${param}_${val_clean}_${TIMESTAMP}.env"
            write_env_file "${ENV_FILE}" "${model}" \
                "${param}" "${val}" "${MLFLOW_EXP}" "${ARTIFACTS_BASE}"

            JOB_NAME="sw_re_${param:0:6}_${val_clean:0:6}"
            JOB_ID=$(submit_job "${ENV_FILE}" "${JOB_NAME}")
            SUBMITTED_AT=$(date +"%Y-%m-%dT%H:%M:%S")
            echo "[INFO]   ${param}=${val}  →  job ${JOB_ID}"

            printf '%s\n' \
                "${JOB_ID},${SUBMITTED_AT},${model},re,validation,${param},${val},\"$(param_val prompt_styles "${param}" "${val}")\",$(param_val max_concurrency "${param}" "${val}"),$(param_val batch_size "${param}" "${val}"),$(param_val max_new_tokens "${param}" "${val}"),$(param_val max_model_len "${param}" "${val}"),$(param_val max_num_seqs "${param}" "${val}"),$(param_val max_num_batched_tokens "${param}" "${val}"),$(param_val kv_cache_dtype "${param}" "${val}"),$(param_val block_size "${param}" "${val}"),$(param_val dtype "${param}" "${val}"),$(param_val enable_prefix_caching "${param}" "${val}"),$(param_val enable_chunked_prefill "${param}" "${val}"),$(param_val calculate_kv_scales "${param}" "${val}"),$(param_val tensor_parallel_size "${param}" "${val}"),$(param_val gpu_memory_util "${param}" "${val}"),${MLFLOW_EXP},${ARTIFACTS_BASE},${ENV_FILE}" \
                >> "${REGISTRY_FILE}"
        done
    done

    echo ""
done

echo "[INFO] All RE jobs submitted."
echo "[INFO] Registry: ${REGISTRY_FILE}"
echo ""
echo "  Monitor jobs:     squeue -u \$USER"
echo "  View registry:    column -t -s, ${REGISTRY_FILE} | less -S"
echo "  Count submitted:  wc -l ${REGISTRY_FILE}"
