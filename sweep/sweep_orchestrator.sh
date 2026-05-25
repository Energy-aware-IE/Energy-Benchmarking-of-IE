#!/bin/bash
# =============================================================================
# Sweep Orchestrator
# =============================================================================
# Submits one SLURM job per parameter variation (ablation study).
# Design: one parameter changes at a time; all others stay at baseline.
#
# Usage:
#   bash sweep_orchestrator.sh [--dry-run]
#
# Output:
#   - One SLURM job per (model, language, param, value) combination
#   - Job registry CSV: sweep_registry/job_registry_<timestamp>.csv
#   - Per-run env files: sweep_configs/run_<id>.env  (sourced by SLURM job)
#
# To monitor after submission:
#   squeue -u $USER
#   column -t -s, sweep_registry/job_registry_<timestamp>.csv | less -S
#
# To add a new model or language: edit MODELS / LANGUAGES arrays below.
# To add a new parameter variation: edit BASELINE and PARAM_VARIATIONS below.
# =============================================================================

set -euo pipefail

DRY_RUN=false
MULTILANG_MODE=false
for _arg in "$@"; do
    [[ "${_arg}" == "--dry-run"   ]] && DRY_RUN=true
    [[ "${_arg}" == "--multilang" ]] && MULTILANG_MODE=true
done

REPO_ROOT="/path/to/repo"
HOST_BASE="${REPO_ROOT}/src/serve_vllm"
SWEEP_SLURM_SCRIPT="${HOST_BASE}/sweep_single_run_slurm.sh"
SWEEP_CONFIGS_DIR="${HOST_BASE}/sweep_configs"
REGISTRY_DIR="${HOST_BASE}/sweep_registry"

# =============================================================================
# ── Multilang mode settings ───────────────────────────────────────────────────
# =============================================================================
# Pass --multilang to use all XTREME languages, a dedicated artifacts dir under
# experiment_sweep_results/multilang_xtream/, and experiment names prefixed with
# "multilang_sweep_" so they are distinct from single-language runs in MLflow.
if [[ "${MULTILANG_MODE}" == "true" ]]; then
    MLFLOW_EXP_PREFIX="multilang_sweep_"
    RESULTS_BASE="${HOST_BASE}/experiment_sweep_results/multilang_xtream"
else
    MLFLOW_EXP_PREFIX="sweep_"
    RESULTS_BASE="${HOST_BASE}/experiment_sweep_results"
fi

mkdir -p "${SWEEP_CONFIGS_DIR}" "${REGISTRY_DIR}" "${RESULTS_BASE}"
mkdir -p "${HOST_BASE}/slurm_out/slurm_sweep"

# =============================================================================
# ── Experiment dimensions ─────────────────────────────────────────────────────
# =============================================================================
# Add more models or languages to extend the sweep matrix.
# Each (model, language) pair gets its own MLflow experiment and artifacts dir.

MODELS=(
    "llama-3-3-70B-it"
    # "gemma-3-27b-it"
    # "gemma-3-12b-it"
    # "mistral"
)

LANGUAGES=(
    "de"
    # "en" "de" "fr" "zh" "ar" "hi" "ko" "ja" "ru" "es"
    # "bg" "el" "id" "it" "nl" "pt" "th" "tr" "ur" "vi" "yo"
)

# --multilang overrides the LANGUAGES array with all 10 core XTREME languages.
# Edit the array below to add/remove languages for the multilang run.
LANGUAGES_MULTILANG=(
    "en" "de" "fr" "zh" "ar" "hi" "ko" "ja" "ru" "es"
)
if [[ "${MULTILANG_MODE}" == "true" ]]; then
    LANGUAGES=("${LANGUAGES_MULTILANG[@]}")
fi

# =============================================================================
# ── Baseline configuration ────────────────────────────────────────────────────
# =============================================================================
# All runs that vary parameter P use these values for every other parameter.
# A dedicated "baseline" run is submitted first for each (model, language).

declare -A BASELINE=(
    # Inference script parameters
    [prompt_styles]="1,2,3,4,5,6,7,8,9"
    [max_concurrency]="256"
    [batch_size]="1024"
    [max_new_tokens]="150"
    [sample_limit]="10000"
    # Output format / constrained decoding
    # Allowed combined values (file_type__constrained_decoding):
    #   json__false | json__json_xgrammar | dst__false | dst__dst_ebnf
    #   dst__dst_outlines | xml__false | xml__xml_xgrammar | yaml__false
    [format_mode]="json__false"
    # vLLM server parameters
    [max_model_len]="1024"
    [max_num_seqs]="1024"
    [max_num_batched_tokens]="8192"
    [kv_cache_dtype]="fp8"
    [block_size]="16"
    [dtype]="bfloat16"
    [enable_prefix_caching]="true"
    [enable_chunked_prefill]="true"
    [calculate_kv_scales]="false"
    [tensor_parallel_size]="4"
    [gpu_memory_util]="0.95"
)

# =============================================================================
# ── Parameter variation lists ─────────────────────────────────────────────────
# =============================================================================
# One entry per parameter. Space-separated list of values to try.
# The baseline value should be INCLUDED so the baseline run covers it.
# Non-baseline values generate ablation runs.
#
# Parameters with a single value are included for completeness but won't
# generate ablation runs (only a baseline run for that parameter exists).
#
# format_mode encodes both file_type and constrained_decoding as a coupled pair
# because changing file_type implies a different inference script.

declare -A PARAM_VARIATIONS=(
    [prompt_styles]="1,2,3,4,5,6,7,8,9"                          # always same; no ablation
    [max_concurrency]="256 512 1024"
    [batch_size]="128 512 1024"
    [max_new_tokens]="50 100 150 200 250 300"
    [sample_limit]="10000"                                         # no ablation
    [format_mode]="json__false json__json_xgrammar dst__false dst__dst_ebnf dst__dst_outlines xml__false xml__xml_xgrammar yaml__false"
    [max_model_len]="768 1024 2048"
    [max_num_seqs]="512 1024 2048"
    [max_num_batched_tokens]="8192 16384 32768"
    [kv_cache_dtype]="fp8 auto"
    [block_size]="16 32 64 128"
    [dtype]="bfloat16"                                             # no ablation
    [enable_prefix_caching]="true false"
    [enable_chunked_prefill]="true false"
    [calculate_kv_scales]="false true"
    [tensor_parallel_size]="2 4"                                     # no ablation
    [gpu_memory_util]="0.95"                                       # no ablation
)

# Ordered list of parameters to sweep (controls order in registry and loop)
SWEEP_PARAMS=(
    prompt_styles
    max_concurrency
    batch_size
    max_new_tokens
    format_mode
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
    local lang="$3"
    local override_param="$4"
    local override_val="$5"
    local mlflow_exp="$6"
    local artifacts_base="$7"

    cat > "${env_file}" <<EOF
# Auto-generated by sweep_orchestrator.sh — do not edit manually.
# (model=${model}, lang=${lang}, param_changed=${override_param}, val=${override_val})

SWEEP_MODEL="${model}"
SWEEP_LANGUAGE="${lang}"
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
SWEEP_FORMAT_MODE="$(param_val format_mode "$override_param" "$override_val")"

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
        --output="${HOST_BASE}/slurm_out/slurm_sweep/slurm_%x_%j.out" \
        --error="${HOST_BASE}/slurm_out/slurm_sweep/slurm_%x_%j.err" \
        --export="ALL,SWEEP_ENV_FILE=${env_file}" \
        "${SWEEP_SLURM_SCRIPT}" \
    | awk '{print $NF}'
}

# =============================================================================
# ── Registry setup ────────────────────────────────────────────────────────────
# =============================================================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REGISTRY_FILE="${REGISTRY_DIR}/job_registry_${TIMESTAMP}.csv"

# CSV header
printf '%s\n' \
    "job_id,submitted_at,model,language,param_changed,param_value,prompt_styles,max_concurrency,batch_size,max_new_tokens,format_mode,max_model_len,max_num_seqs,max_num_batched_tokens,kv_cache_dtype,block_size,dtype,enable_prefix_caching,enable_chunked_prefill,calculate_kv_scales,tensor_parallel_size,gpu_memory_util,mlflow_experiment,artifacts_base,env_file" \
    > "${REGISTRY_FILE}"

echo "[INFO] Sweep started: ${TIMESTAMP}"
echo "[INFO] Registry: ${REGISTRY_FILE}"
echo "[INFO] Dry-run: ${DRY_RUN}"
echo ""

# =============================================================================
# ── Main sweep loop ───────────────────────────────────────────────────────────
# =============================================================================
for model in "${MODELS[@]}"; do
    _MULTILANG_LANGS_FOR_MODEL=()
    for lang in "${LANGUAGES[@]}"; do

        # Unique MLflow experiment and artifacts dir for this (model, language) sweep
        MLFLOW_EXP="${MLFLOW_EXP_PREFIX}${lang}_${model}_${TIMESTAMP}"
        ARTIFACTS_BASE="${RESULTS_BASE}/sweep_${lang}_${model}_${TIMESTAMP}"
        mkdir -p "${ARTIFACTS_BASE}"

        echo "[INFO] ============================================================"
        echo "[INFO] Sweep: model=${model}  language=${lang}"
        echo "[INFO] MLflow experiment: ${MLFLOW_EXP}"
        echo "[INFO] Artifacts base: ${ARTIFACTS_BASE}"
        echo "[INFO] ============================================================"

        # ── 1. Baseline run ────────────────────────────────────────────────────
        BL_ENV="${SWEEP_CONFIGS_DIR}/run_${model}_${lang}_BASELINE_${TIMESTAMP}.env"
        write_env_file "${BL_ENV}" "${model}" "${lang}" \
            "BASELINE" "BASELINE" "${MLFLOW_EXP}" "${ARTIFACTS_BASE}"

        JOB_NAME="sw_bl_${model:0:6}_${lang}"
        JOB_ID=$(submit_job "${BL_ENV}" "${JOB_NAME}")
        SUBMITTED_AT=$(date +"%Y-%m-%dT%H:%M:%S")
        echo "[INFO] Baseline job submitted: ${JOB_ID}"

        # Append baseline row to registry
        printf '%s\n' "${JOB_ID},${SUBMITTED_AT},${model},${lang},BASELINE,BASELINE,\"${BASELINE[prompt_styles]}\",${BASELINE[max_concurrency]},${BASELINE[batch_size]},${BASELINE[max_new_tokens]},\"${BASELINE[format_mode]}\",${BASELINE[max_model_len]},${BASELINE[max_num_seqs]},${BASELINE[max_num_batched_tokens]},${BASELINE[kv_cache_dtype]},${BASELINE[block_size]},${BASELINE[dtype]},${BASELINE[enable_prefix_caching]},${BASELINE[enable_chunked_prefill]},${BASELINE[calculate_kv_scales]},${BASELINE[tensor_parallel_size]},${BASELINE[gpu_memory_util]},${MLFLOW_EXP},${ARTIFACTS_BASE},${BL_ENV}" \
            >> "${REGISTRY_FILE}"

        # ── 2. Ablation runs ───────────────────────────────────────────────────
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
                val_clean="${val_clean:0:20}"  # truncate long values

                ENV_FILE="${SWEEP_CONFIGS_DIR}/run_${model}_${lang}_${param}_${val_clean}_${TIMESTAMP}.env"
                write_env_file "${ENV_FILE}" "${model}" "${lang}" \
                    "${param}" "${val}" "${MLFLOW_EXP}" "${ARTIFACTS_BASE}"

                JOB_NAME="sw_${param:0:6}_${val_clean:0:6}_${lang}"
                JOB_ID=$(submit_job "${ENV_FILE}" "${JOB_NAME}")
                SUBMITTED_AT=$(date +"%Y-%m-%dT%H:%M:%S")
                echo "[INFO]   ${param}=${val}  →  job ${JOB_ID}"

                # Build row with current config (baseline + this override)
                printf '%s\n' \
                    "${JOB_ID},${SUBMITTED_AT},${model},${lang},${param},${val},\"$(param_val prompt_styles "${param}" "${val}")\",$(param_val max_concurrency "${param}" "${val}"),$(param_val batch_size "${param}" "${val}"),$(param_val max_new_tokens "${param}" "${val}"),\"$(param_val format_mode "${param}" "${val}")\",$(param_val max_model_len "${param}" "${val}"),$(param_val max_num_seqs "${param}" "${val}"),$(param_val max_num_batched_tokens "${param}" "${val}"),$(param_val kv_cache_dtype "${param}" "${val}"),$(param_val block_size "${param}" "${val}"),$(param_val dtype "${param}" "${val}"),$(param_val enable_prefix_caching "${param}" "${val}"),$(param_val enable_chunked_prefill "${param}" "${val}"),$(param_val calculate_kv_scales "${param}" "${val}"),$(param_val tensor_parallel_size "${param}" "${val}"),$(param_val gpu_memory_util "${param}" "${val}"),${MLFLOW_EXP},${ARTIFACTS_BASE},${ENV_FILE}" \
                    >> "${REGISTRY_FILE}"
            done
        done

        echo ""
        _MULTILANG_LANGS_FOR_MODEL+=("${lang}")
    done

    # ── After all languages: write multilang summary env for the analysis script ──
    if [[ "${MULTILANG_MODE}" == "true" ]]; then
        MULTILANG_OUT_DIR="${RESULTS_BASE}/multilang_${model}_${TIMESTAMP}"
        mkdir -p "${MULTILANG_OUT_DIR}"
        SUMMARY_ENV="${REGISTRY_DIR}/multilang_${model}_${TIMESTAMP}.env"
        MLFLOW_TRACKING_URI_VAL="${MLFLOW_TRACKING_URI:-$(grep -oP '(?<=MLFLOW_TRACKING_URI=)[^\s]+' "${HOST_BASE}/.env" 2>/dev/null | head -1 || echo '')}"
        LANG_CSV="$(IFS=,; echo "${_MULTILANG_LANGS_FOR_MODEL[*]}")"
        cat > "${SUMMARY_ENV}" <<EOF
# Auto-generated by sweep_orchestrator.sh --multilang
MULTILANG_GROUP="${TIMESTAMP}"
MULTILANG_MODEL="${model}"
MULTILANG_LANGUAGES="${LANG_CSV}"
MULTILANG_REGISTRY="${REGISTRY_FILE}"
MULTILANG_OUT_DIR="${MULTILANG_OUT_DIR}"
MULTILANG_SUMMARY_EXPERIMENT="multilang_summary_${model}_${TIMESTAMP}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI_VAL}"
EOF
        echo "[INFO] ============================================================"
        echo "[INFO] Multilang summary env: ${SUMMARY_ENV}"
        echo "[INFO] After all jobs complete, run:"
        echo "[INFO]   source ${HOST_BASE}/.env && python ${HOST_BASE}/multilang_analysis.py --env ${SUMMARY_ENV}"
        echo "[INFO] ============================================================"
    fi
done

echo "[INFO] All jobs submitted."
echo "[INFO] Registry: ${REGISTRY_FILE}"
echo ""
echo "  Monitor jobs:     squeue -u \$USER"
echo "  View registry:    column -t -s, ${REGISTRY_FILE} | less -S"
echo "  Count submitted:  wc -l ${REGISTRY_FILE}"
