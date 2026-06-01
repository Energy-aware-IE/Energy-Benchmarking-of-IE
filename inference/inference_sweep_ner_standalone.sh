#!/bin/bash
# =============================================================================
# Sweep NER Inference Task
# =============================================================================
# Runs NER evaluation (lang_eval_mlflow_mi_gol_all9_semaphore.py) across all
# requested prompt styles for a sweep run.
#
# Called by sweep_single_run_slurm.sh (via srun bash), NOT submitted directly.
#
# Required env vars (all exported by sweep_single_run_slurm.sh):
#   SWEEP_LANGUAGE, SWEEP_PROMPT_STYLES, SWEEP_BATCH_SIZE,
#   SWEEP_MAX_NEW_TOKENS, SWEEP_MAX_CONCURRENCY, SWEEP_SAMPLE_LIMIT,
#   SWEEP_MLFLOW_EXPERIMENT, ARTIFACTS_DIR, METRICS_CSV,
#   HOST_BASE, VENV_PATH, MODEL_NAME, SLURM_JOB_ID,
#   METRICS_SCRIPT, METRICS_INTERVAL_MS, EXPERIMENT_RESULT_BASE
# =============================================================================

ulimit -n 65536

# =============================================================================
# ── Read sweep parameters from env ───────────────────────────────────────────
# =============================================================================
language="${SWEEP_LANGUAGE}"
batch_size="${SWEEP_BATCH_SIZE}"
max_new_tokens="${SWEEP_MAX_NEW_TOKENS}"
max_concurrency="${SWEEP_MAX_CONCURRENCY}"
sample_limit="${SWEEP_SAMPLE_LIMIT:-10000}"
cooldown_seconds=10

# Split comma-separated prompt styles into array (e.g. "1,2,3,4,5,6,7,8,9")
IFS=',' read -ra prompt_styles <<< "${SWEEP_PROMPT_STYLES}"

# Resolve repository root from this script location so evaluation paths stay portable.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_LOCAL="$(cd "${SCRIPT_DIR}/.." && pwd)"

# DSPy program paths (same as inference_gemma_multi_prompt_standalone.sh)
DSPY_PROGRAM_FILE="${HOST_BASE}/prompts_in_all_languages/optimized_ner_en.json"
DSPY_PROGRAM_FILE_MATCHED="${HOST_BASE}/prompts_in_all_languages/optimized_ner_en_matched.json"

# =============================================================================
# ── Select Python evaluation script and extra args based on format_mode ───────
# =============================================================================
# SWEEP_FILE_TYPE and SWEEP_CONSTRAINED are exported by sweep_single_run_slurm.sh
# from SWEEP_FORMAT_MODE (e.g. "json__false" → file_type=json, constrained=false)
#
# Python script mapping:
#   json__false                          → lang_eval_mlflow_mi_gol_all9_semaphore.py        (no extra arg)
#   json__json_xgrammar                  → lang_eval_mlflow_mi_gol_all9_semaphore_guided.py  --decoding-mode guided_json
#   dst__dst_ebnf                        → lang_eval_mlflow_mi_gol_all9_semaphore_guided.py  --decoding-mode guided_dst
#   dst__dst_outlines                    → lang_eval_mlflow_mi_gol_all9_semaphore_guided.py  --decoding-mode guided_dst_outlines
#   xml__xml_xgrammar                    → lang_eval_mlflow_mi_gol_all9_semaphore_guided.py  --decoding-mode guided_xml
#   dst__false | xml__false | yaml__false → lang_eval_mlflow_mi_gol_all9_semaphore_other_formats.py  --output-format <file_type>

EVAL_PY="${REPO_ROOT_LOCAL}/evaluation/xtreme/lang_eval_ner_sweep.py"
FORMAT_EXTRA_ARG=""

case "${SWEEP_FORMAT_MODE:-json__false}" in
    json__false|json__json_xgrammar|dst__false|dst__dst_ebnf|dst__dst_outlines|xml__false|xml__xml_xgrammar|yaml__false)
        ;;
    *)
        echo "[WARN] Unknown SWEEP_FORMAT_MODE '${SWEEP_FORMAT_MODE}', continuing with default evaluator."
        ;;
esac

echo "[INFO] Format mode     : ${SWEEP_FORMAT_MODE}"
echo "[INFO] Eval script     : ${EVAL_PY}"
echo "[INFO] Format extra arg: ${FORMAT_EXTRA_ARG:-<none>}"

# =============================================================================
# ── Log inherited context ─────────────────────────────────────────────────────
# =============================================================================
echo "[INFO] === NER Sweep Inference Task ==="
echo "[INFO] Language       : ${language}"
echo "[INFO] Prompt styles  : ${SWEEP_PROMPT_STYLES}"
echo "[INFO] Batch size     : ${batch_size}"
echo "[INFO] Max new tokens : ${max_new_tokens}"
echo "[INFO] Max concurrency: ${max_concurrency}"
echo "[INFO] Sample limit   : ${sample_limit}"
echo "[INFO] MLflow exp     : ${SWEEP_MLFLOW_EXPERIMENT}"
echo "[INFO] Artifacts dir  : ${ARTIFACTS_DIR}"
echo "[INFO] Metrics CSV    : ${METRICS_CSV}"
echo "[INFO] Model          : ${MODEL_NAME}"

# =============================================================================
# ── Source .env (MLflow credentials, ENERGY_URL, etc.) ───────────────────────
# =============================================================================
if [[ -f "${HOST_BASE}/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${HOST_BASE}/.env"
    set +a
fi

if [[ -z "${ENERGY_URL:-}" ]]; then
    echo "[FATAL] ENERGY_URL is not set — check ${HOST_BASE}/.env"
    exit 2
fi

export METRICS_INTERVAL_MS

# =============================================================================
# ── Activate venv ─────────────────────────────────────────────────────────────
# =============================================================================
# shellcheck source=/dev/null
source "${VENV_PATH}/bin/activate"

# =============================================================================
# ── Start NVML GPU metrics scraper ───────────────────────────────────────────
# =============================================================================
echo "[INFO] Starting GPU metrics scraper..."
nohup python3 "${METRICS_SCRIPT}" \
    > "${ARTIFACTS_DIR}/metrics_scraper_${SLURM_JOB_ID}.log" 2>&1 &
SCRAPER_PID=$!
echo "[INFO] Scraper PID: ${SCRAPER_PID}"

trap 'echo "[INFO] Stopping scraper PID ${SCRAPER_PID}"; kill "${SCRAPER_PID}" 2>/dev/null || true' EXIT

# =============================================================================
# ── Inference loop over prompt styles ─────────────────────────────────────────
# =============================================================================
cd "${HOST_BASE}" || exit 1

# =============================================================================
# ── Write initial out_inference.json (updated again after loop) ───────────────
# =============================================================================
# Written BEFORE the prompt-style loop so each per-style Python eval script
# can discover and log it as an MLflow artifact (the Python scripts glob for
# *out_inference.json inside ARTIFACTS_DIR during each run).
SERVER_CONFIG_SRC="${ARTIFACTS_DIR}/server_config_${SLURM_JOB_ID}.json"
OUT_INFERENCE_CONFIG="${ARTIFACTS_DIR}/sweep_${SWEEP_PARAM_CHANGED}_job_${SLURM_JOB_ID}_m_${MODEL_NAME}_l_${language}_out_inference.json"

_write_inference_config() {
    local exit_code="${1:-null}"
    local completed_at
    completed_at="$(date -u +'%Y-%m-%d %H:%M:%S')"

    if [[ -f "${SERVER_CONFIG_SRC}" ]]; then
        sed '$ s/}//' "${SERVER_CONFIG_SRC}" | sed '$ s/,*$/,/' > "${OUT_INFERENCE_CONFIG}"
        cat >> "${OUT_INFERENCE_CONFIG}" <<EOF_JSON
  "inference_language": "${language}",
  "prompt_styles": "$(IFS=,; echo "${prompt_styles[*]}")",
  "batch_size": ${batch_size},
  "max_new_tokens": ${max_new_tokens},
  "max_concurrency": ${max_concurrency},
  "sample_limit": ${sample_limit},
  "cooldown_seconds": ${cooldown_seconds},
  "artifacts_dir": "${ARTIFACTS_DIR}",
  "metrics_csv": "${METRICS_CSV}",
  "format_mode": "${SWEEP_FORMAT_MODE}",
  "param_changed": "${SWEEP_PARAM_CHANGED}",
  "param_value": "${SWEEP_PARAM_VALUE}",
  "inference_completed_at": "${completed_at}",
  "inference_exit_code": ${exit_code}
}
EOF_JSON
    else
        echo "[WARNING] server_config not found at ${SERVER_CONFIG_SRC}; writing minimal out_inference.json"
        cat > "${OUT_INFERENCE_CONFIG}" <<EOF_JSON
{
  "job_id": "${SLURM_JOB_ID}",
  "model_name": "${MODEL_NAME}",
  "inference_language": "${language}",
  "prompt_styles": "$(IFS=,; echo "${prompt_styles[*]}")",
  "batch_size": ${batch_size},
  "max_new_tokens": ${max_new_tokens},
  "max_concurrency": ${max_concurrency},
  "sample_limit": ${sample_limit},
  "cooldown_seconds": ${cooldown_seconds},
  "artifacts_dir": "${ARTIFACTS_DIR}",
  "metrics_csv": "${METRICS_CSV}",
  "format_mode": "${SWEEP_FORMAT_MODE}",
  "param_changed": "${SWEEP_PARAM_CHANGED}",
  "param_value": "${SWEEP_PARAM_VALUE}",
  "inference_completed_at": "${completed_at}",
  "inference_exit_code": ${exit_code}
}
EOF_JSON
    fi
}

# Write initial version (exit_code not yet known)
_write_inference_config "null"
echo "[INFO] Initial out_inference config written: ${OUT_INFERENCE_CONFIG}"

LAST_EXIT_CODE=0

for style in "${prompt_styles[@]}"; do
    echo "[INFO] ===== Prompt style ${style} ====="

    # Skip this prompt style if its mean_metrics file already exists in ARTIFACTS_DIR.
    # This allows resubmitted jobs to resume from where they stopped rather than
    # re-running prompts that already completed and were logged to MLflow.
    if compgen -G "${ARTIFACTS_DIR}/mean_metrics_B_${batch_size}_prompt${style}_*.json" > /dev/null 2>&1; then
        echo "[INFO] Prompt style ${style}: mean_metrics already exists — skipping (resume mode)."
        continue
    fi

    start_timestamp=$(date +"%Y-%m-%dT%H:%M:%S.%6N")

    # Select extra arguments based on style number
    if [[ "${style}" -eq 9 ]]; then
        EXTRA_STYLE_ARGS="--dspy-program-path ${DSPY_PROGRAM_FILE}"
    elif [[ "${style}" -eq 10 ]]; then
        EXTRA_STYLE_ARGS="--dspy-program-path ${DSPY_PROGRAM_FILE_MATCHED}"
    else
        # Styles 1-8: use generic (English-instruction) template
        EXTRA_STYLE_ARGS="--use-generic-template"
    fi

    # Run evaluation for this prompt style directly (no nested srun).
    # All SWEEP_* and MLFLOW_* env vars are already exported by sweep_single_run_slurm.sh.
    python3 "${EVAL_PY}" \
            --language "${language}" \
            --model "/${MODEL_NAME}" \
            --system-prompt-choice "${style}" \
            --batch-size "${batch_size}" \
            --max-new-tokens "${max_new_tokens}" \
            --max-concurrency "${max_concurrency}" \
            --limit "${sample_limit}" \
            --run-start-timestamp "${start_timestamp}" \
            ${FORMAT_EXTRA_ARG} \
            ${EXTRA_STYLE_ARGS}

    STYLE_EXIT_CODE=$?
    echo "[INFO] Prompt style ${style} finished: exit code ${STYLE_EXIT_CODE}"
    [[ ${STYLE_EXIT_CODE} -ne 0 ]] && LAST_EXIT_CODE=${STYLE_EXIT_CODE}

    # Cooldown between prompt styles so GPU utilization resets between runs
    # (ensures energy_corrected windows are cleanly separated per style)
    if [[ "${style}" != "${prompt_styles[-1]}" ]]; then
        echo "[INFO] Cooling down for ${cooldown_seconds}s before next prompt style..."
        sleep ${cooldown_seconds}
    fi
done

echo "[INFO] All prompt styles completed. Last exit code: ${LAST_EXIT_CODE}"

# =============================================================================
# ── Final update to out_inference.json (with exit code + completion time) ──────
# =============================================================================
_write_inference_config "${LAST_EXIT_CODE}"
echo "[INFO] Final out_inference config updated: ${OUT_INFERENCE_CONFIG}"

exit ${LAST_EXIT_CODE}
