#!/bin/bash
# =============================================================================
# Sweep Single Run — SLURM Job Script (RE / DocRED)
# =============================================================================
# One SLURM job = one parameter configuration for RE sweep.
# All parameters are read from the env file pointed to by SWEEP_ENV_FILE.
# Do NOT submit this script directly — use sweep_orchestrator_re.sh.
#
# Required: SWEEP_ENV_FILE must be set (done by sweep_orchestrator_re.sh via
#   sbatch --export=ALL,SWEEP_ENV_FILE=<path>).
# =============================================================================
#SBATCH --nodes=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:4
#SBATCH --mem=742520 
#SBATCH --exclusive
#SBATCH --time=3:00:00
#SBATCH --account=<your-slurm-account>

#SBATCH --output=/path/to/repo/src/serve_vllm/slurm_out/slurm_sweep_re/slurm_%x_%j.out
#SBATCH --error=/path/to/repo/src/serve_vllm/slurm_out/slurm_sweep_re/slurm_%x_%j.err

ulimit -n 65536

# =============================================================================
# ── Validate and source the per-run env file ──────────────────────────────────
# =============================================================================
if [[ -z "${SWEEP_ENV_FILE:-}" ]]; then
    echo "[FATAL] SWEEP_ENV_FILE is not set. Submit via sweep_orchestrator_re.sh."
    exit 1
fi
if [[ ! -f "${SWEEP_ENV_FILE}" ]]; then
    echo "[FATAL] SWEEP_ENV_FILE not found: ${SWEEP_ENV_FILE}"
    exit 1
fi

# Export all variables from the env file so child processes (srun steps) receive them
set -a
# shellcheck source=/dev/null
source "${SWEEP_ENV_FILE}"
set +a

# Append the actual SLURM job ID for traceability
echo "SLURM_JOB_ID_ACTUAL=${SLURM_JOB_ID}" >> "${SWEEP_ENV_FILE}"

# =============================================================================
# ── Paths ─────────────────────────────────────────────────────────────────────
# =============================================================================
REPO_ROOT="/path/to/repo"
BASE="${REPO_ROOT}/src"
HOST_BASE="${BASE}/serve_vllm"
VENV_PATH="${REPO_ROOT}/venv"
SIF_IMAGE="${HOST_BASE}/containers/vllm_serve_otel.sif"
HOST_MODEL="${BASE}/models/base/${SWEEP_MODEL}"
HOST_MON="${HOST_BASE}/monitoring"
TRITON_CACHE_DIR="${HOST_BASE}/triton_cache"
PYTHON_INCLUDE="$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')"
mkdir -p "${TRITON_CACHE_DIR}"

# =============================================================================
# ── Per-run artifacts directory ───────────────────────────────────────────────
# =============================================================================
VAL_SAFE="${SWEEP_PARAM_VALUE//[^a-zA-Z0-9._-]/_}"
ARTIFACTS_DIR="${SWEEP_ARTIFACTS_BASE}/${SWEEP_PARAM_CHANGED}_${VAL_SAFE}_${SLURM_JOB_ID}"
mkdir -p "${ARTIFACTS_DIR}"

# =============================================================================
# ── Exports for inference task ────────────────────────────────────────────────
# =============================================================================
export HOST_BASE
export VENV_PATH
export MODEL_NAME="${SWEEP_MODEL}"
export HOST_MODEL
export EXPERIMENT_RESULT_BASE="${HOST_BASE}/experiment_sweep_results/RE"
export ARTIFACTS_DIR
export METRICS_SCRIPT="${HOST_BASE}/collect_gpu_nvml.py"
export METRICS_INTERVAL_MS=100
export SLURM_JOB_ID
export JOB_ID="${SLURM_JOB_ID}"

export NODE_NAME="${SLURMD_NODENAME}"
export CHAT_URL="http://${NODE_NAME}:8000/v1/chat/completions"
export COMPLETIONS_URL="http://${NODE_NAME}:8000/v1/completions"
export ENERGY_URL="http://${NODE_NAME}:9400/metrics"
export VLLM_METRICS_URL="http://${NODE_NAME}:8000/metrics"

# Shared NVML metrics CSV for the full job (covers all prompt styles)
export METRICS_CSV="${ARTIFACTS_DIR}/gpu_metrics_${SLURM_JOB_ID}.csv"

# Override MLflow experiment name for this sweep
export MLFLOW_EXPERIMENT_NAME="${SWEEP_MLFLOW_EXPERIMENT}"

# RE-specific: DocRED split
export SWEEP_SPLIT="${SWEEP_SPLIT:-validation}"

# Load .env (MLflow credentials, ENERGY_URL etc.)
if [[ -f "${HOST_BASE}/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${HOST_BASE}/.env"
    set +a
fi

# =============================================================================
# ── Parse format_mode → file_type + constrained ──────────────────────────────────────
# =============================================================================
SWEEP_FILE_TYPE="${SWEEP_FORMAT_MODE:-json__false}"
SWEEP_FILE_TYPE="${SWEEP_FILE_TYPE%%__*}"         # e.g. "json"
SWEEP_CONSTRAINED="${SWEEP_FORMAT_MODE:-json__false}"
SWEEP_CONSTRAINED="${SWEEP_CONSTRAINED##*__}"     # e.g. "json_xgrammar" or "false"
export SWEEP_FILE_TYPE
export SWEEP_CONSTRAINED

# Inference task: always RE standalone (format dispatch is inside the script)
INFERENCE_TASK="${HOST_BASE}/inference_sweep_re_standalone.sh"
export INFERENCE_TASK

# =============================================================================
# ── Build vLLM optional flags ──────────────────────────────────────────────────
# =============================================================================
VLLM_OPTIONAL_FLAGS=""
[[ "${SWEEP_ENABLE_PREFIX_CACHING}" == "true" ]]  && VLLM_OPTIONAL_FLAGS+=" --enable-prefix-caching"
[[ "${SWEEP_ENABLE_CHUNKED_PREFILL}" == "true" ]]  && VLLM_OPTIONAL_FLAGS+=" --enable-chunked-prefill"
[[ "${SWEEP_CALCULATE_KV_SCALES}" == "true" ]]     && VLLM_OPTIONAL_FLAGS+=" --calculate-kv-scales"

# For constrained decoding (xgrammar), set guided-decoding-backend
case "${SWEEP_CONSTRAINED}" in
    json_xgrammar)
        VLLM_OPTIONAL_FLAGS+=" --guided-decoding-backend xgrammar" ;;
esac

# =============================================================================
# ── Summary ───────────────────────────────────────────────────────────────────
# =============================================================================
echo "======================================================================"
echo "  RE Sweep Run Summary"
echo "  Job ID          : ${SLURM_JOB_ID}"
echo "  Node            : $(hostname)"
echo "  Model           : ${SWEEP_MODEL}"
echo "  Task            : RE (DocRED)"
echo "  Split           : ${SWEEP_SPLIT}"
echo "  Param changed   : ${SWEEP_PARAM_CHANGED} = ${SWEEP_PARAM_VALUE}"
echo "  MLflow exp      : ${SWEEP_MLFLOW_EXPERIMENT}"
echo "  Artifacts       : ${ARTIFACTS_DIR}"
echo "  Inference task  : ${INFERENCE_TASK}"
echo "  vLLM flags      : --max-model-len ${SWEEP_MAX_MODEL_LEN}"
echo "                    --max-num-seqs ${SWEEP_MAX_NUM_SEQS}"
echo "                    --max-num-batched-tokens ${SWEEP_MAX_NUM_BATCHED_TOKENS}"
echo "                    --kv-cache-dtype ${SWEEP_KV_CACHE_DTYPE}"
echo "                    --block-size ${SWEEP_BLOCK_SIZE}"
echo "                    --dtype ${SWEEP_DTYPE}${VLLM_OPTIONAL_FLAGS}"
echo "======================================================================"

mkdir -p "${HOST_MON}"/{prometheus,jaeger,grafana/{data,logs,provisioning,dashboards}}
mkdir -p "${HOST_BASE}/slurm_out/slurm_sweep_re"

# Write a per-job server config JSON for reproducibility
SERVER_CONFIG_FILE="${ARTIFACTS_DIR}/server_config_${SLURM_JOB_ID}.json"
cat > "${SERVER_CONFIG_FILE}" <<EOF_JSON
{
  "job_id": "${SLURM_JOB_ID}",
  "node": "${SLURMD_NODENAME}",
  "node_name": "${SLURMD_NODENAME}",
  "number_of_allocated_gpus": "${SLURM_GPUS_ON_NODE:-1}",
  "started_at": "$(date -u +'%Y-%m-%d %H:%M:%S')",
  "sweep_env_file": "${SWEEP_ENV_FILE}",
  "param_changed": "${SWEEP_PARAM_CHANGED}",
  "param_value": "${SWEEP_PARAM_VALUE}",
  "mlflow_experiment": "${SWEEP_MLFLOW_EXPERIMENT}",
  "model": "${SWEEP_MODEL}",
  "model_name": "${SWEEP_MODEL}",
  "model_path": "${HOST_MODEL}",
  "task": "re",
  "split": "${SWEEP_SPLIT}",
  "inference_task": "${INFERENCE_TASK}",
  "max_model_len": ${SWEEP_MAX_MODEL_LEN},
  "max_num_seqs": ${SWEEP_MAX_NUM_SEQS},
  "max_num_batched_tokens": ${SWEEP_MAX_NUM_BATCHED_TOKENS},
  "kv_cache_dtype": "${SWEEP_KV_CACHE_DTYPE}",
  "block_size": ${SWEEP_BLOCK_SIZE},
  "dtype": "${SWEEP_DTYPE}",
  "enable_prefix_caching": ${SWEEP_ENABLE_PREFIX_CACHING},
  "enable_chunked_prefill": ${SWEEP_ENABLE_CHUNKED_PREFILL},
  "calculate_kv_scales": ${SWEEP_CALCULATE_KV_SCALES},
  "tensor_parallel_size": ${SWEEP_TENSOR_PARALLEL_SIZE},
  "gpu_memory_utilization": ${SWEEP_GPU_MEMORY_UTIL},
  "gpu_memory_util": ${SWEEP_GPU_MEMORY_UTIL},
  "metrics_interval_ms": "${METRICS_INTERVAL_MS}",
  "batch_size": ${SWEEP_BATCH_SIZE},
  "max_new_tokens": ${SWEEP_MAX_NEW_TOKENS},
  "max_concurrency": ${SWEEP_MAX_CONCURRENCY},
  "prompt_styles": "${SWEEP_PROMPT_STYLES}"
}
EOF_JSON
echo "[INFO] Server config: ${SERVER_CONFIG_FILE}"

# Write a copy in experiment_configs/ so eval scripts can discover it
EXPERIMENT_CONFIGS_DIR="${HOST_BASE}/experiment_configs"
mkdir -p "${EXPERIMENT_CONFIGS_DIR}"
SERVCONF_PATH="${EXPERIMENT_CONFIGS_DIR}/sweep_${SWEEP_PARAM_CHANGED}_job_${SLURM_JOB_ID}_m_${SWEEP_MODEL}_re_out_servconf.json"
cp "${SERVER_CONFIG_FILE}" "${SERVCONF_PATH}"
echo "[INFO] Servconf copy: ${SERVCONF_PATH}"

# =============================================================================
# ── Start monitoring stack ────────────────────────────────────────────────────
# =============================================================================
srun --overlap -n1 --cpus-per-task=6 --cpu-bind=none bash <<MONEOF &
  set -x
  singularity exec --nv \
    docker://nvidia/dcgm-exporter:3.1.7-3.1.4-ubuntu20.04 \
    /usr/bin/dcgm-exporter --address=:9400 --collect-interval=100 \
    > "${HOST_MON}/dcgm_${SLURM_JOB_ID}.log" 2>&1 &

  singularity exec \
    -B "${HOST_MON}":/monitoring:rw \
    docker://prom/prometheus:v3.4.0 \
    prometheus --config.file=/monitoring/prometheus.yml \
    > "${HOST_MON}/prometheus_${SLURM_JOB_ID}.log" 2>&1 &

  export SINGULARITYENV_GF_SECURITY_ADMIN_PASSWORD="admin"
  singularity exec \
    -B "${HOST_MON}/grafana/data":/var/lib/grafana:rw \
    -B "${HOST_MON}/grafana/logs":/var/log/grafana:rw \
    -B "${HOST_MON}/grafana/provisioning":/etc/grafana/provisioning:ro \
    -B "${HOST_MON}/grafana/dashboards":/var/lib/grafana/dashboards:ro \
    docker://grafana/grafana:latest \
    /run.sh \
    > "${HOST_MON}/grafana_${SLURM_JOB_ID}.log" 2>&1

  wait
MONEOF
MON_PID=$!

sleep 5

# =============================================================================
# ── Start vLLM server ─────────────────────────────────────────────────────────
# =============================================================================
# shellcheck disable=SC2086
srun --overlap -n1 --cpus-per-task=30 --cpu-bind=none --gres=gpu:${SWEEP_TENSOR_PARALLEL_SIZE} bash <<VLLMEOF &
  set -x
  export TORCH_COMPILE_DEBUG=1
  export TORCH_DYNAMO_DISABLE=1
  export TORCHDYNAMO_DISABLE=1
  export TRITON_CACHE_DIR=/triton_cache

  singularity exec --nv \
    -B "${HOST_MODEL}":/${SWEEP_MODEL}:ro \
    -B "${HOST_MON}":/monitoring \
    -B "${HOST_BASE}/templates":/templates:ro \
    -B "${TRITON_CACHE_DIR}":/triton_cache \
    -B "${PYTHON_INCLUDE}":/usr/include/python3.10:ro \
    "${SIF_IMAGE}" \
    vllm serve "/${SWEEP_MODEL}" \
      --host 0.0.0.0 --port 8000 \
      --tensor-parallel-size ${SWEEP_TENSOR_PARALLEL_SIZE} \
      --dtype ${SWEEP_DTYPE} \
      --kv-cache-dtype ${SWEEP_KV_CACHE_DTYPE} \
      --block-size ${SWEEP_BLOCK_SIZE} \
      --gpu-memory-utilization ${SWEEP_GPU_MEMORY_UTIL} \
      --max-model-len ${SWEEP_MAX_MODEL_LEN} \
      --max-num-seqs ${SWEEP_MAX_NUM_SEQS} \
      --max-num-batched-tokens ${SWEEP_MAX_NUM_BATCHED_TOKENS} \
      ${VLLM_OPTIONAL_FLAGS} \
    > "${HOST_MON}/vllm_${SLURM_JOB_ID}.log" 2>&1
VLLMEOF
VLLM_PID=$!

# =============================================================================
# ── Cleanup trap ──────────────────────────────────────────────────────────────
# =============================================================================
_cleanup() {
    echo "[INFO] Cleanup: stopping monitoring (PID ${MON_PID}) and vLLM (PID ${VLLM_PID})"
    kill "${MON_PID}" 2>/dev/null || true
    kill "${VLLM_PID}" 2>/dev/null || true
    sleep 2
    wait "${MON_PID}" "${VLLM_PID}" 2>/dev/null || true
}
trap _cleanup EXIT

# =============================================================================
# ── Wait for vLLM ready ───────────────────────────────────────────────────────
# =============================================================================
echo "[INFO] Waiting for vLLM server..."
VLLM_STARTUP_TIMEOUT=1200   # 20 minutes max for server startup
VLLM_STARTUP_ELAPSED=0
until curl -s "http://0.0.0.0:8000/health" > /dev/null 2>&1; do
    sleep 5
    VLLM_STARTUP_ELAPSED=$((VLLM_STARTUP_ELAPSED + 5))
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[ERROR] vLLM process (PID ${VLLM_PID}) exited before becoming healthy." \
             "See ${HOST_MON}/vllm_${SLURM_JOB_ID}.log for details."
        exit 1
    fi
    if [[ ${VLLM_STARTUP_ELAPSED} -ge ${VLLM_STARTUP_TIMEOUT} ]]; then
        echo "[ERROR] vLLM server did not become healthy within ${VLLM_STARTUP_TIMEOUT}s. Aborting."
        exit 1
    fi
    echo "[INFO] Still waiting... (${VLLM_STARTUP_ELAPSED}s elapsed)"
done
echo "[INFO] Server ready."

# =============================================================================
# ── Check DCGM exporter health ────────────────────────────────────────────────
# =============================================================================
DCGM_OK=false
for _attempt in 1 2 3; do
    if curl -s --max-time 5 "http://0.0.0.0:9400/metrics" | grep -q "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION"; then
        DCGM_OK=true
        break
    fi
    echo "[WARN] DCGM check attempt ${_attempt}/3 failed, retrying in 10s..."
    sleep 10
done

if [[ "${DCGM_OK}" == "false" ]]; then
    echo "[ERROR] DCGM exporter not reachable on ${SLURMD_NODENAME}:9400 — energy_corrected will be unavailable."
    echo "[INFO] Resubmitting job excluding node ${SLURMD_NODENAME}..."

    CURRENT_EXCLUDES=$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | grep -oP 'ExcNodeList=\K\S+' | sed 's/(null)//')
    if [[ -n "${CURRENT_EXCLUDES}" ]]; then
        NEW_EXCLUDES="${CURRENT_EXCLUDES},${SLURMD_NODENAME}"
    else
        NEW_EXCLUDES="${SLURMD_NODENAME}"
    fi

    sbatch \
        --exclude="${NEW_EXCLUDES}" \
        --export=ALL,SWEEP_ENV_FILE="${SWEEP_ENV_FILE}" \
        -o "${HOST_BASE}/slurm_out/slurm_sweep_re/slurm_sw_re_retry_${SLURM_JOB_ID}_%j.out" \
        -e "${HOST_BASE}/slurm_out/slurm_sweep_re/slurm_sw_re_retry_${SLURM_JOB_ID}_%j.err" \
        "${BASH_SOURCE[0]}"
    echo "[INFO] Resubmitted. Exiting current job."
    exit 0
fi
echo "[INFO] DCGM exporter healthy on ${SLURMD_NODENAME}:9400."

# =============================================================================
# ── DCGM mid-job watchdog ─────────────────────────────────────────────────────
# =============================================================================
_dcgm_watchdog() {
    while true; do
        sleep 60
        if ! curl -s --max-time 5 "http://0.0.0.0:9400/metrics" \
                | grep -q "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION"; then
            echo "[ERROR] DCGM watchdog: exporter died on ${SLURMD_NODENAME}:9400 mid-job. Resubmitting..."
            CURRENT_EXCLUDES=$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
                | grep -oP 'ExcNodeList=\K\S+' | sed 's/(null)//')
            if [[ -n "${CURRENT_EXCLUDES}" ]]; then
                NEW_EXCLUDES="${CURRENT_EXCLUDES},${SLURMD_NODENAME}"
            else
                NEW_EXCLUDES="${SLURMD_NODENAME}"
            fi
            sbatch \
                --exclude="${NEW_EXCLUDES}" \
                --export=ALL,SWEEP_ENV_FILE="${SWEEP_ENV_FILE}" \
                -o "${HOST_BASE}/slurm_out/slurm_sweep_re/slurm_sw_re_retry_${SLURM_JOB_ID}_%j.out" \
                -e "${HOST_BASE}/slurm_out/slurm_sweep_re/slurm_sw_re_retry_${SLURM_JOB_ID}_%j.err" \
                "${BASH_SOURCE[0]}"
            kill 0
            exit 0
        fi
    done
}
_dcgm_watchdog &
DCGM_WATCHDOG_PID=$!

# =============================================================================
# ── Run inference task ────────────────────────────────────────────────────────
# =============================================================================
srun --overlap -n1 --cpus-per-task=16 --cpu-bind=none \
    -o "${ARTIFACTS_DIR}/inference_${SLURM_JOB_ID}.out" \
    -e "${ARTIFACTS_DIR}/inference_${SLURM_JOB_ID}.err" \
    bash "${INFERENCE_TASK}"

INFERENCE_EXIT_CODE=$?
echo "[INFO] Inference done: exit code ${INFERENCE_EXIT_CODE}"

# =============================================================================
# ── Retry on CPU-binding failure (exit code 193) ──────────────────────────────
# =============================================================================
if [[ ${INFERENCE_EXIT_CODE} -eq 193 ]]; then
    echo "[ERROR] srun CPU-bind conflict (exit 193) on ${SLURMD_NODENAME} — resubmitting excluding this node."
    CURRENT_EXCLUDES=$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | grep -oP 'ExcNodeList=\K\S+' | sed 's/(null)//')
    if [[ -n "${CURRENT_EXCLUDES}" ]]; then
        NEW_EXCLUDES="${CURRENT_EXCLUDES},${SLURMD_NODENAME}"
    else
        NEW_EXCLUDES="${SLURMD_NODENAME}"
    fi
    sbatch \
        --exclude="${NEW_EXCLUDES}" \
        --export=ALL,SWEEP_ENV_FILE="${SWEEP_ENV_FILE}" \
        -o "${HOST_BASE}/slurm_out/slurm_sweep_re/slurm_sw_re_retry_${SLURM_JOB_ID}_%j.out" \
        -e "${HOST_BASE}/slurm_out/slurm_sweep_re/slurm_sw_re_retry_${SLURM_JOB_ID}_%j.err" \
        "${BASH_SOURCE[0]}"
    echo "[INFO] Resubmitted. Exiting current job."
    exit 0
fi

exit ${INFERENCE_EXIT_CODE}
