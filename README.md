# Energy Benchmarking of LLM Inference for Information Extraction

This repository contains the experimental infrastructure used in the paper
*"Energy-Aware Benchmarking of Generative Information Extraction"* (anonymous
submission).

---

## Overview

We measure **GPU energy consumption** of LLM inference across four Information
Extraction (IE) tasks, four model sizes, and multiple output-format modes,
serving all models via **vLLM** on multi-GPU HPC nodes.

### Tasks

| Task | Dataset | Languages |
|---|---|---|
| Named Entity Recognition (NER) | XTREME | 10 typologically diverse languages |
| African NER | MasakhaNER 2.0 | 14 African languages |
| Event Extraction (EE) | RAMS | English |
| Relation Extraction (RE) | DocRED | English |



### Models

| Model | Parameters |
|---|---|
| Gemma-3-4B-it | ~4 B |
| Gemma-3-12B-it | ~12 B |
| Gemma-3-27B-it | ~27 B |
| LLaMA-3.3-70B-it | ~70 B |

### Output Format Modes

Each model × task combination is evaluated under four structured output formats:

| Mode | Description |
|---|---|
| `json__false` | JSON, unconstrained decoding |
| `json__json_xgrammar` | JSON, constrained decoding via XGrammar |
| `yaml__false` | YAML, unconstrained decoding |
| `dst__false` | DST (domain-specific template), unconstrained |

---

## Repository Layout

```
Energy-Benchmarking-of-IE/
├── sweep/                          # SLURM sweep orchestration
│   ├── sweep_orchestrator.sh           # NER/XTREME sweep driver
│   ├── sweep_orchestrator_ee.sh        # Event Extraction sweep driver
│   ├── sweep_orchestrator_re.sh        # Relation Extraction sweep driver
│   ├── sweep_orchestrator_masakha.sh   # MasakhaNER sweep driver
│   ├── sweep_single_run_slurm.sh       # SLURM job script (NER)
│   ├── sweep_single_run_slurm_ee.sh    # SLURM job script (EE)
│   ├── sweep_single_run_slurm_re.sh    # SLURM job script (RE)
│   ├── sweep_single_run_slurm_masakha.sh
│   └── run_http_transport_ablation.sh  # HTTP keep-alive/pooling ablation (reviewer response, Appendix A.8)
├── inference/                      # Per-task inference bash scripts
│   ├── inference_sweep_ner_standalone.sh
│   ├── inference_sweep_ee_standalone.sh
│   ├── inference_sweep_re_standalone.sh
│   └── inference_sweep_masakha_standalone.sh
├── prompts_in_all_languages/       # Prompt templates and few-shot examples
│   ├── *_system_prompt.json            # Per-language/system prompts
│   ├── ner_few_shots_*.json            # NER few-shot exemplars
│   ├── ee_*.json / re_*.json           # EE/RE prompt resources
│   └── gollie_prompts.py               # GoLLIE-style prompt helpers
├── evaluation/                     # Python evaluation scripts
│   ├── xtreme/
│   │   ├── lang_eval_ner_sweep.py      # NER evaluation + MLflow logging (HTTP_TRANSPORT_MODE/SOCKET_MONITOR instrumentation)
│   │   ├── dspy_ner.py                 # DSPy-based NER helper
│   │   ├── utils.py                    # Shared utilities (prompts, parsing, telemetry)
│   │   └── analyze_http_transport_ablation.py  # Statistical report for the HTTP transport ablation
│   ├── masakha/
│   │   └── lang_eval_masakha_sweep.py
│   ├── ee/
│   │   └── lang_eval_ee_sweep.py
│   └── re/
│       └── lang_eval_re_sweep.py
├── monitoring/                     # Energy & GPU monitoring
│   ├── collect_gpu_nvml.py             # NVML-based GPU metrics collector (10 Hz)
│   └── prometheus.yml                  # Prometheus scrape config
├── docker/                         # Container definitions
│   ├── inference/Dockerfile            # Training/fine-tuning container
│   └── vllm_serve_otel/Dockerfile      # vLLM serving container with OpenTelemetry
├── results/
│   ├── results.csv                     # Aggregated NER/EE/RE results, XTREME languages (2 893 rows)
│   ├── results_masakha.csv             # Aggregated NER results, MasakhaNER 2.0 languages (1 359 rows)
│   ├── results_serving_params.csv      # Serving-layer parameter ablation, XTREME (8 491 rows)
│   └── results_serving_params_masakha.csv  # Serving-layer parameter ablation, MasakhaNER (8 932 rows)
├── examples_of_outputs/            # Curated raw run artifacts backing specific paper claims
│   ├── README.md                       # Maps each subfolder to the claim it supports
│   ├── fsm_overhead_38x/               # FSM (Outlines) energy inflation up to ~38–44x
│   ├── scenario_d_pareto_optimal/      # xgrammar +74.0% F1 surge on EE
│   ├── scale_vs_efficiency/            # Gemma-3-4B vs 27B energy/F1 trade-off on NER
│   ├── batch_size_tuning_gain/         # HTTP batch_size ablation on RE
│   ├── llama70b_gemma27b_baselines/    # Baseline runs, Llama-3.3-70B & Gemma-3-27B
│   ├── mistral_large_baselines/        # Baseline runs, Mistral-Large-Instruct-2411 (reviewer response)
│   ├── http_transport_ablation/        # HTTP keep-alive/pooling vs. TIME_WAIT/recall/energy (reviewer response, Appendix A.8)
│   └── re_yaml_dst_vocab_fix/          # RE YAML/DST closed-set vocabulary fix: ~70% F1 gap narrows to ~0-9% (reviewer response)
├── templates/                      # Chat format Jinja2 templates
│   ├── chatml.jinja
│   ├── llama2.jinja
│   └── mistral.jinja
├── requirements.txt
└── README.md
```

---

## How It Works

```
sweep_orchestrator.sh
    └─► sbatch sweep_single_run_slurm.sh   (one job per config)
            ├─► srun vllm serve <model>     (GPU serving via Singularity)
            ├─► srun collect_gpu_nvml.py    (GPU energy sampling at 10 Hz)
            └─► srun inference_sweep_*.sh   (batch inference)
                    └─► evaluation/*.py     (F1 + energy → MLflow)
```

1. **`sweep_orchestrator.sh`** iterates over all combinations of model,
   language, prompt style, and format mode, writing a small `.env` file per
   run and submitting it as a SLURM job via `sbatch`.

2. **`sweep_single_run_slurm.sh`** (the SLURM job script) sources the `.env`
   file, starts `vllm serve` in a Singularity container, waits for it to
   become healthy, then launches the inference task and the GPU energy sampler
   as concurrent `srun` steps.

3. **`inference_sweep_*.sh`** drives the Python evaluation script for each
   requested prompt style sequentially, with a cooldown between runs.

4. **`evaluation/*.py`** sends batched async HTTP requests to the vLLM server,
   collects energy/throughput telemetry from Prometheus, evaluates task-specific
   metrics (seqeval F1 for NER, SRL F1 for EE, relation F1 for RE), and logs
   all results to MLflow.

5. **`collect_gpu_nvml.py`** polls the Prometheus `nvidia_gpu_*` metrics at
   configurable intervals (default 100 ms) and writes them to a CSV for
   offline energy analysis.

---

## Setup

### Prerequisites

- Multi-GPU compute node(s) (tested with 4× H100 / A100 40 GB per node)
- SLURM workload manager
- Singularity/Apptainer (for containerized vLLM serving)
- Python ≥ 3.10
- MLflow tracking server (`mlflow server` or a remote URI)
- Prometheus + DCGM exporter exposing GPU power metrics

### Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the `sweep/` directory (sourced automatically by the job
scripts):

```bash
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=energy_ner_sweep
ENERGY_URL=http://<node-hostname>:9400/metrics   # Prometheus DCGM endpoint
```

### Required external services

This repository expects the following external services/components to be
available before running sweeps:

- **SLURM**: schedules and executes all sweep jobs (`sbatch`, `srun`).
- **MLflow tracking server**: receives experiment runs, parameters, and metrics.
- **Prometheus + DCGM exporter**: exposes GPU telemetry and power/energy metrics
   used by the evaluation scripts.

At runtime, ensure these service endpoints are reachable from compute nodes
via environment variables (for example `MLFLOW_TRACKING_URI` and `ENERGY_URL`).

### Additional software for container workflow

If you build/run the provided Docker images, ensure the following software is
installed on the host environment:

- **Docker Engine** (or compatible OCI runtime) for building images from
   `docker/`.
- **NVIDIA GPU driver + CUDA-compatible runtime** on host nodes.
- **NVIDIA Container Toolkit** (for GPU access inside Docker containers).
- **Singularity/Apptainer** if your cluster executes containers as `.sif`
   images.

Optional (only if telemetry export is enabled in the serving container):

- **OpenTelemetry Collector or Jaeger OTLP endpoint** reachable from serving
   jobs.

### What is included in the provided Docker images

The repository ships two container definitions under `docker/`, each with a
different purpose:

- **`docker/vllm_serve_otel/Dockerfile`** (serving image):
   CUDA base image, Python 3.10 toolchain, `vllm`, `transformers`,
   `bitsandbytes`, and OpenTelemetry instrumentation packages for trace export.
- **`docker/inference/Dockerfile`** (training/inference utilities image):
   CUDA base image with Python virtual environment plus core ML/evaluation
   packages such as `torch`, `transformers`, `datasets`, `peft`, `trl`,
   `mlflow`, `seqeval`, `numpy`, `requests`, and related utilities.

In short, the Docker definitions provide the CUDA + Python runtime and most
ML/evaluation dependencies required by this workflow.

### Build the vLLM serving container

```bash
cd docker/vllm_serve_otel
docker build -t vllm_serve_otel .
singularity build vllm_serve_otel.sif docker-daemon://vllm_serve_otel:latest
# Place the .sif file at the path referenced in sweep/sweep_single_run_slurm.sh
```

---

## Running a Sweep

```bash
# 1. Edit sweep/sweep_orchestrator.sh:
#    - Set REPO_ROOT to the absolute path of this repository
#    - Set MODEL_LIST, LANGUAGE_LIST, FORMAT_MODES to your desired combinations
#    - Set your SLURM --account and --partition

# NER (XTREME) sweep
bash sweep/sweep_orchestrator.sh

# MasakhaNER sweep
bash sweep/sweep_orchestrator_masakha.sh

# Event Extraction sweep
bash sweep/sweep_orchestrator_ee.sh

# Relation Extraction sweep
bash sweep/sweep_orchestrator_re.sh
```

Each orchestrator submits one SLURM job per (model × language × format_mode)
combination. Results are logged to MLflow automatically.

---

## Results

The `results/` directory has four aggregated CSVs, each row already averaged
over N = 3 independent runs of that configuration:

| File | Task(s) | Languages | Rows | What it sweeps |
|---|---|---|---|---|
| `results.csv` | NER, EE, RE | XTREME (10) / `en` | 2 893 | model × language × output format × decoding × prompt style |
| `results_masakha.csv` | NER | MasakhaNER 2.0 (14) | 1 359 | model × language × output format × prompt style |
| `results_serving_params.csv` | NER, EE, RE | XTREME (10) / `en` | 8 491 | one-parameter-at-a-time serving-layer ablation (see below) |
| `results_serving_params_masakha.csv` | NER | MasakhaNER 2.0 (14) | 8 932 | one-parameter-at-a-time serving-layer ablation (see below) |

### `results.csv` / `results_masakha.csv`
Each row is one (task, model, language, format_mode, prompt_style)
configuration.

| Column | Description |
|---|---|
| `task` | `NER`, `EE`, or `RE` |
| `model` | `4B`, `12B`, `27B`, `70B` |
| `language` | ISO code (`results.csv` NER: 10 XTREME langs, EE/RE: `en`; `results_masakha.csv`: 14 MasakhaNER langs) |
| `format_mode` | `json__false`, `json__json_xgrammar`, `yaml__false`, `dst__false`, plus (in `results.csv`) `xml__false`, `xml__xml_xgrammar`, `dst__dst_ebnf`, `dst__dst_outlines` |
| `prompt_style` | Integer prompt index (P1–P9 for NER, P1–P8 for EE/RE) |
| `f1` | Mean task F1 score across 3 runs (`ner_f1` / `ee_arg_c_f1` / `re_f1`) |
| `energy_j` | Mean total GPU energy in Joules across 3 runs (idle-corrected) |
| `J_per_entity` | Joules per predicted entity (or argument / triple) |
| `J_per_TP` | Joules per true positive |
| `F1_per_J` | F1 per Joule (energy efficiency) |

**Coverage** (`results.csv`, rows by task):

| Task | Models | Languages | Formats | Rows | Notes |
|---|---|---|---|---|---|
| NER (XTREME) | 4 | 10 | up to 8 | 2 637 |  |
| Event Extraction | 4 | 1 | 4 | 128 |  |
| Relation Extraction | 4 | 1 | 4 | 128 |  |

`results_masakha.csv` covers NER only, across 4 output formats (`json__false`,
`yaml__false`, `dst__false`, `xml__false`) and 14 MasakhaNER languages; not
every model × language × format × prompt cell is filled (1 359 / a larger
theoretical maximum).

### `results_serving_params.csv` / `results_serving_params_masakha.csv`

Each row is one (task, model, language, `param_changed`, `param_value`,
prompt_style) configuration — the one-parameter-at-a-time (OPAAT) serving-layer
sweep backing Table 3/4 and Figure 4 of the paper. `param_changed` is one of:
`batch_size`, `max_concurrency`, `max_num_seqs`, `max_new_tokens`,
`max_model_len`, `max_num_batched_tokens`, `block_size`, `kv_cache_dtype`,
`enable_prefix_caching`, `enable_chunked_prefill`, `calculate_kv_scales`,
`tensor_parallel_size`. All other columns match the schema above.

---

## Example Raw Outputs

`examples_of_outputs/` contains a curated subset of raw per-run artifacts
(`mean_metrics_*.json`, `gpu_metrics.csv`, generated responses, telemetry)
that back specific headline numbers cited in the paper and in the rebuttal to
reviewers — as opposed to `results/`, which only has the final aggregated
metrics. See `examples_of_outputs/README.md` for the full mapping of each
subfolder to the paper claim it substantiates (FSM decoding overhead,
constrained-decoding F1 gains on EE, the 4B-vs-27B scale/efficiency
trade-off, batch-size tuning gains, and Llama-3.3-70B / Gemma-3-27B /
Mistral-Large-Instruct-2411 baseline comparisons, RE upd). All contents are
anonymized (usernames, node names, job IDs, paths, and timestamps stripped
or replaced with placeholders); no scientific data was altered.

---

## Monitoring

Start Prometheus using the repository config:

```bash
prometheus --config.file=monitoring/prometheus.yml
```

The default scrape target is the DCGM exporter on port 9400; adjust
`monitoring/prometheus.yml` to match your cluster. If you use Grafana in your
environment, point it at this Prometheus instance and import your dashboard of
choice.

---

## Citation

Anonymous submission. Citation will be added upon acceptance.

---

## License

Code is released under the MIT License.
Datasets used (XTREME, MasakhaNER 2.0, RAMS, DocRED) are subject to their own
respective licenses — please consult each dataset's repository for details.

