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
│   └── sweep_single_run_slurm_masakha.sh
├── inference/                      # Per-task inference bash scripts
│   ├── inference_sweep_ner_standalone.sh
│   ├── inference_sweep_ee_standalone.sh
│   ├── inference_sweep_re_standalone.sh
│   └── inference_sweep_masakha_standalone.sh
├── evaluation/                     # Python evaluation scripts
│   ├── xtreme/
│   │   ├── lang_eval_ner_sweep.py      # NER evaluation + MLflow logging
│   │   └── utils.py                    # Shared utilities (prompts, parsing, telemetry)
│   ├── masakha/
│   │   └── lang_eval_masakha_sweep.py
│   ├── ee/
│   │   └── lang_eval_ee_sweep.py
│   └── re/
│       └── lang_eval_re_sweep.py
├── monitoring/                     # Energy & GPU monitoring
│   ├── collect_gpu_nvml.py             # NVML-based GPU metrics collector (10 Hz)
│   ├── prometheus.yml                  # Prometheus scrape config
│   └── grafana/                        # Grafana dashboard provisioning
│       └── provisioning/
│           ├── dashboards/
│           └── datasources/
├── docker/                         # Container definitions
│   ├── inference/Dockerfile            # Training/fine-tuning container
│   └── vllm_serve_otel/Dockerfile      # vLLM serving container with OpenTelemetry
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

- Multi-GPU compute node(s) (tested with 4× H100 / A100 80 GB per node)
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

## Monitoring

Start Prometheus and Grafana with the provided configs:

```bash
prometheus --config.file=monitoring/prometheus.yml

# In a separate terminal:
grafana-server --homepath /usr/share/grafana
```

The Grafana dashboard provisions automatically from
`monitoring/grafana/provisioning/`. The default Prometheus scrape target is
the DCGM exporter on port 9400; adjust `monitoring/prometheus.yml` to match
your cluster.

---

## Citation

Anonymous submission. Citation will be added upon acceptance.

---

## License

Code is released under the MIT License.
Datasets used (XTREME, MasakhaNER 2.0, RAMS, DocRED) are subject to their own
respective licenses — please consult each dataset's repository for details.
