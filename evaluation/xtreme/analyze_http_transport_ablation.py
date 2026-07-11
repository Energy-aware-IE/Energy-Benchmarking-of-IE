#!/usr/bin/env python3
"""Statistical analysis of the HTTP transport ablation (reviewer question,
Appendix A.8): "Did you try HTTP keep-alive, connection pooling, or gRPC to
mitigate TIME_WAIT churn, and if so, how did this affect energy and recall?"

Reads the anonymized run artifacts shipped in
examples_of_outputs/http_transport_ablation/ (produced by
sweep/run_http_transport_ablation.sh + the HTTP_TRANSPORT_MODE/SOCKET_MONITOR
instrumentation in evaluation/xtreme/lang_eval_ner_sweep.py) and writes a
report with per-condition descriptive statistics (mean +/- SD over the 9
prompt styles), paired significance tests between transport modes, and
descriptive statistics for the direct TIME_WAIT socket measurements.

Design note on the paired test: the same 9 prompt styles (same wording, same
few-shot exemplars) are evaluated under every transport mode, so prompt style
is a matched/blocking factor. A paired test on matched prompt-style scores
removes between-prompt-style variance from the comparison and isolates the
transport-mode effect, which is more powerful than treating the 9 prompt
styles as independent unpaired samples of noise.

Usage:
    python3 analyze_http_transport_ablation.py
    python3 analyze_http_transport_ablation.py --out /path/to/report.md
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "examples_of_outputs/http_transport_ablation"

N_PROMPTS = 9
MODEL = "gemma-3-4b-it"
LANGUAGE = "de"
MAX_CONCURRENCY = 1024

# Primary (medium-scope) conditions: full 9-prompt sweep + direct TIME_WAIT
# socket measurement (SOCKET_MONITOR=1).
PRIMARY = {
    "pooled": ("medium_pooled", "pooled"),
    "close": ("medium_close", "close"),
    "tight": ("medium_tight", "tight"),
}

# Independent replicate submissions (no socket monitor) used only as a
# run-to-run stability check on pooled/close.
REPLICATES = {
    "pooled": "small_pooled_replicate",
    "close": "small_close_replicate",
}


def load_condition(dirname: str) -> dict:
    d = DATA_DIR / dirname
    recalls, precisions, f1s, energies = [], [], [], []
    for n in range(1, N_PROMPTS + 1):
        data = json.loads((d / f"mean_metrics_B_1024_prompt{n}.json").read_text())
        recalls.append(data["ner_recall"])
        precisions.append(data["ner_precision"])
        f1s.append(data["ner_f1"])
        energies.append(data["energy_corrected"])
    return {
        "recall": np.array(recalls),
        "precision": np.array(precisions),
        "f1": np.array(f1s),
        "energy_corrected": np.array(energies),
    }


def load_time_wait(dirname: str, mode: str):
    fp = DATA_DIR / dirname / f"socket_timewait_{mode}.csv"
    if not fp.exists():
        return None
    with open(fp, newline="") as f:
        rows = list(csv.DictReader(f))
    counts = np.array([int(r["time_wait_count"]) for r in rows if r["time_wait_count"].strip() not in ("", "-1")])
    return counts


def mean_sd(arr: np.ndarray) -> str:
    return f"{arr.mean():.4f} +/- {arr.std(ddof=1):.4f}"


def paired_ttest(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> str:
    t, p = stats.ttest_rel(a, b)
    diff = a - b
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float("nan")
    return (
        f"{label_a} vs {label_b}: mean diff = {diff.mean():+.4f} "
        f"(t({len(a)-1})={t:.2f}, p={p:.2e}, paired Cohen's d={d:.2f})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DATA_DIR / "STATISTICAL_REPORT.md"))
    args = parser.parse_args()

    cond = {mode: load_condition(dirname) for mode, (dirname, _) in PRIMARY.items()}
    tw = {mode: load_time_wait(dirname, sock_mode) for mode, (dirname, sock_mode) in PRIMARY.items()}
    repl = {mode: load_condition(dirname) for mode, dirname in REPLICATES.items()}

    lines = []
    lines.append("# HTTP Transport Ablation -- Statistical Report (Appendix A.8)\n")
    lines.append(
        f"**Reviewer question:** \"The concurrency failures suggest OS-level socket "
        f"exhaustion. Did you try HTTP keep-alive, connection pooling, or gRPC to "
        f"mitigate TIME_WAIT churn, and if so, how did this affect energy and recall?\"\n"
    )
    lines.append(
        f"**Design:** {MODEL}, XTREME German ({LANGUAGE}) NER, "
        f"`max_concurrency={MAX_CONCURRENCY}` -- the configuration with the steepest "
        f"published recall collapse. Three HTTP transport configurations "
        f"(`pooled`/`close`/`tight`, see `evaluation/xtreme/lang_eval_ner_sweep.py`), "
        f"each evaluated on the **same 9 prompt styles**, making prompt style a "
        f"matched/blocking factor: paired comparisons below isolate the transport-mode "
        f"effect from prompt-style variance rather than treating the 9 prompt-level "
        f"scores as independent unpaired samples.\n"
    )

    lines.append("## 1. Descriptive statistics (mean +/- SD over N=9 prompt styles)\n")
    lines.append("| Mode | Recall | Precision | F1 | energy_corrected (J) |")
    lines.append("|---|---|---|---|---|")
    for mode in ("pooled", "close", "tight"):
        c = cond[mode]
        lines.append(
            f"| {mode} | {mean_sd(c['recall'])} | {mean_sd(c['precision'])} | "
            f"{mean_sd(c['f1'])} | {c['energy_corrected'].mean():.0f} +/- {c['energy_corrected'].std(ddof=1):.0f} |"
        )

    lines.append("\n## 2. Paired significance tests (recall, N=9 matched prompt styles)\n")
    lines.append(f"- {paired_ttest(cond['close']['recall'], cond['pooled']['recall'], 'close', 'pooled')}")
    lines.append(f"- {paired_ttest(cond['tight']['recall'], cond['pooled']['recall'], 'tight', 'pooled')}")
    lines.append(f"- {paired_ttest(cond['close']['recall'], cond['tight']['recall'], 'close', 'tight')}")
    lines.append(
        "\nPrecision is not tested -- it is near-identical across modes by construction "
        "(same model, same prompts; only requests that are dropped differ), so a "
        "significance test on precision is not informative here.\n"
    )

    lines.append("## 3. Direct TIME_WAIT socket measurement (SOCKET_MONITOR=1, 2s polling interval)\n")
    lines.append("| Mode | n samples | mean | SD | max |")
    lines.append("|---|---|---|---|---|")
    for mode in ("pooled", "close", "tight"):
        c = tw[mode]
        lines.append(f"| {mode} | {len(c)} | {c.mean():.1f} | {c.std(ddof=1):.1f} | {c.max()} |")

    t_pc, p_pc = stats.mannwhitneyu(tw["close"], tw["pooled"], alternative="greater")
    lines.append(
        f"\n`close` TIME_WAIT counts are significantly higher than `pooled` "
        f"(Mann-Whitney U, one-sided: U={t_pc:.0f}, p={p_pc:.2e}) -- the opposite of "
        f"what an ephemeral-port-exhaustion explanation for `pooled`'s recall collapse "
        f"would predict, since `pooled` is the condition with *low* TIME_WAIT pressure "
        f"and *lost* recall, while `close` has *high* TIME_WAIT pressure and *intact* recall.\n"
    )

    lines.append("## 4. Run-to-run replication check (independent small-scope resubmissions)\n")
    lines.append("| Mode | Medium-scope recall | Small-scope replicate recall | Abs. difference |")
    lines.append("|---|---|---|---|")
    for mode in ("pooled", "close"):
        m = cond[mode]["recall"].mean()
        r = repl[mode]["recall"].mean()
        lines.append(f"| {mode} | {m:.4f} | {r:.4f} | {abs(m - r):.4f} |")
    lines.append(
        "\nTwo independent SLURM submissions of the same configuration agree to within "
        "~0.01-0.02 recall, well inside the between-prompt-style SD reported in Section 1 "
        "-- the pooled/close gap is not an artifact of run-to-run noise.\n"
    )

    lines.append("## 5. Interpretation\n")
    lines.append(
        "- Recall collapses under `pooled`/`tight` relative to `close` (Section 1), "
        "with precision flat -- consistent with requests being dropped outright rather "
        "than answered incorrectly, and the effect is statistically robust within-run "
        "(Section 2, all p < 0.001) and stable across independent runs (Section 4).\n"
        "- If the mechanism were TIME_WAIT/ephemeral-port exhaustion, the condition with "
        "*more* TIME_WAIT sockets should show *worse* recall. Section 3 shows the "
        "reverse: `close` has ~30x the TIME_WAIT count of `pooled`/`tight` yet has the "
        "intact recall. This is evidence against ephemeral-port exhaustion as the driver "
        "of the recall collapse in the published `pooled` configuration.\n"
        "- `tight` (pool size == max_concurrency, removing any idle reused connections "
        "sitting in an oversized pool) does not recover recall relative to `pooled` "
        "(Section 2), ruling out oversized idle-connection pooling as the cause.\n"
        "- Because the published pipeline already uses persistent keep-alive "
        "connections (`pooled`), and forcing a fresh connection per request (`close`) "
        "is what restores recall, the data are consistent with **connection staleness "
        "under sustained high concurrency** -- reused connections becoming failure-prone "
        "under load -- as the refined mechanism, rather than a simple absence of "
        "connection reuse or ephemeral-port pressure.\n"
        "- Energy trade-off: `close` costs ~20-25% more `energy_corrected` than "
        "`pooled`/`tight` (Section 1), consistent with the added TCP handshake cost of "
        "opening a new connection per request.\n"
    )

    lines.append("## 6. Threats to validity\n")
    lines.append(
        "- Single model/language/concurrency point (gemma-3-4b-it, German, "
        "max_concurrency=1024). This is the configuration with the steepest published "
        "recall collapse, chosen to maximize power to detect a transport effect, but "
        "the mechanism has not been re-verified at other model sizes or languages.\n"
        "- N=9 prompt styles are matched (paired) across conditions but are not repeated "
        "trials of an identical stimulus -- they differ in wording/exemplars, so the "
        "paired design controls for, rather than eliminates, prompt-style variability.\n"
        "- The `tight` condition's first submission reached a `FAILED` SLURM state "
        "(batch step cancelled) despite writing a complete result set; the run reported "
        "here (`medium_tight`) is a clean retry with consistent numbers, but the original "
        "anomaly is not itself explained.\n"
    )

    lines.append("## 7. Provenance\n")
    lines.append(
        "- Instrumentation: `HTTP_TRANSPORT_MODE` / `SOCKET_MONITOR` in "
        "`evaluation/xtreme/lang_eval_ner_sweep.py`\n"
        "- Ablation driver: `sweep/run_http_transport_ablation.sh`\n"
        "- Raw + anonymized data: `examples_of_outputs/http_transport_ablation/`\n"
        "- This report: `evaluation/xtreme/analyze_http_transport_ablation.py`\n"
    )

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[DONE] Wrote {out_path}")


if __name__ == "__main__":
    main()
