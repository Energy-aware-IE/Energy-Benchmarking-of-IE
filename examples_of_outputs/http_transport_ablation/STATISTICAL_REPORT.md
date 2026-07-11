# HTTP Transport Ablation -- Statistical Report (Appendix A.8)

**Reviewer question:** "The concurrency failures suggest OS-level socket exhaustion. Did you try HTTP keep-alive, connection pooling, or gRPC to mitigate TIME_WAIT churn, and if so, how did this affect energy and recall?"

**Design:** gemma-3-4b-it, XTREME German (de) NER, `max_concurrency=1024` -- the configuration with the steepest published recall collapse. Three HTTP transport configurations (`pooled`/`close`/`tight`, see `evaluation/xtreme/lang_eval_ner_sweep.py`), each evaluated on the **same 9 prompt styles**, making prompt style a matched/blocking factor: paired comparisons below isolate the transport-mode effect from prompt-style variance rather than treating the 9 prompt-level scores as independent unpaired samples.

## 1. Descriptive statistics (mean +/- SD over N=9 prompt styles)

| Mode | Recall | Precision | F1 | energy_corrected (J) |
|---|---|---|---|---|
| pooled | 0.4901 +/- 0.0790 | 0.6551 +/- 0.0446 | 0.5564 +/- 0.0506 | 19535 +/- 5657 |
| close | 0.6508 +/- 0.0460 | 0.6550 +/- 0.0447 | 0.6520 +/- 0.0389 | 23509 +/- 6112 |
| tight | 0.4784 +/- 0.0844 | 0.6551 +/- 0.0450 | 0.5480 +/- 0.0534 | 19286 +/- 5613 |

## 2. Paired significance tests (recall, N=9 matched prompt styles)

- close vs pooled: mean diff = +0.1607 (t(8)=5.57, p=5.29e-04, paired Cohen's d=1.86)
- tight vs pooled: mean diff = -0.0118 (t(8)=-1.25, p=2.46e-01, paired Cohen's d=-0.42)
- close vs tight: mean diff = +0.1724 (t(8)=5.53, p=5.52e-04, paired Cohen's d=1.84)

Precision is not tested -- it is near-identical across modes by construction (same model, same prompts; only requests that are dropped differ), so a significance test on precision is not informative here.

## 3. Direct TIME_WAIT socket measurement (SOCKET_MONITOR=1, 2s polling interval)

| Mode | n samples | mean | SD | max |
|---|---|---|---|---|
| pooled | 42 | 116.6 | 155.8 | 654 |
| close | 47 | 3848.8 | 2070.9 | 6539 |
| tight | 43 | 129.8 | 183.7 | 637 |

`close` TIME_WAIT counts are significantly higher than `pooled` (Mann-Whitney U, one-sided: U=1819, p=4.10e-12) -- the opposite of what an ephemeral-port-exhaustion explanation for `pooled`'s recall collapse would predict, since `pooled` is the condition with *low* TIME_WAIT pressure and *lost* recall, while `close` has *high* TIME_WAIT pressure and *intact* recall.

## 4. Run-to-run replication check (independent small-scope resubmissions)

| Mode | Medium-scope recall | Small-scope replicate recall | Abs. difference |
|---|---|---|---|
| pooled | 0.4901 | 0.4981 | 0.0080 |
| close | 0.6508 | 0.6505 | 0.0003 |

Two independent SLURM submissions of the same configuration agree to within ~0.01-0.02 recall, well inside the between-prompt-style SD reported in Section 1 -- the pooled/close gap is not an artifact of run-to-run noise.

## 5. Interpretation

- Recall collapses under `pooled`/`tight` relative to `close` (Section 1), with precision flat -- consistent with requests being dropped outright rather than answered incorrectly, and the effect is statistically robust within-run (Section 2, all p < 0.001) and stable across independent runs (Section 4).
- If the mechanism were TIME_WAIT/ephemeral-port exhaustion, the condition with *more* TIME_WAIT sockets should show *worse* recall. Section 3 shows the reverse: `close` has ~30x the TIME_WAIT count of `pooled`/`tight` yet has the intact recall. This is evidence against ephemeral-port exhaustion as the driver of the recall collapse in the published `pooled` configuration.
- `tight` (pool size == max_concurrency, removing any idle reused connections sitting in an oversized pool) does not recover recall relative to `pooled` (Section 2), ruling out oversized idle-connection pooling as the cause.
- Because the published pipeline already uses persistent keep-alive connections (`pooled`), and forcing a fresh connection per request (`close`) is what restores recall, the data are consistent with **connection staleness under sustained high concurrency** -- reused connections becoming failure-prone under load -- as the refined mechanism, rather than a simple absence of connection reuse or ephemeral-port pressure.
- Energy trade-off: `close` costs ~20-25% more `energy_corrected` than `pooled`/`tight` (Section 1), consistent with the added TCP handshake cost of opening a new connection per request.

## 6. Threats to validity

- Single model/language/concurrency point (gemma-3-4b-it, German, max_concurrency=1024). This is the configuration with the steepest published recall collapse, chosen to maximize power to detect a transport effect, but the mechanism has not been re-verified at other model sizes or languages.
- N=9 prompt styles are matched (paired) across conditions but are not repeated trials of an identical stimulus -- they differ in wording/exemplars, so the paired design controls for, rather than eliminates, prompt-style variability.
- The `tight` condition's first submission reached a `FAILED` SLURM state (batch step cancelled) despite writing a complete result set; the run reported here (`medium_tight`) is a clean retry with consistent numbers, but the original anomaly is not itself explained.

## 7. Provenance

- Instrumentation: `HTTP_TRANSPORT_MODE` / `SOCKET_MONITOR` in `evaluation/xtreme/lang_eval_ner_sweep.py`
- Ablation driver: `sweep/run_http_transport_ablation.sh`
- Raw + anonymized data: `examples_of_outputs/http_transport_ablation/`
- This report: `evaluation/xtreme/analyze_http_transport_ablation.py`

