# RE YAML/DST Prompt-Vocabulary Fix — Before/After Comparison

## Background

The original RE (DocRED) system prompts had an asymmetry: the JSON prompt
(`prompts_in_all_languages/re_system_prompts.json`) always inlined the full
closed-set relation vocabulary (96 DocRED relation names, e.g. "member of
political party", "headquarters location", "owned by"), but the YAML and DST
system prompts (`RE_YAML_SYSTEM_TEMPLATE` / `RE_DST_SYSTEM_TEMPLATE` in
`evaluation_scripts/re/re_lang_eval_mlflow_mi_gol_all9_semaphore_other_formats.py`)
did not consistently ground the model in that same vocabulary. Under
exact-match relation-triple scoring, this penalised valid-but-differently-
worded relation labels for YAML/DST specifically — not a genuine format
effect, but an artifact of unequal prompt grounding across formats.

The templates were corrected to inject the same closed-set vocabulary
(`{valid_relations}` placeholder, filled via `build_valid_relations_str`) into
the YAML/DST system prompts, and all 4 models were rerun on RE for both
formats (`rerun_re_yaml_dst_vocab_fix.sh` → `resubmit_sweep_run.sh`).

## 1. Vocabulary-compliance rate (mechanism check)

Fraction of predicted relation triples whose relation string is one of the
96 valid DocRED relation names (case-insensitive exact match), computed
directly from each run's `generated_responses/*.json`:

| Model | Format | Original (buggy) | Corrected (this folder) | JSON baseline (unaffected) |
|---|---|---|---|---|
| Gemma-3-4B | YAML | 8.3% (n=88,768) | 53.3% (n=87,246) | 53.1% |
| Gemma-3-4B | DST | 10.4% (n=110,556) | 56.1% (n=114,350) | 53.1% |
| Gemma-3-12B | YAML | 14.3% (n=86,645) | 91.7% (n=85,202) | — |
| Gemma-3-12B | DST | 12.0% (n=107,375) | 84.0% (n=106,129) | — |
| Gemma-3-27B | YAML | 17.5% (n=68,923) | 84.0% (n=89,579) | — |
| Gemma-3-27B | DST | 19.2% (n=65,948) | 86.2% (n=114,958) | — |
| Llama-3.3-70B | YAML | 21.6% (n=94,022) | 86.4% (n=82,488) | — |
| Llama-3.3-70B | DST | 20.1% (n=136,077) | 83.4% (n=131,861) | — |

This is the direct mechanistic evidence for the bug and the fix: before the
fix, only 8-22% of predicted relation strings were even members of the valid
DocRED vocabulary (the model was free-associating relation names it was
never given); after the fix, compliance jumps to 53-92%, converging toward
the JSON baseline's ~53% for the one model (4B) where a JSON baseline is
also available in this repo (`format_sensitivity_re_697/`). Predicted
relations outside the closed set can never exact-match a gold triple, so
this directly explains the F1 gap in Section 2.

## 2. F1 impact

Mean `re_f1` across all 8 prompt styles (`mean_metrics_B_128_prompt*.json`):

| Model | JSON (unaffected) | YAML: original → corrected | DST: original → corrected | JSON vs. corrected YAML | JSON vs. corrected DST |
|---|---|---|---|---|---|
| Gemma-3-4B | 0.0666 | 0.0202 → **0.0723** | 0.0177 → **0.0640** | **−8.6%** (YAML wins) | +3.9% |
| Gemma-3-12B | 0.1468 | 0.0411 → **0.1447** | 0.0350 → **0.1449** | +1.4% | +1.3% |
| Gemma-3-27B | 0.1784 | 0.0491 → **0.1748** | 0.0407 → **0.1636** | +2.0% | +8.3% |
| Llama-3.3-70B | 0.1855 | 0.0527 → **0.1681** | 0.0536 → **0.1734** | +9.3% | +6.5% |

The original ~70%-relative-drop finding (still documented in
`format_sensitivity_re_697/`, which reflects the pre-fix runs — kept
unchanged as it accurately documents what was actually published) narrows to
roughly 0-9% once YAML/DST are given the same relation vocabulary as JSON.
JSON is best-or-tied-best at 3 of 4 scales (12B, 27B, 70B); at 4B, corrected
YAML actually **outperforms** JSON by 8.6%.

## Contents of this folder

Anonymized raw artifacts for the 8 corrected reruns (`mean_metrics_*.json`,
`re_metrics/`, `telemetry/`, `generated_responses/`, `gpu_metrics.csv`,
`server_config.json`), one subfolder per (model, format):

- `gemma4b_RE_yaml_fixed/`, `gemma4b_RE_dst_fixed/`
- `gemma12b_RE_yaml_fixed/`, `gemma12b_RE_dst_fixed/`
- `gemma27b_RE_yaml_fixed/`, `gemma27b_RE_dst_fixed/`
- `llama70b_RE_yaml_fixed/`, `llama70b_RE_dst_fixed/`

`gemma4b_RE_yaml_fixed/` and `gemma4b_RE_dst_fixed/` additionally include a
`sample_prompt_prompt1.json` showing the corrected system prompt in full
(same document as the existing `format_sensitivity_re_697/` JSON/YAML
example, for direct comparison): the closed-set vocabulary is now inlined
via `{valid_relations}` substitution exactly as in the JSON prompt.

JSON baselines and the original (pre-fix) YAML/DST runs are **not**
duplicated here — JSON is unaffected by this fix and its numbers are
unchanged from `results.csv` / `format_sensitivity_re_697/`; the original
YAML/DST numbers used for the "before" column above are likewise the
existing published numbers in `results.csv` (untouched by this update).

## Status

- `results.csv` / `results_serving_params.csv` in this repo have **not**
  been updated with these corrected numbers (out of scope for this folder;
  left untouched intentionally).
- The template fix and rerun driver
  (`rerun_re_yaml_dst_vocab_fix.sh`, `RE_YAML_SYSTEM_TEMPLATE` /
  `RE_DST_SYSTEM_TEMPLATE`) currently live only in the internal working
  repository, not yet mirrored into this repo's `evaluation/`/`sweep/`.
