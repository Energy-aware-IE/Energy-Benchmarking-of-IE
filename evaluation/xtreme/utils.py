"""
Utility functions for XTREME NER evaluation.

Contains: language mappings, prompt builders, response parsers, BIO tagging,
telemetry scraping, CSV analysis, and other helpers.
"""

import ast
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------
# Path setup & env
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
serve_vllm_dir = os.path.join(script_dir, "../../")
sys.path.insert(0, os.path.abspath(serve_vllm_dir))

load_dotenv()

CHAT_URL = os.getenv("CHAT_URL")
COMPLETIONS_URL = os.getenv("COMPLETIONS_URL")
ENERGY_URL = os.getenv("ENERGY_URL")
VLLM_METRICS_URL = os.getenv("VLLM_METRICS_URL")
ENERGY_METRIC_NAME = "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
FEW_SHOTS_PATH = os.path.join(
    serve_vllm_dir, "prompts_in_all_languages/ner_few_shots_full_panx.json"
)
metrics_csv_path = os.environ.get("METRICS_CSV")
job_id = os.getenv("JOB_ID")

# ---------- GoLLIE prompt headers ----------
from prompts_in_all_languages.gollie_prompts import ENGLISH_HEADER

LANG2HEADER = {
    "de": ENGLISH_HEADER,
    "en": ENGLISH_HEADER,
    "zh": ENGLISH_HEADER,
    "bg": ENGLISH_HEADER,
    "it": ENGLISH_HEADER,
    "es": ENGLISH_HEADER,
    "fr": ENGLISH_HEADER,
    "et": ENGLISH_HEADER,
}


# =====================================================================
# Language mappings
# =====================================================================


def resolve_xtreme_subset(language: str) -> str:
    mapping = {
        "ar": "PAN-X.ar",
        "bg": "PAN-X.bg",
        "de": "PAN-X.de",
        "en": "PAN-X.en",
        "pl": "PAN-X.pl",
        "es": "PAN-X.es",
        "fr": "PAN-X.fr",
        "el": "PAN-X.el",
        "hi": "PAN-X.hi",
        "id": "PAN-X.id",
        "it": "PAN-X.it",
        "ja": "PAN-X.ja",
        "ko": "PAN-X.ko",
        "nl": "PAN-X.nl",
        "pt": "PAN-X.pt",
        "ru": "PAN-X.ru",
        "th": "PAN-X.th",
        "tr": "PAN-X.tr",
        "ur": "PAN-X.ur",
        "vi": "PAN-X.vi",
        "zh": "PAN-X.zh",
        "yo": "PAN-X.yo",
    }
    if language not in mapping:
        raise ValueError(
            f"Unsupported language: {language}. Available: {', '.join(sorted(mapping.keys()))}"
        )
    return mapping[language]


def full_lang_name(language: str) -> str:
    return {
        "af": "Afrikaans",
        "de": "German",
        "en": "English",
        "nl": "Dutch",
        "es": "Spanish",
        "fr": "French",
        "it": "Italian",
        "pt": "Portuguese",
        "bg": "Bulgarian",
        "ru": "Russian",
        "bn": "Bengali",
        "hi": "Hindi",
        "mr": "Marathi",
        "ur": "Urdu",
        "fa": "Persian",
        "el": "Greek",
        "zh": "Chinese",
        "my": "Burmese",
        "ar": "Arabic",
        "he": "Hebrew",
        "ja": "Japanese",
        "ko": "Korean",
        "kk": "Kazakh",
        "tr": "Turkish",
        "et": "Estonian",
        "fi": "Finnish",
        "hu": "Hungarian",
        "id": "Indonesian",
        "jv": "Javanese",
        "ms": "Malay",
        "tl": "Tagalog",
        "ml": "Malayalam",
        "ta": "Tamil",
        "te": "Telugu",
        "sw": "Swahili",
        "yo": "Yoruba",
        "ka": "Georgian",
        "th": "Thai",
        "vi": "Vietnamese",
        "eu": "Basque",
    }.get(language, language)


def build_artifacts_dir(language: str, batch_size: int, model_name: str):
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_name)
    return f"xtreme_{language}_B{batch_size}_{safe_model}_{job_id}"


# =====================================================================
# Prompt building
# =====================================================================

SYSTEM_TEMPLATE = (
    "You are a Named Entity Recognition (NER) annotator for the {language} language.\n"
    "- Input: a single {language} sentence (do not translate).\n"
    "- Output: only a single, valid JSON object with only three arrays: `PER`, `ORG`, `LOC`.\n"
    "- Do not output any extra text, explanations, or markdown."
    "- Find entities of mentioned categories only (PER- person, ORG- organization, LOC- location).\n"
    "- JSON must be parseable by json.loads.\n"
    "- Extract only entities that appear verbatim in the input sentence (no hallucinations).\n"
    "- Keep entity surface forms exactly as they appear; do not normalize or translate."
)


def load_gemma_system_template(
    language: str, system_prompt_choice: int, use_generic_template=False
) -> str:
    if use_generic_template:
        template_path = os.path.join(
            serve_vllm_dir, "prompts_in_all_languages/Template_system_prompt.json"
        )
    else:
        template_path = os.path.join(
            serve_vllm_dir,
            f"prompts_in_all_languages/{full_lang_name(language)}_system_prompt.json",
        )

    with open(template_path, encoding="utf-8") as f:
        data = json.load(f)

    if str(system_prompt_choice) not in data:
        raise ValueError(f"No system prompt found for choice: {system_prompt_choice}")
    else:
        prompt = (
            data[str(system_prompt_choice)]
            if isinstance(data[str(system_prompt_choice)], str)
            else "\n".join(data[str(system_prompt_choice)])
        )

    if use_generic_template:
        prompt = prompt.replace("{language}", full_lang_name(language))

    return prompt


def build_system_msg(language: str) -> dict:
    return {"role": "system", "content": SYSTEM_TEMPLATE.format(language=full_lang_name(language))}


def load_few_shots(language: str):
    with open(FEW_SHOTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if language not in data:
        raise ValueError(f"No few-shot examples found for language: {language}")
    return data[language]


def make_messages_for(sentence: str, language: str, examples_for_lang: list):
    system_msg = build_system_msg(language)
    return [system_msg] + examples_for_lang + [{"role": "user", "content": f"Sentence: {sentence}"}]


def build_gollie_prompt(sentence: str, language: str, use_generic_template=False) -> str:
    if language in LANG2HEADER and not use_generic_template:
        header = LANG2HEADER[language]
    else:
        template_header = ENGLISH_HEADER
        header = template_header.replace("English language", f"{full_lang_name(language)} language")

    text_literal = json.dumps(sentence, ensure_ascii=False)
    return (
        header
        + f"\ntext = {text_literal}\n# Annotate entities in the {full_lang_name(language)} language.\nresult = [\n"
    )


def build_gemma_messages(
    sentence: str,
    language: str,
    system_prompt_choice: int,
    examples_for_lang: list,
    use_generic_template=False,
) -> list:
    sys_msg = load_gemma_system_template(
        language=language,
        system_prompt_choice=system_prompt_choice,
        use_generic_template=use_generic_template,
    )

    messages = [{"role": "system", "content": sys_msg}]

    pairs_for_choice = {
        1: 0,
        2: 1,
        3: 0,
        4: 1,
        5: 2,
        6: 3,
        7: 0,
        8: 0,
    }
    n_pairs = pairs_for_choice.get(system_prompt_choice, 0)

    if n_pairs > 0:
        for i in range(min(n_pairs * 2, len(examples_for_lang))):
            messages.append(examples_for_lang[i])

    messages.append({"role": "user", "content": sentence})
    return messages


def save_messages_to_file(messages_list, out_dir, model_name, language, batch_size):
    messages_dir = os.path.join(out_dir, "messages")
    os.makedirs(messages_dir, exist_ok=True)
    messages_path = os.path.join(
        messages_dir, f"{model_name.replace('/', '')}_messages_{language}_B{batch_size}.json"
    )
    with open(messages_path, "w", encoding="utf-8") as f:
        json.dump(messages_list, f, indent=2, ensure_ascii=False)
    return messages_path


def create_sample_prompt(
    model_name, language, system_prompt_choice=None, use_generic_template=False
):
    sample_sentence = "John Smith from Microsoft visited Berlin last week."
    few_shots = load_few_shots(language)

    if (
        model_name in ["/gemma-3-4b-it", "/gemma-3-12b-it", "/gemma-3-27b-it", "/mistral", "/llama-3-3-70B-it"]
        and system_prompt_choice is not None
    ):
        messages = build_gemma_messages(
            sample_sentence,
            language,
            system_prompt_choice,
            few_shots,
            use_generic_template=use_generic_template,
        )
        return {
            "prompt_type": "gemma_chat",
            "messages": messages,
            "system_prompt_choice": system_prompt_choice,
            "use_generic_template": use_generic_template,
        }
    else:
        prompt = build_gollie_prompt(
            sample_sentence, language, use_generic_template=use_generic_template
        )
        return {
            "prompt_type": "gollie_completion",
            "prompt": prompt,
            "use_generic_template": use_generic_template,
        }


# =====================================================================
# Response parsing
# =====================================================================


def parse_response_json_like(response_text: str) -> dict:
    cleaned = re.sub(r"^\s*\d+\s*=\s*", "", response_text, flags=re.MULTILINE)
    cleaned = cleaned.strip().strip("'\"")

    start = cleaned.find("{")
    if start < 0:
        return {}
    depth = 0
    end_idx = None
    for i, ch in enumerate(cleaned[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i
                break

    if end_idx is not None:
        block = cleaned[start : end_idx + 1]
    else:
        block = cleaned[start:] + "}" * depth

    block = re.sub(r",\s*}", "}", block)
    block = block.replace("'", '"')

    parsed = {}
    for loader in (json.loads, lambda s: ast.literal_eval(s)):
        try:
            parsed = loader(block)
            if isinstance(parsed, dict):
                break
        except Exception:
            continue

    result = {}
    if isinstance(parsed, dict):
        for k, v in parsed.items():
            key = k.strip().upper()
            if isinstance(v, str):
                items = [v.strip()] if v.strip() else []
            elif isinstance(v, (list, tuple)):
                items = [str(x).strip() for x in v if str(x).strip()]
            else:
                items = []
            result[key] = items
    return result


CLASS_PATTERNS = {
    "PER": re.compile(r'PER\s*\(\s*mention\s*=\s*(?P<q>["\'])(?P<val>.+?)\1\s*\)'),
    "ORG": re.compile(r'ORG\s*\(\s*mention\s*=\s*(?P<q>["\'])(?P<val>.+?)\1\s*\)'),
    "LOC": re.compile(r'LOC\s*\(\s*mention\s*=\s*(?P<q>["\'])(?P<val>.+?)\1\s*\)'),
}


def parse_gollie_output_to_entities(text: str) -> dict:
    if "]" in text:
        text = text.split("]", 1)[0]
    out = {"PER": [], "ORG": [], "LOC": []}
    for k, pat in CLASS_PATTERNS.items():
        vals = [m.group("val").strip() for m in pat.finditer(text)]
        seen = set()
        dedup = []
        for v in vals:
            if v not in seen:
                seen.add(v)
                dedup.append(v)
        out[k] = dedup
    return out


def robust_gemma_entity_extraction(text: str) -> dict:
    if "```json" in text:
        json_match = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()
        else:
            text = re.sub(r"```json\s*\n", "", text, 1)

    result = parse_response_json_like(text)
    if any(result.get(key, []) for key in ["PER", "ORG", "LOC"]):
        return result

    out = {"PER": [], "ORG": [], "LOC": []}
    for key in ["PER", "ORG", "LOC"]:
        pattern = rf'"{key}"\s*:\s*\[([^\]]*)'
        m = re.search(pattern, text)
        if m:
            items = re.findall(r'"([^"]+)"', m.group(1))
            out[key] = items
    return out


# =====================================================================
# BIO tagging
# =====================================================================


def get_bio_tags_language_aware(sentence, entities, language="en"):
    if language in ["zh", "ar"]:
        chars = list(sentence.replace(" ", ""))
        tags = ["O"] * len(chars)
        for ent_type, mentions in entities.items():
            for mention in mentions:
                mention_clean = mention.replace(" ", "")
                mention_chars = list(mention_clean)
                for i in range(len(chars) - len(mention_chars) + 1):
                    if chars[i : i + len(mention_chars)] == mention_chars:
                        if all(tags[i + j] == "O" for j in range(len(mention_chars))):
                            tags[i] = "B-" + ent_type
                            for j in range(1, len(mention_chars)):
                                tags[i + j] = "I-" + ent_type
                        break
        return chars, tags
    else:
        tokens = sentence.split()
        tags = ["O"] * len(tokens)
        for ent_type, mentions in entities.items():
            for mention in mentions:
                mention_tokens = mention.split()
                for i in range(len(tokens) - len(mention_tokens) + 1):
                    if tokens[i : i + len(mention_tokens)] == mention_tokens:
                        if all(tags[i + j] == "O" for j in range(len(mention_tokens))):
                            tags[i] = "B-" + ent_type
                            for j in range(1, len(mention_tokens)):
                                tags[i + j] = "I-" + ent_type
                        break
        return tokens, tags


def get_bio_tags(sentence, entities):
    return get_bio_tags_language_aware(sentence, entities, "en")


# =====================================================================
# Serialization helper
# =====================================================================


def numpy_serializer(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {obj.__class__.__name__} not serializable")


# =====================================================================
# Telemetry: energy & vLLM metrics scraping
# =====================================================================


def read_energy_joules() -> float:
    """Read DCGM energy metric; returns joules (DCGM exposes millijoules)."""
    try:
        r = requests.get(ENERGY_URL, timeout=1.0).text
        for line in r.splitlines():
            if line.startswith(ENERGY_METRIC_NAME):
                return float(line.split()[-1]) / 1000.0
        raise RuntimeError(f"{ENERGY_METRIC_NAME} not found in /metrics")
    except Exception:
        return 0.0


def read_vllm_metrics() -> dict:
    _keys = [
        "prompt_tokens_total",
        "generation_tokens_total",
        "e2e_latency_sum",
        "e2e_latency_count",
        "time_to_first_sum",
        "time_to_first_count",
        "time_per_token_sum",
        "time_per_token_count",
        "request_prefill_time_sum",
        "request_prefill_time_count",
        "request_inference_time_sum",
        "request_inference_time_count",
        "request_decode_time_sum",
        "request_decode_time_count",
    ]
    try:
        text = requests.get(VLLM_METRICS_URL).text
        m = {k: None for k in _keys}

        _name_map = {
            "vllm:prompt_tokens_total": "prompt_tokens_total",
            "vllm:generation_tokens_total": "generation_tokens_total",
            "vllm:e2e_request_latency_seconds_count": "e2e_latency_count",
            "vllm:time_to_first_token_seconds_sum": "time_to_first_sum",
            "vllm:time_to_first_token_seconds_count": "time_to_first_count",
            "vllm:time_per_output_token_seconds_sum": "time_per_token_sum",
            "vllm:time_per_output_token_seconds_count": "time_per_token_count",
            "vllm:request_prefill_time_seconds_sum": "request_prefill_time_sum",
            "vllm:request_prefill_time_seconds_count": "request_prefill_time_count",
            "vllm:request_inference_time_seconds_sum": "request_inference_time_sum",
            "vllm:request_inference_time_seconds_count": "request_inference_time_count",
            "vllm:request_decode_time_seconds_sum": "request_decode_time_sum",
            "vllm:request_decode_time_seconds_count": "request_decode_time_count",
        }

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            raw_name, raw_val = parts[0], parts[1]
            name = raw_name.split("{", 1)[0]
            try:
                val = float(raw_val)
            except ValueError:
                continue

            # e2e_latency_sum uses startswith (handles bucket labels)
            if name.startswith("vllm:e2e_request_latency_seconds_sum"):
                m["e2e_latency_sum"] = val
            elif name in _name_map:
                m[_name_map[name]] = val

        for k, v in m.items():
            if v is None:
                m[k] = 0.0
        return m
    except Exception:
        return {k: 0.0 for k in _keys}


def analyze_metrics_csv(t0_epoch, t1_epoch, prefill_wall_s, prompt_t, gen_t):
    """Analyze metrics CSV file using wall-clock epoch timestamps."""
    csv_path = os.environ.get("METRICS_CSV")
    if not csv_path or not os.path.exists(csv_path):
        print(f"Metrics CSV file not found: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)
        df["datetime"] = pd.to_datetime(
            df["timestamp"], format="%Y-%m-%dT%H:%M:%S.%f", errors="coerce"
        ).dt.tz_localize("Europe/Berlin", ambiguous="infer", nonexistent="shift_forward")

        print("CSV range:", df["datetime"].min(), "→", df["datetime"].max())

        t0_dt = pd.to_datetime(t0_epoch, unit="s", utc=True).tz_convert("Europe/Berlin")
        t1_dt = pd.to_datetime(t1_epoch, unit="s", utc=True).tz_convert("Europe/Berlin")
        print("t1_epoch:", t1_dt)
        print("t0_epoch:", t0_dt)
        print("prefill_wall_s:", prefill_wall_s)

        prefill_end_epoch = t0_epoch + max(prefill_wall_s, 0.0)
        prefill_end_dt = pd.to_datetime(prefill_end_epoch, unit="s", utc=True).tz_convert(
            "Europe/Berlin"
        )
        print("prefill_end_epoch", prefill_end_dt)

        print(len(df), "rows in CSV")
        t0_idx = (df["datetime"] - t0_dt).abs().idxmin()
        print("t0 idx ", t0_idx)
        pref_nearest_idx = (df["datetime"] - prefill_end_dt).abs().idxmin()
        print("nearest prefill end idx ", pref_nearest_idx)
        t1_idx = (df["datetime"] - t1_dt).abs().idxmin()
        print("t1 idx ", t1_idx)

        t0_energy = float(df.at[t0_idx, "energy_consumption"])
        print("energy t0", t0_energy)
        prefill_end_energy = float(df.at[pref_nearest_idx, "energy_consumption"])
        print("energy prefill end", prefill_end_energy)
        t1_energy = float(df.at[t1_idx, "energy_consumption"])
        print("energy t1", t1_energy)

        prefill_energy_joules = (prefill_end_energy - t0_energy) / 1000.0
        generation_energy_joules = (t1_energy - prefill_end_energy) / 1000.0
        prefill_energy_per_token = prefill_energy_joules / max(prompt_t, 1)
        generation_energy_per_token = generation_energy_joules / max(gen_t, 1)

        result = {
            "csv_prefill_energy_j": prefill_energy_joules,
            "csv_generation_energy_j": generation_energy_joules,
            "csv_prefill_energy_per_token_j": prefill_energy_per_token,
            "csv_generation_energy_per_token_j": generation_energy_per_token,
            "csv_total_energy_j": prefill_energy_joules + generation_energy_joules,
        }

        if "gpu_util" in df.columns or "power_usage" in df.columns:
            mask_prefill = (df["datetime"] >= t0_dt) & (df["datetime"] <= prefill_end_dt)
            mask_gen = (df["datetime"] >= prefill_end_dt) & (df["datetime"] <= t1_dt)
            prefill_df = df.loc[mask_prefill]
            gen_df = df.loc[mask_gen]

            if "gpu_util" in df.columns:
                mean_val = prefill_df["gpu_util"].mean()
                if pd.notna(mean_val):
                    result["csv_prefill_avg_gpu_util"] = float(mean_val)
                result["csv_generation_avg_gpu_util"] = float(gen_df["gpu_util"].mean())
                result["gpu_util_avg_overall"] = float(
                    df.loc[(df["datetime"] >= t0_dt) & (df["datetime"] <= t1_dt)]["gpu_util"].mean()
                )
            if "power_usage" in df.columns:
                mean_val_pw = prefill_df["power_usage"].mean()
                if pd.notna(mean_val_pw):
                    result["csv_prefill_avg_power_w"] = float(mean_val_pw)
                result["csv_generation_avg_power_w"] = float(gen_df["power_usage"].mean())
                result["power_draw_avg_overall"] = float(
                    df.loc[(df["datetime"] >= t0_dt) & (df["datetime"] <= t1_dt)][
                        "power_usage"
                    ].mean()
                )

        return result
    except Exception as e:
        print(f"Error analyzing metrics CSV: {e}")
        return None


def compute_energy_corrected(
    csv_path: str,
    util_col: str = "gpu_util",
    energy_col: str = "energy_consumption",
    timestamp_col: str = "timestamp",
    util_threshold: float = 0.0,
    energy_in_mJ: bool = True,
    start_timestamp: str = None,
) -> dict:
    """
    Compute energy between the FIRST and LAST non-zero GPU-util samples.
    Optionally filters rows to those >= start_timestamp.
    """
    if not csv_path or not os.path.isfile(csv_path):
        return {"ok": False, "reason": f"CSV missing: {csv_path}"}

    try:
        df = pd.read_csv(
            csv_path,
            parse_dates=[timestamp_col],
            dtype={util_col: "float64", energy_col: "float64"},
            on_bad_lines="skip",
            engine="python",
        )
    except Exception as e:
        return {"ok": False, "reason": f"Failed to read CSV: {e}"}

    if timestamp_col not in df or util_col not in df or energy_col not in df:
        return {
            "ok": False,
            "reason": f"CSV lacks required columns: {timestamp_col}, {util_col}, {energy_col}",
        }

    df = (
        df.dropna(subset=[timestamp_col, util_col, energy_col])
        .sort_values(timestamp_col, kind="mergesort")
        .reset_index(drop=True)
    )

    if start_timestamp:
        try:
            start_dt = pd.to_datetime(start_timestamp)
            df = df[df[timestamp_col] >= start_dt].reset_index(drop=True)
        except Exception as e:
            return {"ok": False, "reason": f"Invalid start_timestamp format: {e}"}

    if df.empty:
        return {"ok": False, "reason": "No valid rows after filtering."}

    util = df[util_col].to_numpy()
    active_idx = np.nonzero(util > util_threshold)[0]

    if active_idx.size == 0:
        return {"ok": False, "reason": "GPU util never > 0."}

    start_idx = int(active_idx[0])
    end_idx = int(active_idx[-1])

    e0 = float(df.at[start_idx, energy_col])
    e1 = float(df.at[end_idx, energy_col])

    if energy_in_mJ:
        e0 /= 1000.0
        e1 /= 1000.0

    energy_corrected = max(e1 - e0, 0.0)

    t_start = pd.to_datetime(df.at[start_idx, timestamp_col])
    t_end = pd.to_datetime(df.at[end_idx, timestamp_col])
    latency_s = max((t_end - t_start).total_seconds(), 0.0)

    return {
        "ok": True,
        "energy_corrected": energy_corrected,
        "latency_corrected": latency_s,
        "t_start": t_start.isoformat(),
        "t_end": t_end.isoformat(),
        "start_util": float(df.at[start_idx, util_col]),
        "end_util": float(df.at[end_idx, util_col]),
        "n_rows": int(len(df)),
        "n_active": int(active_idx.size),
    }
