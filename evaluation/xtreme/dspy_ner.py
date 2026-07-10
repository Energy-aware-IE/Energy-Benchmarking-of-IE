"""
DSPy-based NER components for XTREME evaluation.

Provides Signature, Module, LM wrapper, and metric for NER entity extraction.
Compatible with DSPy GEPA optimizer for prompt optimization.

Usage:
    from dspy_ner import setup_dspy, create_ner_program, ner_prediction_metric

    # Set up DSPy with vLLM endpoint
    setup_dspy("/gemma-3-4b-it")

    # Create program (or load GEPA-optimised version)
    program = create_ner_program()
    # program = create_ner_program("path/to/optimized.json")

    # Run prediction
    result = program(sentence="John Smith visited Berlin.")
    # result.per -> ["John Smith"], result.org -> [], result.loc -> ["Berlin"]
"""

import dspy
import requests
from utils import CHAT_URL, COMPLETIONS_URL

# =====================================================================
# Language Model: vLLM endpoint wrapper for DSPy
# =====================================================================


class VLLMLanguageModel(dspy.LM):
    """DSPy LM wrapper for vLLM OpenAI-compatible endpoints.

    Routes all requests through the chat endpoint so instruction-tuned models
    receive proper chat-template formatting (critical for GEPA reflection).

    Args:
        model_name: Model identifier served by vLLM.
        chat_url: Chat completions endpoint URL.
        completions_url: Legacy completions endpoint URL (unused, kept for compat).
        max_tokens: Default max generation tokens.
    """

    def __init__(self, model_name: str, chat_url: str = None,
                 completions_url: str = None, max_tokens: int = 150):
        super().__init__(model=model_name)
        self.model_name = model_name
        self.chat_url = chat_url or CHAT_URL
        self.completions_url = completions_url or COMPLETIONS_URL
        self.default_max_tokens = max_tokens

    def __call__(self, prompt=None, messages=None, **kwargs):
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)

        if messages is not None:
            url = self.chat_url
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": max_tokens,
            }
        else:
            # Wrap plain-text prompt as a chat message so instruction-tuned
            # models get proper chat-template formatting (avoids echo/repeat
            # behaviour seen with the raw completions endpoint).
            url = self.chat_url
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt or ""}],
                "temperature": 0.0,
                "max_tokens": max_tokens,
            }

        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if "choices" in data and data["choices"]:
            choice = data["choices"][0]
            if "message" in choice and isinstance(choice["message"], dict):
                text = choice["message"].get("content", "")
            else:
                text = choice.get("text", "")
        else:
            text = ""

        return [text]


# =====================================================================
# Signature: structured NER input/output schema
# =====================================================================


class NERExtractionSignature(dspy.Signature):
    """Given a sentence, predict all named entities and classify each
    as PER (person), ORG (organisation), or LOC (location).
    Return the entity mentions exactly as they appear in the sentence.
    Always return all three categories, even if empty."""

    sentence = dspy.InputField(desc="The input sentence to analyse for named entities")
    per = dspy.OutputField(
        desc="List of person-name mentions found in the sentence (empty list if none)"
    )
    org = dspy.OutputField(
        desc="List of organisation-name mentions found in the sentence (empty list if none)"
    )
    loc = dspy.OutputField(
        desc="List of location-name mentions found in the sentence (empty list if none)"
    )


# =====================================================================
# Module: NER prediction program (optimisable by GEPA)
# =====================================================================


class NERPredictor(dspy.Module):
    """DSPy module for NER entity extraction.

    Wraps dspy.Predict (or dspy.ChainOfThought) with NERExtractionSignature.
    GEPA optimises the instructions inside the signature for better predictions.

    Args:
        use_chain_of_thought: If True, use ChainOfThought (adds rationale step).
            Slower but may improve accuracy. Default: False (use Predict).
    """

    def __init__(self, use_chain_of_thought=False):
        super().__init__()
        if use_chain_of_thought:
            self.predict = dspy.ChainOfThought(NERExtractionSignature)
        else:
            self.predict = dspy.Predict(NERExtractionSignature)

    def forward(self, sentence):
        def _to_list(x):
            """Normalise LM output to a clean Python list of strings."""
            if x is None:
                return []
            if isinstance(x, list):
                return [str(v).strip() for v in x if str(v).strip()]
            if isinstance(x, str):
                s = x.strip()
                return [s] if s else []
            return []

        try:
            out = self.predict(sentence=sentence)
            return dspy.Prediction(
                per=_to_list(getattr(out, "per", [])),
                org=_to_list(getattr(out, "org", [])),
                loc=_to_list(getattr(out, "loc", [])),
            )
        except Exception:
            # Parse failure → score as "predicted nothing" so GEPA can still evaluate
            return dspy.Prediction(per=[], org=[], loc=[])


# =====================================================================
# Metric: P/R/F1 scoring for evaluation and GEPA optimisation
# =====================================================================


def _set_prf1(pred_set, gold_set):
    """Compute set-level precision, recall, F1 for one entity type."""
    pred_s = set(pred_set)
    gold_s = set(gold_set)
    if not pred_s and not gold_s:
        return 1.0, 1.0, 1.0
    tp = len(pred_s & gold_s)
    precision = tp / len(pred_s) if pred_s else 0.0
    recall = tp / len(gold_s) if gold_s else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def ner_prediction_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    """Score NER prediction using macro-averaged P/R/F1.

    Works in two modes (required by GEPA):
      - Evaluation  (pred_name is None): returns float score
      - Optimisation (pred_name is set): returns Prediction(score, feedback)
    """
    p_per, r_per, f1_per = _set_prf1(pred.per, example.gold_per)
    p_org, r_org, f1_org = _set_prf1(pred.org, example.gold_org)
    p_loc, r_loc, f1_loc = _set_prf1(pred.loc, example.gold_loc)

    macro_f1 = (f1_per + f1_org + f1_loc) / 3.0
    macro_p = (p_per + p_org + p_loc) / 3.0
    macro_r = (r_per + r_org + r_loc) / 3.0

    if pred_name is None:
        return macro_f1

    # Optimisation mode: build detailed feedback for GEPA reflection LM
    missed_per = sorted(set(example.gold_per) - set(pred.per))
    extra_per = sorted(set(pred.per) - set(example.gold_per))
    missed_org = sorted(set(example.gold_org) - set(pred.org))
    extra_org = sorted(set(pred.org) - set(example.gold_org))
    missed_loc = sorted(set(example.gold_loc) - set(pred.loc))
    extra_loc = sorted(set(pred.loc) - set(example.gold_loc))

    parts = [
        f"Macro-avg F1={macro_f1:.2f}, P={macro_p:.2f}, R={macro_r:.2f}.",
        f"PER: P={p_per:.2f} R={r_per:.2f} F1={f1_per:.2f} "
        f"(missed={missed_per}, spurious={extra_per})",
        f"ORG: P={p_org:.2f} R={r_org:.2f} F1={f1_org:.2f} "
        f"(missed={missed_org}, spurious={extra_org})",
        f"LOC: P={p_loc:.2f} R={r_loc:.2f} F1={f1_loc:.2f} "
        f"(missed={missed_loc}, spurious={extra_loc})",
    ]

    has_missed = missed_per or missed_org or missed_loc
    has_extra = extra_per or extra_org or extra_loc

    if has_missed and not has_extra:
        parts.append("LOW RECALL: Improve instructions to predict ALL entity mentions.")
    elif has_extra and not has_missed:
        parts.append("LOW PRECISION: Improve instructions to avoid hallucinating entities.")
    elif has_missed and has_extra:
        parts.append("BOTH PRECISION AND RECALL issues: Be more accurate.")
    else:
        parts.append("Perfect prediction.")

    return dspy.Prediction(score=macro_f1, feedback=" | ".join(parts))


# =====================================================================
# Setup helpers
# =====================================================================


def setup_dspy(model_name, chat_url=None, completions_url=None):
    """Configure DSPy globally with a vLLM language model.

    Returns the LM instance (also usable as GEPA's reflection_lm).
    """
    lm = VLLMLanguageModel(model_name, chat_url, completions_url)
    dspy.configure(lm=lm)
    return lm


def create_ner_program(program_path=None, use_chain_of_thought=False):
    """Create NER program, optionally loading GEPA-optimised state from disk.

    Args:
        program_path: Path to a saved DSPy program JSON (from program.save()).
        use_chain_of_thought: Use ChainOfThought instead of Predict.

    Returns:
        NERPredictor instance ready for inference or further optimisation.
    """
    program = NERPredictor(use_chain_of_thought=use_chain_of_thought)
    if program_path:
        program.load(program_path)
    return program
