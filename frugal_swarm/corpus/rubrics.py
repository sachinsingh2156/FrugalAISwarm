"""
Scoring rubrics per task family.

Original families:
  - multistep_qa:        exact-match / keyword-overlap
  - document_analysis:   F1 on key extracted terms
  - workflow_planning:   keyword coverage

Education-domain families (all high_stakes=false):
  - formative_assessment_drafting:  keyword F1 on learning objectives coverage
  - curriculum_question_generation: keyword F1 on question content alignment
  - lesson_adaptation:              keyword F1 on adaptation criteria coverage
  - knowledge_base_retrieval:       keyword F1 on factual content recall
"""
from __future__ import annotations

import string


_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "was", "are",
    "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may",
    "might", "must", "shall", "that", "this", "these", "those",
    "it", "its", "they", "their", "them", "we", "our", "you",
    "your", "he", "she", "his", "her", "which", "what", "who",
    "step", "steps", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "first", "second", "third", "fourth", "fifth", "sixth",
    "finally", "lastly", "then", "next", "also", "additionally",
    "ensure", "use", "make", "using", "through", "into", "out",
}


def _tokenize(text: str) -> set[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return set(text.split())


def _content_tokens(text: str) -> set[str]:
    """Tokenize and strip stopwords for rubrics sensitive to paraphrase."""
    return _tokenize(text) - _STOP_WORDS


def _content_f1(pred: str, ref: str) -> float:
    """F1 on content (stopword-filtered) tokens — handles paraphrase better."""
    pred_toks = _content_tokens(pred)
    ref_toks = _content_tokens(ref)
    if not ref_toks:
        return 1.0 if not pred_toks else 0.0
    tp = len(pred_toks & ref_toks)
    if tp == 0:
        return 0.0
    precision = tp / max(len(pred_toks), 1)
    recall = tp / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


def _keyword_f1(pred: str, ref: str) -> float:
    pred_toks = _tokenize(pred)
    ref_toks = _tokenize(ref)
    if not ref_toks:
        return 1.0 if not pred_toks else 0.0
    tp = len(pred_toks & ref_toks)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_toks)
    recall = tp / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


def _exact_match(pred: str, ref: str) -> float:
    return 1.0 if _tokenize(pred) == _tokenize(ref) else 0.0


def _contains_match(pred: str, ref: str) -> float:
    """Partial credit: fraction of ref tokens found in pred."""
    ref_tokens = _tokenize(ref)
    pred_tokens = _tokenize(pred)
    found = len(ref_tokens & pred_tokens) / max(len(ref_tokens), 1)
    return found


def score(task_family: str, prediction: str, reference: str) -> float:
    """
    Return a score in [0, 1] for a (prediction, reference) pair.

    multistep_qa:                    combined exact + keyword-overlap score
    document_analysis:               F1 on extracted key terms
    workflow_planning:               keyword coverage
    formative_assessment_drafting:   keyword F1 on learning objectives
    curriculum_question_generation:  keyword F1 on question alignment
    lesson_adaptation:               keyword F1 on adaptation criteria
    knowledge_base_retrieval:        keyword F1 on factual recall
    """
    if task_family == "multistep_qa":
        exact = _exact_match(prediction, reference)
        if exact == 1.0:
            return 1.0
        return min(1.0, _contains_match(prediction, reference))

    elif task_family == "document_analysis":
        # Models write full sentences; reference is bare terms.
        # Recall (contains_match) is the right signal: did the model include the key terms?
        return _contains_match(prediction, reference)

    elif task_family == "workflow_planning":
        # References use numbered steps; models write prose paragraphs.
        # Content F1 (stopword-filtered) avoids penalising paraphrase of step numbers.
        return _content_f1(prediction, reference)

    # ── Education families ─────────────────────────────────────────────────────
    elif task_family == "formative_assessment_drafting":
        # Score on coverage of key pedagogical terms / learning objectives
        return _keyword_f1(prediction, reference)

    elif task_family == "curriculum_question_generation":
        # Score on alignment between generated questions and reference criteria
        return _keyword_f1(prediction, reference)

    elif task_family == "lesson_adaptation":
        # Score on adaptation criteria coverage (level, modality, context)
        return _keyword_f1(prediction, reference)

    elif task_family == "knowledge_base_retrieval":
        # Score on factual content recall
        return _keyword_f1(prediction, reference)

    else:
        # Generic fallback
        return _keyword_f1(prediction, reference)


def is_success(task_family: str, prediction: str, reference: str, threshold: float = 0.4) -> bool:
    """Binary success judgement (for H3 metric computation)."""
    return score(task_family, prediction, reference) >= threshold


# ── Rubric metadata ────────────────────────────────────────────────────────────

RUBRIC_DESCRIPTIONS: dict[str, str] = {
    "multistep_qa":                   "Exact + keyword overlap (multi-step reasoning)",
    "document_analysis":              "Keyword F1 on extracted terms",
    "workflow_planning":              "Keyword F1 on plan step coverage",
    "formative_assessment_drafting":  "Keyword F1 on learning objectives (edu, non-automated)",
    "curriculum_question_generation": "Keyword F1 on question alignment (edu, non-automated)",
    "lesson_adaptation":              "Keyword F1 on adaptation criteria (edu, non-automated)",
    "knowledge_base_retrieval":       "Keyword F1 on factual recall (edu, non-automated)",
}

HIGH_STAKES_FAMILIES: set[str] = set()  # none — policy: no high-stakes automated scoring
