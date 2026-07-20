"""
Compact example: phoneme-aware ASR correction with a seq2seq model.

Expected labeled record shape:

{
    "whisper_text": "the patient takes low sartin daily",
    "corrected_text": "the patient takes losartan daily",
    "domain_terms": ["losartan", "lisinopril", "loratadine"]
}

Install:
    pip install torch transformers sentencepiece
    pip install g2p-en  # optional, used for real English grapheme-to-phoneme
    pip install panphon # optional, used for feature-aware phoneme costs

This script shows:
    1. one dummy inference
    2. one supervised backprop step
"""

from __future__ import annotations
from g2p_en import G2p
import gc
import panphon.distance
import time
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

print(torch.cuda.memory_allocated() / 1024**2, "MB allocated")
print(torch.cuda.memory_reserved() / 1024**2, "MB reserved")

MODEL_NAME = "google/flan-t5-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device is {DEVICE}")
MAX_CANDIDATES = 5
MIN_CANDIDATE_SCORE = 0.55
INSERT_DELETE_COST = 1.0

_panphon_distance = panphon.distance.Distance()

_g2p = G2p()

def text_to_phonemes(text: str) -> str:
    phones = _g2p(text)
    return " ".join(piece for piece in phones if piece.strip())



def normalize_phone(phone: str) -> str:
    """Remove ARPAbet stress digits, e.g. AH0 -> AH."""
    return "".join(char for char in phone.upper() if not char.isdigit())


def phoneme_substitution_cost(left: str, right: str) -> float:
    """
    Continuous substitution cost in [0, 1].

    If PanPhon is installed and the phones are IPA-like, use its feature edit
    distance. For g2p-en ARPAbet phones, use a coarse feature-overlap fallback.
    """
    left = normalize_phone(left)
    right = normalize_phone(right)

    if left == right:
        return 0.0

    if _panphon_distance is not None:
        return min(1.0, float(_panphon_distance.feature_edit_distance(left, right)))


def phoneme_alignment_cost(left: list[str], right: list[str]) -> float:
    """Levenshtein-style alignment cost with continuous phoneme substitutions."""
    previous = [j * INSERT_DELETE_COST for j in range(len(right) + 1)]
    for i, left_token in enumerate(left, start=1):
        current = [i * INSERT_DELETE_COST]
        for j, right_token in enumerate(right, start=1):
            insert = current[j - 1] + INSERT_DELETE_COST
            delete = previous[j] + INSERT_DELETE_COST
            replace = previous[j - 1] + phoneme_substitution_cost(left_token, right_token)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def best_phoneme_span_score(term_phones: list[str], transcript_phones: list[str]) -> float:
    """
    Score whether a term could appear somewhere in the transcript phoneme stream.

    1.0 means an exact phoneme subsequence match.
    Lower scores allow small insertions/deletions/substitutions.
    """
    if not term_phones or not transcript_phones:
        return 0.0

    best_score = 0.0
    target_len = len(term_phones)
    min_window = max(1, target_len - 2)
    max_window = min(len(transcript_phones), target_len + 2)

    for window_len in range(min_window, max_window + 1):
        for start in range(0, len(transcript_phones) - window_len + 1):
            window = transcript_phones[start : start + window_len]
            cost = phoneme_alignment_cost(term_phones, window)
            max_cost = max(target_len, window_len) * INSERT_DELETE_COST
            score = 1.0 - cost / max_cost
            best_score = max(best_score, score)

    return max(0.0, min(1.0, best_score))


def retrieve_phonetic_candidates(
    whisper_text: str,
    domain_terms: list[str],
    max_candidates: int = MAX_CANDIDATES,
    min_score: float = MIN_CANDIDATE_SCORE,
) -> list[tuple[str, float]]:
    """Return domain terms whose phonemes approximately match the transcript phones."""
    transcript_phones = text_to_phonemes(whisper_text).split()
    scored_terms = []

    for term in domain_terms:
        term_phones = text_to_phonemes(term).split()
        score = best_phoneme_span_score(term_phones, transcript_phones)
        if score >= min_score:
            scored_terms.append((term, score))

    return sorted(scored_terms, key=lambda item: item[1], reverse=True)[:max_candidates]


def format_input(record: dict[str, object]) -> str:
    """The model sees the transcript, its phonemes, and phonetic candidate terms."""
    whisper_text = str(record["whisper_text"])
    phones = text_to_phonemes(whisper_text)
    domain_terms = list(record.get("domain_terms", []))
    candidates = retrieve_phonetic_candidates(whisper_text, domain_terms)
    candidate_text = ", ".join(f"{term[0]}" for term in candidates)

    return (
        "Correct ASR errors. Return only the corrected transcript.\n"
        f"UNCORRECTED: {whisper_text}\n"
        f"POSSIBLE WORDS: {candidate_text}\n"
    )


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
    record = {
        "whisper_text": "the patient takes low sartin daily",
        "corrected_text": "the patient takes losartan daily",
        "domain_terms": ["losartan", "lisinopril", "loratadine"],
    }

    # One dummy inference.
    model.eval()
    input_text = format_input(record)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=4,
        )

    prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("INPUT:\n", input_text)
    print("\nPREDICTION:\n", prediction)

    # One supervised backprop step.
    # In real training, batch many records and use DataLoader + padding.
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    labels = tokenizer(
        text_target=record["corrected_text"],
        return_tensors="pt",
        truncation=True,
    ).input_ids.to(DEVICE)

    optimizer.zero_grad(set_to_none=True)
    output = model(**inputs, labels=labels)
    loss = output.loss
    backprop_start_time = time.time()
    loss.backward()
    optimizer.step()
    print(f"Backprop took {time.time() - backprop_start_time} seconds")

    print("\nONE-STEP TRAINING LOSS:", float(loss.detach().cpu()))


if __name__ == "__main__":
    main()
