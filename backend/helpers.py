import re
import torch

MEL_CHUNK = 200  # 2 seconds ≈ 200 frames
def chunk_mel(mel, chunk_size=MEL_CHUNK):
    """
    mel: (1, 80, T)

    returns list of (1,80,chunk_size)
    """

    chunks = []

    T = mel.size(-1)

    for start in range(0, T, chunk_size):
        c = mel[..., start:start + chunk_size]

        if c.size(-1) < chunk_size:
            c = torch.nn.functional.pad(
                c,
                (0, chunk_size - c.size(-1))
            )

        chunks.append(c)

    return chunks, T


def collate(batch):

    waveforms = []
    texts = []

    for wav, txt in batch:
        waveforms.append(wav)
        texts.append(txt)

    return waveforms, texts


def wer(reference: str, hypothesis: str, do_normalize=True) -> float:

    if do_normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
    r, h = reference.split(), hypothesis.split()

    # dp[j] = distance between the first i reference words and first j hypothesis words
    dp = list(range(len(h) + 1))

    for i, rw in enumerate(r, 1):
        prev = dp[0]
        dp[0] = i

        for j, hw in enumerate(h, 1):
            old = dp[j]
            dp[j] = min(
                dp[j] + 1,                   # deletion
                dp[j - 1] + 1,               # insertion
                prev + (rw != hw)           # substitution
            )
            prev = old

    return dp[-1] / len(r) if r else float("inf")


def normalize_text(text):
    text = text.lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text