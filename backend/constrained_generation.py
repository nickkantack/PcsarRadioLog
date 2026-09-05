from collections import Counter
from constants import VALIDATION_SPLIT, BATCH_SIZE
from dataset import AudioDataset
from helpers import collate
from loops import validate
from transformers import LogitsProcessor
import math
import pickle

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from constants import SAMPLE_RATE

# ---------------------------------------------------------------------------
# Build the vocabulary of favored phrases
# ---------------------------------------------------------------------------

def count_label_ngrams(train_loader, processor,
                       min_len=2, max_len=5):
    """
    Count all overlapping token sequences of lengths min_len..max_len.

    get_label_text(batch) -> list[str]
    """
    counts = Counter()

    for batch in train_loader:

        texts = batch[1]

        tokens = processor.tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
        )["input_ids"]

        for ids in tokens:
            for n in range(min_len, min(max_len, len(ids)) + 1):
                for i in range(len(ids) - n + 1):
                    counts[tuple(ids[i:i+n])] += 1

    return counts


def count_how_often_each_top_ngram_was_one_token_away_from_completing(train_loader, processor, prefix_to_top_ngram_map,
                       min_len=2, max_len=5):

    sans_last_token_counts = Counter()

    for batch in train_loader:

        texts = batch[1]

        tokens = processor.tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
        )["input_ids"]

        for sequence in tokens:
            for start_index in range(len(sequence) - min_len + 1):
                for end_index in range(start_index + min_len - 1, min(start_index + max_len, len(sequence) - min_len + 1)):
                    prefix = tuple(tokens[start_index:end_index])
                    print(prefix)
                    top_ngrams = prefix_to_top_ngram_map[prefix] if prefix in prefix_to_top_ngram_map else []
                    for ngram in top_ngrams:
                        sans_last_token_counts[ngram] += 1

    return sans_last_token_counts


def top_ngrams(counts, n):
    return counts.most_common(n)


# ---------------------------------------------------------------------------
# Trie
# ---------------------------------------------------------------------------

class TokenTrie:
    def __init__(self):
        self.children = {}       # token_id -> node
        self.bias = 0.0          # bias for arriving at this node

    def add(self, tokens, bias):
        node = self

        for token in tokens:
            if token not in node.children:
                node.children[token] = TokenTrie()
            node = node.children[token]

        node.bias = bias

    def find(self, tokens):
        """
        Return the trie node corresponding to tokens,
        or None if tokens aren't a prefix in the trie.
        """
        node = self

        for token in tokens:
            node = node.children.get(token)
            if node is None:
                return None

        return node


def build_trie(ngrams, biases):
    """
    ngrams: [(token_tuple, corpus_count), ...]
    biases: {token_tuple: bias}
    """
    trie = TokenTrie()

    for tokens, _ in ngrams:
        trie.add(tokens, biases[tokens])

    return trie


# ---------------------------------------------------------------------------
# LogitsProcessor
# ---------------------------------------------------------------------------


class TrieLogitBias(LogitsProcessor):
    def __init__(self, trie, max_len=5):
        self.trie = trie
        self.max_len = max_len

    def __call__(self, input_ids, scores):
        """
        input_ids: [batch, sequence_length]
        scores:    [batch, vocab_size]

        For each item, find the longest suffix of generated tokens
        that exists in the trie. Add the biases of all possible
        continuations from that node.
        """
        """
        print(scores)
        plt.hist(scores[0][torch.isfinite(scores[0])].detach().cpu().numpy(), bins=30)
        plt.show()
        raise Exception()
        """

        for b in range(input_ids.shape[0]):
            ids = input_ids[b].tolist()

            # Longest matching suffix is preferable.
            node = None

            for n in range(min(self.max_len - 1, len(ids)), 0, -1):
                candidate = self.trie.find(ids[-n:])

                if candidate is not None:
                    node = candidate
                    break

            if node is None:
                continue

            for token_id, child in node.children.items():
                scores[b, token_id] += child.bias

        return scores


def make_initial_biases(top_phrases, default_bias=1.0):
    return {
        tokens: default_bias
        for tokens, _ in top_phrases
    }


def calibrate_biases(top_phrases, model_counts, biases, learning_rate):

    for phrase, corpus_count in top_phrases:
        model_count = model_counts.get(phrase, 0)

        max_abs_shift = 0.2
        shift = learning_rate * math.log(corpus_count / max(model_count, 1))
        if abs(shift) > max_abs_shift:
            shift = max_abs_shift if shift > 0 else -max_abs_shift

        if phrase in biases:
            biases[phrase] += shift
        else:
            biases[phrase] = shift

    return biases


def count_model_ngrams(model, train_loader, processor, 
                       favored, device, logits_processors, max_len=5):
    """
    Run the unmodified model and count favored phrases in its outputs.

    favored is a set of token tuples.
    """
    counts = Counter()

    model.eval()

    batch_number = 0
    with torch.no_grad():
        for waveforms, texts in train_loader:
            for waveform, _ in zip(waveforms, texts):

                mel = processor.feature_extractor(
                    waveform.numpy(),
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                ).input_features.to(device)

                generated_ids = model.generate(
                    input_features=mel,
                    num_beams=5,
                    do_sample=False,
                    early_stopping=True,
                    logits_processor=logits_processors if logits_processors is not None else []
                )

                for ids in generated_ids.tolist():
                    for n in range(2, min(max_len, len(ids)) + 1):
                        for i in range(len(ids) - n + 1):
                            phrase = tuple(ids[i:i+n])

                            if phrase in favored:
                                counts[phrase] += 1
            batch_number += 1
            print(f"\rFinished with {batch_number}/{len(train_loader)} batches", end="")
        print()

    return counts


def count_model_grams(
    model,
    train_loader,
    processor,
    favored_ngrams,
    device,
    logits_processors=None,
    max_len=5,
):
    """
    Run the model once and count favored ngrams
    in its generated output.

    Returns:
        ngram_counts: Counter keyed by token tuples of length 2..max_len
    """
    ngram_counts = Counter()

    model.eval()

    special_ids = set(processor.tokenizer.all_special_ids)

    batch_number = 0

    with torch.no_grad():
        for waveforms, texts in train_loader:
            for waveform, _ in zip(waveforms, texts):

                mel = processor.feature_extractor(
                    waveform.numpy(),
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                ).input_features.to(device)

                generated_ids = model.generate(
                    input_features=mel,
                    num_beams=5,
                    do_sample=False,
                    early_stopping=True,
                    logits_processor=(
                        logits_processors
                        if logits_processors is not None
                        else []
                    ),
                )

                for ids in generated_ids.tolist():

                    # Match the label counting behavior:
                    # don't count special tokens.
                    ids = [
                        token_id
                        for token_id in ids
                        if token_id not in special_ids
                    ]

                    # 2-grams through max_len-grams
                    for n in range(2, min(max_len, len(ids)) + 1):
                        for i in range(len(ids) - n + 1):
                            phrase = tuple(ids[i:i+n])

                            if phrase in favored_ngrams:
                                ngram_counts[phrase] += 1

            batch_number += 1
            print(
                f"\rFinished with "
                f"{batch_number}/{len(train_loader)} batches",
                end=""
            )

    print()

    return ngram_counts


def run_constrained_generation_experiment(train_loader, val_loader, model, processor, whisper, device, writer=None):

    """
    print("Running baseline validation with no logits processor...")
    val_loss, average_wer = validate(model, val_loader, processor, whisper, device, 0, None)
    logits_processor = None
    biases = {}
    """

    print("Counting ngrams in labels...")
    counts = count_label_ngrams(
        train_loader,
        processor,
        min_len=2,
        max_len=5,
    )

    # Drop all ngrams that didn't occur a minimum number of times
    min_ngram_freq_to_keep = 10
    ngrams_to_del = []
    for ngram, count in counts.items():
        if count < min_ngram_freq_to_keep:
            ngrams_to_del.append(ngram)
    for ngram in ngrams_to_del:
        del counts[ngram]
    
    # Create a map of token sequence to all the top ngrams that need only one more token to complete
    prefix_to_top_ngram_map = {}
    for ngram in counts:
        prefix = ngram[:-1]
        if prefix not in prefix_to_top_ngram_map:
            prefix_to_top_ngram_map[prefix] = [ngram]
        else:
            prefix_to_top_ngram_map[prefix].append(ngram)

    sans_last_token_counts = count_how_often_each_top_ngram_was_one_token_away_from_completing(
        train_loader,
        processor,
        prefix_to_top_ngram_map,
        min_len=2,
        max_len=5
    )

    top_ngram_to_completion_probability = {}
    for ngram, count in counts:
        opportunity_count = sans_last_token_counts[ngram]
        top_ngram_to_completion_probability[ngram] = count / opportunity_count

    ngrams_in_completion_probability_order = top_ngram_to_completion_probability.keys()
    ngrams_in_completion_probability_order = sorted(ngrams_in_completion_probability_order,
                                                    key=lambda x: ngrams_in_completion_probability_order[x], reverse=True)

    ngram_to_textual_representation = {}
    for ngram in counts:
        text = processor.tokenizer.decode(
            list(ngram),
            skip_special_tokens=True,
        )
        ngram_to_textual_representation[ngram] = text
    
    for ngram in ngrams_in_completion_probability_order[:50]:
        print(f"'{ngram_to_textual_representation[ngram]}' appeared {counts[ngram]} times and completed {100 * top_ngram_to_completion_probability[ngram]:.0f}% of opportunities.")

    return

    # TODO iterate again over the labels and tokenize them, then keep one counter for how often each ngram in 
    # counts above sees all but its last token (and does not hit the end of the sequence), and how often the 
    # entire ngram sequence appears. This allows us to compute, for each ngram, what fraction of the time it
    # appears when the label is one token away from completing the ngram.

    # TODO filter out all ngrams that appeared fewer than some threshold number of times (e.g. 10), then filter
    # out all ngrams that were continued from their one-token-remaining state with probability below p, where
    # p is at most 50% (most conservative) and trends downward towards 0% in line with our confidence that 
    # acoustic information made all model misses a matter of marginal logit value changes (most optimistic).

    print("Computing top phrases...")
    top_phrases = top_ngrams(counts, n=1000)

    print(f"Starting WER: {average_wer:.3f}")

    passes = 30
    for i in range(passes):

        """
        for tokens, count in top_phrases[:50]:
            text = processor.tokenizer.decode(
                list(tokens),
                skip_special_tokens=True,
            )
            biases = make_initial_biases(
                top_phrases,
                default_bias=1.0,
            )
            print(f"{count:5d}  bias={biases[tokens]:7.3f}  {text!r}")
        """

        print("Counting ngrams in model output...")
        ngram_model_counts = count_model_grams(
            whisper,
            train_loader,
            processor,
            top_phrases,
            device,
            logits_processors=[logits_processor] if logits_processor is not None else None,
            max_len=5,
        )

        print("Calibrating biases")
        biases = calibrate_biases(
            top_phrases,
            ngram_model_counts,
            biases,
            learning_rate=1e-2
        )

        print("Building trie...")
        trie = build_trie(top_phrases, biases)

        logits_processor = TrieLogitBias(trie)

        print("Validating with new trie...")
        _, average_wer = validate(model, val_loader, processor, whisper, device, i + 1, [logits_processor])
        if writer:
            writer.add_scalar("Loss/validation", val_loss, i + 1)
            writer.add_scalar("WER/validation", average_wer, i + 1)

        print(f"After pass {i + 1}/{passes}, WER: {average_wer:.3f}")

        with open(f"ngram_biases_{i + 1}.pickle", "wb") as file:
            pickle.dump(biases, file)
        with open(f"top_phrases_{i + 1}.pickle", "wb") as file:
            pickle.dump(top_phrases, file)

    print("Done")
