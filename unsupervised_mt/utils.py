import io
import unicodedata
import re
import numpy as np
from typing import List
import torch


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r'([.!?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    return s.strip()


def load_embeddings(emb_path, language, encoding='utf-8', newline='\n', errors='ignore'):
    word2emb = dict()
    with io.open(emb_path, 'r', encoding=encoding, newline=newline, errors=errors) as f:
        emb_dim = int(f.readline().split()[1])
        for w in ['<sos>', '<eos>', '<unk>', '<pad>']:
            word2emb[language + '-' + w] = np.random.uniform(0, 1, size=emb_dim)

        for line in f.readlines()[1:]:
            orig_word, emb = line.rstrip().split(' ', 1)
            emb = np.fromstring(emb, sep=' ')
            word = normalize_string(orig_word)

            # if word is not in dictionary or if it is, but better embedding is provided
            if word not in word2emb or word == orig_word:
                word2emb[language + '-' + word] = emb
    return word2emb


def load_sentences(corp_path, max_length=10, encoding='utf-8', newline='\n', errors='ignore'):
    with io.open(corp_path, 'r', encoding=encoding, newline=newline, errors=errors) as f:
        sentences = list(map(normalize_string, f.readlines()))
    return list(filter(lambda s: len(s.split(' ')) < max_length, sentences))


def pad_monolingual_batch(batch: List[int], pad_index):
    max_length = np.max([len(s) for s in batch])
    return [s + (max_length - len(s)) * [pad_index] for s in batch]


def noise_sentence(sentence: List[int], pad_index, drop_probability=0.1, permutation_constraint=3):
    sentence = list(filter(lambda index: index != pad_index, sentence))
    eos = sentence[-1:]
    sentence = sentence[:-1]
    np.random.seed()

    sentence = list(filter(lambda index: np.random.binomial(1, 1 - drop_probability), sentence))

    # notation from paper
    alpha = permutation_constraint + 1
    q = np.arange(len(sentence)) + np.random.uniform(0, alpha, size=len(sentence))
    sentence = list(np.array(sentence)[np.argsort(q)])

    return sentence + eos


def noise(batch: torch.Tensor, pad_index, drop_probability=0.1, permutation_constraint=3):
    device = batch.device
    batch = batch.transpose(0, 1).tolist()
    batch = [noise_sentence(s, pad_index, drop_probability, permutation_constraint) for s in batch]
    batch = pad_monolingual_batch(batch, pad_index)
    return torch.tensor(batch, dtype=torch.long, device=device).transpose(0, 1)


def log_probs2indices(decoder_outputs):
    return decoder_outputs.topk(k=1)[1].squeeze(-1)

