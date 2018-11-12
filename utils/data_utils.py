import io
import unicodedata
import re
import numpy as np


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s.strip()


def load_embeddings(emb_path, encoding='utf-8', newline='\n', errors='ignore'):
    word2emb = dict()
    with io.open(emb_path, 'r', encoding=encoding, newline=newline, errors=errors) as f:
        for line in f.readlines()[1:]:
            orig_word, emb = line.rstrip().split(' ', 1)
            emb = np.fromstring(emb, sep=' ')
            word = normalize_string(orig_word)

            # if word is not in dictionary or if it is, but better embedding is provided
            if word not in word2emb or word == orig_word:
                word2emb[word] = emb
    return word2emb


def load_sentences(corp_path, max_len=10, encoding=encoding, new_line='\n', errors='ignore'):
    with io.open(corp_path, 'r', encoding=encoding, newline=newline, errors=errors) as f:
        sentences = list(map(normalize_string, f.readlines()))
    return list(filter(lambda s: len(s.split(' ')) <= max_len, sentences))
