from collections import Counter

from utils.data_utils import load_sentences, load_embeddings


class Vocabulary:
    """
    Code is partially borrowed from https://github.com/IlyaGusev/UNMT/blob/master/utils/vocabulary.py
    """
    def __init__(self, languages):
        self.languages = languages
        self.index2word = list()
        self.word2index = dict()
        self.word2count = Counter()
        for language in self.languages:
            self.add_word('<sos>', language)
            self.add_word('<eos>', language)

    def add_sentence(self, sentence, language, condition):
        for word in sentence.strip().split():
            self.add_word(word, language, condition)

    def add_word(self, word, language, condition):
        if condition(word):
            word = language+"-"+word
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.word2count[word] += 1
                self.index2word.append(word)
            else:
                self.word2count[word] += 1


class Dataset:
    def __init__(self, src_corp_path, tgt_corp_path, src_emb_path, tgt_emb_path):
        self.src_corp_path = src_corp_path
        self.tgt_corp_path = tgt_corp_path
        self.src_emb_path = src_emb_path
        self.tgt_emb_path = tgt_emb_path

        self.src_sentences = load_sentences(self.src_corp_path)
        self.tgt_sentences = load_sentences(self.tgt_corp_path)
        self.src_word2emb = load_embeddings(self.src_emb_path)
        self.tgt_word2emb = load_embeddings(self.tgt_emb_path)

        self.src_vocab = Vocabulary(['src'])
        for sentence in self.src_sentences:
            self.src_vocab.add_sentence(sentence, 'src', lambda w: w in self.src_word2emb)

        self.tgt_vocab = Vocabulary(['tgt'])
        for sentence in self.src_sentences:
            self.tgt_vocab.add_sentence(sentence, 'tgt', lambda w: w in self.tgt_word2emb)
