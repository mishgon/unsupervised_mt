import numpy as np
import torch

from unsupervised_mt.utils import load_sentences, load_embeddings
from unsupervised_mt.vocabulary import Vocabulary


class Dataset:
    def __init__(self, languages, corp_paths, emb_paths, max_length=10, train_fraction=0.9):
        self.languages = languages
        self.corp_paths = {l: p for l, p in zip(self.languages, corp_paths)}
        self.emb_paths = {l: p for l, p in zip(self.languages, emb_paths)}
        self.max_length = max_length
        self.train_fraction = train_fraction

        # load sentences
        self.sentences = {
            l: load_sentences(self.corp_paths[l], max_length=self.max_length)
            for l in self.languages
        }

        # load embeddings
        self.word2emb = {l: load_embeddings(self.emb_paths[l]) for l in self.languages}

        # create vocabularies
        self.vocabs = {l: Vocabulary([l]) for l in self.languages}
        for language in self.languages:
            for word in self.word2emb[language].keys():
                self.vocabs[language].add_word(word, language)

        # embedding matrix
        self.emb_matrix = {
            l: np.array([self.word2emb[l][w.split('-', 1)[1]] for w in self.vocabs[l].index2word], dtype=np.float32)
            for l in self.languages
        }

        # split
        self.ids = {l: list(range(len(self.sentences[l]))) for l in self.languages}
        test_size = int((1 - self.train_fraction) * np.min([len(self.ids[l]) for l in self.languages]))
        self.test_ids = {l: self.ids[l][:test_size] for l in self.languages}
        self.train_ids = {l: self.ids[l][test_size:] for l in self.languages}

        # word to nearest word in another language
        if len(self.languages) == 2:
            l1, l2 = self.languages
            self.word2nearest = {l1: [self.get_nearest(index, l1, l2) for index in range(self.vocabs[l1].size)],
                                 l2: [self.get_nearest(index, l2, l1) for index in range(self.vocabs[l2].size)]}

    def load_sentence(self, language, idx, pad=0):
        return self.vocabs[language].get_indices(self.sentences[language][idx], language=language, pad=pad)

    def load_len(self, language, idx):
        return len(self.load_sentence(language, idx))

    def get_sos_index(self, language):
        return self.vocabs[language].get_sos(language)

    def get_eos_index(self, language):
        return self.vocabs[language].get_eos(language)

    def get_pad_index(self, language):
        return self.vocabs[language].get_pad(language)

    def get_unk_index(self, language):
        return self.vocabs[language].get_unk(language)

    def get_nearest(self, index, language1, language2):
        if index == self.get_sos_index(language1):
            return self.get_sos_index(language2)
        elif index == self.get_eos_index(language1):
            return self.get_eos_index(language2)
        elif index == self.get_pad_index(language1):
            return self.get_pad_index(language2)
        elif index == self.get_unk_index(language1):
            return self.get_unk_index(language2)
        else:
            word = self.vocabs[language1].index2word[index].split('-', 1)[1]
            return np.argmax(self.emb_matrix[language2] @ self.word2emb[language1][word])

    def translate_sentence_word_by_word(self, sentence, language):
        return [self.word2nearest[language][index] for index in sentence]

    def translate_batch_word_by_word(self, batch, language):
        device = batch.device
        batch = batch.transpose(0, 1).tolist()
        batch = [self.translate_sentence_word_by_word(s, language) for s in batch]
        return torch.tensor(batch, dtype=torch.long, device=device).transpose(0, 1)

    def print_sentence(self, sentence, language):
        print([self.vocabs[language].index2word[index].split('-', 1)[1] for index in sentence])

    def print_batch(self, batch, language):
        batch = batch.transpose(0, 1).tolist()
        for sentence in batch:
            self.print_sentence(sentence, language)






