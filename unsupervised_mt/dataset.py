import numpy as np
import torch

from unsupervised_mt.utils import load_embeddings, load_word2nearest, load_train_and_test
import io
from unsupervised_mt.vocabulary import Vocabulary


class Dataset:
    def __init__(self, corp_paths, emb_paths, pairs_paths, max_length=10, test_size=0.1):
        self.languages = ['src', 'tgt']
        self.corp_paths = {l: p for l, p in zip(self.languages, corp_paths)}
        self.emb_paths = {l: p for l, p in zip(self.languages, emb_paths)}
        self.pairs_paths = {l: p for l, p in zip(self.languages, pairs_paths)}
        self.max_length = max_length
        self.test_size = test_size

        # load sentences, embeddings and saved word2nearest
        self.train, self.test = load_train_and_test(*corp_paths, self.max_length, self.test_size, random_state=42)
        self.word2emb = {l: load_embeddings(self.emb_paths[l]) for l in self.languages}
        self.word2nearest = {l: load_word2nearest(self.pairs_paths[l]) for l in self.languages}

        # create vocabularies (including only words from targets,
        # i.e. src vocabulary contains all words having embedding from src sentences
        # and all words from translations of tgt sentences via embedding)
        self.vocabs = {l: Vocabulary([l]) for l in self.languages}
        for l1, l2 in zip(self.languages, self.languages[::-1]):
            for sentence in self.train[l1]:
                for word in sentence.strip().split():
                    if word in self.word2emb[l1]:
                        self.vocabs[l1].add_word(word, l1)
                        self.vocabs[l2].add_word(self.word2nearest[l1][word], l2)

        # embedding matrices
        self.emb_matrix = {
            l: np.array([self.word2emb[l][w.split('-', 1)[1]] for w in self.vocabs[l].index2word], dtype=np.float32)
            for l in self.languages
        }

    def load_sentence(self, language, idx, pad=0, test=False):
        sentence = self.test[idx][0 if language == 'src' else 1] if test else self.train[language][idx]
        return self.vocabs[language].get_indices(sentence, language=language, pad=pad)

    def get_sos_index(self, language):
        return self.vocabs[language].get_sos(language)

    def get_eos_index(self, language):
        return self.vocabs[language].get_eos(language)

    def get_pad_index(self, language):
        return self.vocabs[language].get_pad(language)

    def get_unk_index(self, language):
        return self.vocabs[language].get_unk(language)

    def get_nearest(self, word, l1, l2):
        if word in ['<sos>', 'eos', 'pad', 'unk']:
            return word
        else:
            idx = np.argmax(np.array(list(self.word2emb[l2].values())) @ self.word2emb[l1][word])
            return list(self.word2emb[l2].keys())[idx]

    def save_word2nearest(self, path, l1, l2):
        word2nearest = dict()
        for sentence in self.train[l1]:
            for word in sentence.strip().split():
                if word in self.word2emb[l1] and word not in word2nearest:
                    word2nearest[word] = self.get_nearest(word, l1, l2)
        np.save(path, word2nearest)

    def translate_sentence_word_by_word(self, sentence, l1, l2):
        sentence = [self.vocabs[l1].index2word[index].split('-', 1)[1] for index in sentence]
        sentence = [self.word2nearest[l1][word] for word in sentence]
        sentence = [self.vocabs[l2].word2index[l2 + '-' + word] for word in sentence]
        return sentence

    def translate_batch_word_by_word(self, batch, l1, l2):
        device = batch.device
        batch = batch.transpose(0, 1).tolist()
        batch = [self.translate_sentence_word_by_word(s, l1, l2) for s in batch]
        return torch.tensor(batch, dtype=torch.long, device=device).transpose(0, 1)

    def get_words_from_sentence(self, sentence, language):
        return [self.vocabs[language].index2word[index].split('-', 1)[1] for index in sentence]

    def get_words_from_batch(self, batch, language):
        batch = batch.transpose(0, 1).tolist()
        return [self.get_words_from_sentence(sentence, language) for sentence in batch]






