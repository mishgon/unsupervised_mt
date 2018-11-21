import numpy as np
import torch

from unsupervised_mt.utils import load_sentences, load_embeddings, load_word2nearest
import io
from unsupervised_mt.vocabulary import Vocabulary


class Dataset:
    def __init__(self, languages, corp_paths, emb_paths, pairs_paths, max_length=10, train_fraction=0.9):
        self.languages = languages
        self.corp_paths = {l: p for l, p in zip(self.languages, corp_paths)}
        self.emb_paths = {l: p for l, p in zip(self.languages, emb_paths)}
        self.pairs_paths = {l: p for l, p in zip(self.languages, pairs_paths)}
        self.max_length = max_length
        self.train_fraction = train_fraction

        # load sentences
        self.sentences = {
            l: load_sentences(self.corp_paths[l], max_length=self.max_length)
            for l in self.languages
        }

        # load embeddings
        self.word2emb = {l: load_embeddings(self.emb_paths[l]) for l in self.languages}

        # load word2nearest
        self.word2nearest = {l: load_word2nearest(self.pairs_paths[l]) for l in self.languages}

        # create vocabularies (including only words from targets,
        # i.e. src vocabulary contains all words having embedding from src sentences
        # and all words from translations of tgt sentences via embedding)
        self.vocabs = {l: Vocabulary([l]) for l in self.languages}
        for l1, l2 in zip(self.languages, self.languages[::-1]):
            for sentence in self.sentences[l1]:
                for word in sentence.strip().split():
                    if word in self.word2emb[l1]:
                        self.vocabs[l1].add_word(word, l1)
                        self.vocabs[l2].add_word(self.word2nearest[l1][word], l2)

        # embedding matrices
        self.emb_matrix = {
            l: np.array([self.word2emb[l][w.split('-', 1)[1]]
                         for w in self.vocabs[l].index2word], dtype=np.float32)
            for l in self.languages
        }

        # split
        self.ids = {l: list(range(len(self.sentences[l]))) for l in self.languages}
        test_size = int((1 - self.train_fraction) * np.min([len(self.ids[l]) for l in self.languages]))
        self.test_ids = {l: self.ids[l][:test_size] for l in self.languages}
        self.train_ids = {l: self.ids[l][test_size:] for l in self.languages}

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

    def get_nearest(self, word, l1, l2):
        if word in ['<sos>', 'eos', 'pad', 'unk']:
            return word
        else:
            idx = np.argmax(np.array(list(self.word2emb[l2].values())) @ self.word2emb[l1][word])
            return list(self.word2emb[l2].keys())[idx]

    def save_nearest(self, l1, l2):
        with io.open(l1 + '2' + l2 + '.txt', 'x') as f:
            for sentence in self.sentences[l1]:
                for word in sentence.strip().split():
                    if word in self.word2emb[l1]:
                        f.write(word + ' ' + self.get_nearest(word, l1, l2) + '\n')

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

    def print_sentence(self, sentence, language):
        print([self.vocabs[language].index2word[index].split('-', 1)[1] for index in sentence])

    def print_batch(self, batch, language):
        batch = batch.transpose(0, 1).tolist()
        for sentence in batch:
            self.print_sentence(sentence, language)






