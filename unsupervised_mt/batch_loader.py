import numpy as np
import torch
from unsupervised_mt.utils import pad_monolingual_batch


class BatchLoader:
    def __init__(self, dataset):
        self.languages = dataset.languages
        self.load_sentence = dataset.load_sentence
        self.train_ids = {l: np.arange(len(dataset.train[l])) for l in self.languages}
        self.test_ids = np.arange(len(dataset.test))
        self.pad_index = {l: dataset.vocabs[l].get_pad(l) for l in self.languages}

    def load_raw_monolingual_batch(self, batch_size, language, random_state=None, test=False, ids=None):
        if random_state is not None:
            np.random.seed(random_state)

        if ids is None:
            ids = np.random.choice(self.test_ids if test else self.train_ids[language], size=batch_size)

        return [self.load_sentence(language, idx, test=test) for idx in ids]

    def load_monolingual_batch(self, batch_size, language, random_state=None, test=False, ids=None):
        return torch.tensor(pad_monolingual_batch(
            self.load_raw_monolingual_batch(batch_size, language, random_state, test=test, ids=ids),
            self.pad_index[language]
        ), dtype=torch.long).transpose(0, 1)

    def load_batch(self, batch_size, random_state=None, test=False, ids=None):
        return {l: self.load_monolingual_batch(batch_size, l, random_state, test=test, ids=ids) for l in self.languages}





