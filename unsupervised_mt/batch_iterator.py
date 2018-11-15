import numpy as np
import torch


class BatchIterator:
    def __init__(self, dataset):
        self.languages = dataset.languages
        self.load_sentence = dataset.load_sentence
        self.load_one_hot_sentence = dataset.load_one_hot_sentence
        self.load_embeddings = dataset.load_embeddings
        self.load_len = dataset.load_len
        self.train_ids = dataset.train_ids

    def load_one_language_batch(self, batch_size, language,
                                indices=True, one_hot=False, embeddings=False):
        random_ids = np.random.choice(self.train_ids[language], size=batch_size)
        max_len = np.max([self.load_len(language, idx) for idx in random_ids])

        batch = dict()
        if indices:
            batch.update({
                'indices': torch.tensor([
                    self.load_sentence(language, idx, pad=max_len - self.load_len(language, idx))
                    for idx in random_ids], dtype=torch.long).transpose(0, 1)
            })

        if one_hot:
            batch.update({
                'one-hot': torch.from_numpy(np.array([
                    self.load_one_hot_sentence(language, idx, pad=max_len - self.load_len(language, idx))
                    for idx in random_ids
                ]).astype(np.float32)).transpose(0, 1)
            })

        if embeddings:
            batch.update({
                'embeddings': torch.from_numpy(np.array([
                    self.load_embeddings(language, idx, pad=max_len - self.load_len(language, idx))
                    for idx in random_ids
                ]).transpose(1, 0, 2).astype(np.float32)).transpose(0, 1)
            })

        return batch

    def load_batch(self, batch_size, indices=True, one_hot=False, embeddings=False):
        """
        Load batch which consists of source sentences, target sentences
        """
        return {
            l: self.load_one_language_batch(batch_size, l, indices=indices, one_hot=one_hot, embeddings=embeddings)
            for l in self.languages
        }