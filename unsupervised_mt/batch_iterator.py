import numpy as np
import torch


class BatchIterator:
    def __init__(self, dataset):
        self.load_sentence = dataset.load_sentence
        self.load_one_hot_sentence = dataset.load_one_hot_sentence
        self.load_embeddings = dataset.load_embeddings
        self.load_len = dataset.load_len
        self.train_ids = dataset.train_ids

    def load_one_language_batch(self, batch_size, language,
                                indices=False, one_hot=True, embeddings=False):
        random_ids = np.random.choice(self.train_ids[language], size=batch_size)
        max_len = np.max([self.load_len(language, idx) for idx in random_ids])

        batch = dict()
        if indices:
            batch.update({
                'indices': np.array([
                    self.load_sentence(language, idx, pad=max_len - self.load_len(language, idx))
                    for idx in random_ids])
            })

        if one_hot:
            batch.update({
                'one-hot': np.array([
                    self.load_one_hot_sentence(language, idx, pad=max_len - self.load_len(language, idx))
                    for idx in random_ids
                ]).transpose(1, 0, 2).astype(np.float32)
            })

        if embeddings:
            batch.update({
                'embeddings': np.array([
                    self.load_embeddings(language, idx, pad=max_len - self.load_len(language, idx))
                    for idx in random_ids
                ]).transpose(1, 0, 2).astype(np.float32)
            })

        return batch

    def load_batch(self, batch_size, indices=False, one_hot=True, embeddings=False):
        """
        Load batch which consists of source sentences, target sentences
        """
        return {'src': self.load_one_language_batch(batch_size, 'src',
                                                    indices=indices, one_hot=one_hot, embeddings=embeddings),
                'tgt': self.load_one_language_batch(batch_size, 'tgt',
                                                    indices=indices, one_hot=one_hot, embeddings=embeddings)}




        