import numpy as np
import torch


class BatchIterator:
    def __init__(self, dataset):
        self.load_sentence = dataset.load_sentence
        self.load_embeddings = dataset.load_embeddings
        self.load_len = dataset.load_len
        self.train_ids = dataset.train_ids

    def load_one_language_batch(self, batch_size, language):
        random_ids = np.random.choice(self.train_ids[language], size=batch_size)
        max_len = np.max([self.load_len(language, idx) for idx in random_ids])
        return {
            'indices': np.array([
                self.load_sentence(language, idx, pad=max_len - self.load_len(language, idx))
                for idx in random_ids
            ]),
            'embeddings': np.array([
                self.load_embeddings(language, idx, pad=max_len - self.load_len(language, idx))
                for idx in random_ids
            ]).transpose(1, 0, 2).astype(np.float32)
        }

    def load_batch(self, batch_size):
        """
        Load batch which consists of source sentences, target sentences
        """
        return {'src': self.load_one_language_batch(batch_size, 'src'),
                'tgt': self.load_one_language_batch(batch_size, 'tgt')}




        