from unsupervised_mt.utils.data_utils import load_sentences, load_embeddings
from unsupervised_mt.vocabulary import Vocabulary


class Dataset:
    def __init__(self, src_corp_path, tgt_corp_path, src_emb_path, tgt_emb_path,
                 train_fraction=0.9):
        self.src_corp_path = src_corp_path
        self.tgt_corp_path = tgt_corp_path
        self.src_emb_path = src_emb_path
        self.tgt_emb_path = tgt_emb_path
        self.languages = ['src', 'tgt']

        # load sentences
        self.sentences = {'src': load_sentences(self.src_corp_path),
                          'tgt': load_sentences(self.tgt_corp_path)}

        # load embeddings
        self.word2emb = load_embeddings(self.src_emb_path, language='src')
        self.word2emb.update(load_embeddings(self.tgt_emb_path, language='tgt'))

        # create vocabularies
        self.vocabs = {'src': Vocabulary(['src']),
                       'tgt': Vocabulary(['tgt'])}
        for language in self.languages:
            for sentence in self.sentences[language]:
                self.vocabs[language].add_sentence(sentence, language,
                                                   words_filter=lambda w: language + '-' + w in self.word2emb)

        # split
        self.ids = {'src': list(range(len(self.sentences['src']))),
                    'tgt': list(range(len(self.sentences['tgt'])))}
        test_size = int((1 - train_fraction) * min(len(self.ids['src']), len(self.ids['tgt'])))
        self.test_ids = {'src': self.ids['src'][:test_size],
                         'tgt': self.ids['tgt'][:test_size]}
        self.train_ids = {'src': self.ids['src'][test_size:],
                          'tgt': self.ids['tgt'][test_size:]}

    def load_sentence(self, language, idx, pad=0):
        return self.vocabs[language].get_indices(self.sentences[language][idx], language=language, pad=pad)

    def load_embeddings(self, language, idx, pad=0):
        return [self.word2emb[self.vocabs[language].index2word[index]]
                for index in self.load_sentence(language, idx, pad)]

    def load_len(self, language, idx):
        return len(self.sentences[language][idx].split()) + 1 # length with eos




