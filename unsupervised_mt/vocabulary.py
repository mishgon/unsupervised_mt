from collections import Counter


class Vocabulary:
    """
    Code is borrowed from https://github.com/IlyaGusev/UNMT/blob/master/utils/vocabulary.py
    """
    def __init__(self, languages):
        self.languages = languages
        self.index2word = list()
        self.word2index = dict()
        self.word2count = Counter()
        self.size = 0
        for language in self.languages:
            for w in ['<sos>', '<eos>', '<unk>', '<pad>']:
                self.add_word(w, language)

    def get_sos(self, language):
        return self.word2index[language + '-<sos>']

    def get_eos(self, language):
        return self.word2index[language + '-<eos>']

    def get_unk(self, language):
        return self.word2index[language + '-<unk>']

    def get_pad(self, language):
        return self.word2index[language + '-<pad>']

    def get_index(self, word, language):
        word = language + '-' + word
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.get_unk(language)

    def add_sentence(self, sentence, language, words_filter):
        for word in sentence.strip().split():
            self.add_word(word, language, words_filter)

    def add_word(self, word, language, words_filter=lambda w: True):
        if words_filter(word):
            word = language + '-' + word
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.word2count[word] += 1
                self.index2word.append(word)
                self.size += 1
            else:
                self.word2count[word] += 1

    def get_indices(self, sentence, language, pad):
        return [self.get_index(w, language) for w in sentence.strip().split()] +\
               [self.get_eos(language)] + pad * [self.get_pad(language)]

