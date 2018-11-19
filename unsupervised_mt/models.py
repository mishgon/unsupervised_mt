import numpy as np
import torch
import torch.nn as nn


def identity(input):
    return input


class Embedding(nn.Module):
    def __init__(self, emb_matrix):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix))
        self.embedding.requires_grad = False
        self.vocab_size = self.embedding.num_embeddings
        self.embedding_dim = self.embedding.embedding_dim

    def forward(self, input):
        return self.embedding(input)


class Encoder(nn.Module):
    def __init__(self, embedding: Embedding, rnn):
        super(Encoder, self).__init__()
        assert embedding.embedding_dim == rnn.input_size

        self.embedding = embedding
        self.embedding_dim = embedding.embedding_dim
        self.rnn = rnn
        self.hidden_size = rnn.hidden_size

    def forward(self, inputs):
        """
        inputs: length x batch_size
        outputs: length x batch_size x hidden_size
        hidden: n_layers x batch_size x hidden_size
        """
        outputs, hidden = self.rnn(self.embedding(inputs))
        return outputs, hidden


class DecoderHat(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(DecoderHat, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input):
        return self.softmax(self.linear(input))


class Attention(nn.Module):
    def __init__(self, embedding_dim, hidden_size, max_length):
        """
        input: batch_size x vocab_size
        hidden: n_layers x batch_size x hidden_size
        encoder_outputs: max_length (<= max_length) x batch_size x hidden_size
        attn_weights: batch_size x max_length
        attn_applied: batch_size x 1 x hidden_size

        See figure from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        """
        super(Attention, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.attn = nn.Linear(self.hidden_size + self.embedding_dim, self.max_length, bias=False)
        self.attn_softmax = nn.Softmax(dim=-1)
        self.attn_combine = nn.Linear(self.embedding_dim + self.hidden_size, self.embedding_dim, bias=False)
        self.attn_relu = nn.ReLU()

    def forward(self, embedded, hidden, encoder_outputs):
        attn_weights = self.attn_softmax(self.attn(torch.cat((hidden[-1], embedded), dim=1))).unsqueeze(1)
        length = encoder_outputs.size(0)
        attn_applied = torch.bmm(attn_weights[:, :, :length], encoder_outputs.transpose(0, 1))
        return self.attn_relu(self.attn_combine(torch.cat((embedded, attn_applied.squeeze(1)), 1)))


class Decoder(nn.Module):
    def __init__(self, embedding: Embedding, attention, rnn, hat: DecoderHat, use_cuda=False):
        super(Decoder, self).__init__()
        assert embedding.embedding_dim == rnn.input_size and \
               embedding.vocab_size == hat.vocab_size and \
               attention.embedding_dim == embedding.embedding_dim

        self.embedding = embedding
        self.embedding_dim = embedding.embedding_dim
        self.attention = attention
        self.rnn = rnn
        self.hidden_size = rnn.hidden_size
        self.hat = hat
        self.use_cuda = use_cuda

    def step(self, input, hidden, encoder_outputs):
        """
        input: batch_size
        hidden: n_layers x batch_size x hidden_size
        encoder_outputs: max_length (<= max_length) x batch_size x hidden_size
        embedded: batch_size x embedding_dim
        rnn_input: batch_size x embedding_dim
        output: 1 x batch_size x hidden_size
        """
        embedded = self.embedding(input)
        rnn_input = self.attention(embedded, hidden, encoder_outputs) if self.attention else embedded
        output, hidden = self.rnn(rnn_input.unsqueeze(0), hidden)
        output = self.hat(output.squeeze(0))
        return output, hidden

    def init_input(self, batch_size, sos_index):
        initial_input = torch.full((batch_size,), sos_index, dtype=torch.long)
        initial_input = initial_input.cuda() if self.use_cuda else initial_input
        return initial_input

    def forward(self, hidden, encoder_outputs, sos_index,
                targets, teacher_forcing_ratio=0.5):
        """
        targets: target_length x batch_size
        """
        input = self.init_input(encoder_outputs.size(1), sos_index)
        outputs = []
        for t in range(targets.size(0)):
            output, hidden = self.step(input, hidden, encoder_outputs)
            outputs.append(output.unsqueeze(0))
            if np.random.binomial(1, teacher_forcing_ratio):
                input = targets[t]
            else:
                input = torch.topk(output, k=1)[1].squeeze(-1)

        return torch.cat(outputs)

    def evaluate(self, hidden, encoder_outputs, sos_index, eos_index, n_iters=None):
        """
        hidden: n_layers x batch_size x hidden_size
        encoder_outputs: length x batch_size x hidden_size
        input: batch_size
        """
        input = self.init_input(hidden.size(1), sos_index)
        outputs = []
        ended = np.zeros(hidden.size(1))
        while ~np.all(ended) and n_iters != 0:
            output, hidden = self.step(input, hidden, encoder_outputs)
            outputs.append(output.unsqueeze(0))
            input = torch.topk(output, k=1)[1].squeeze(-1)
            ended += (input == eos_index).cpu().numpy() if self.use_cuda else (input == eos_index).numpy()
            if n_iters is not None:
                n_iters -= 1

        return torch.cat(outputs)


class Seq2Seq(nn.Module):
    def __init__(self, encoder_embedding: Embedding, encoder_rnn,
                 decoder_embedding: Embedding, attention, decoder_rnn, decoder_hat: DecoderHat,
                 use_cuda=False):
        super(Seq2Seq, self).__init__()
        assert encoder_rnn.hidden_size == decoder_rnn.hidden_size and \
               encoder_rnn.num_layers == decoder_rnn.num_layers

        self.encoder = Encoder(encoder_embedding, encoder_rnn)
        self.decoder = Decoder(decoder_embedding, attention, decoder_rnn, decoder_hat, use_cuda)

    def forward(self, inputs, sos_index, targets, teacher_forcing_ratio=0.5):
        encoder_outputs, hidden = self.encoder(inputs)
        decoder_outputs = self.decoder(hidden, encoder_outputs, sos_index, targets, teacher_forcing_ratio)
        return decoder_outputs, encoder_outputs

    def evaluate(self, inputs, sos_index, eos_index, n_iters=None):
        encoder_outputs, hidden = self.encoder(inputs)
        decoder_outputs = self.decoder.evaluate(hidden, encoder_outputs, sos_index, eos_index, n_iters)
        return decoder_outputs


class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super(Discriminator, self).__init__()
        self.hidden_size = hidden_size

        self.layers = nn.Sequential(
            nn.Linear(hidden_size, 100), nn.ReLU(),
            nn.Linear(100, 10), nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, encoder_outputs):
        outputs = []
        for t in range(encoder_outputs.size(0)):
            outputs.append(self.layers(encoder_outputs[t]).unsqueeze(0))

        return torch.cat(outputs)
