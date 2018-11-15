import numpy as np
import torch
import torch.nn as nn


def identity(input):
    return input


class PretrainedEmbedding(nn.Module):
    def __init__(self, emb_matrix):
        super(PretrainedEmbedding, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix))
        self.embedding.requires_grad = False
        self.input_size = self.embedding.num_embeddings
        self.embedding_dim = self.embedding.embedding_dim

    def forward(self, input):
        return self.embedding(input)


class EncoderRNN(nn.Module):
    def __init__(self, embedding: PretrainedEmbedding, rnn):
        super(EncoderRNN, self).__init__()
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
        embedded = self.embedding(inputs)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden


class DecoderHat(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderHat, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input):
        return self.softmax(self.linear(input))


class DecoderRNN(nn.Module):
    def __init__(self, embedding: PretrainedEmbedding, rnn, hat: DecoderHat,
                 max_encoder_length, sos_index, eos_index,
                 use_cuda=False, use_attention=True):
        super(DecoderRNN, self).__init__()
        assert embedding.embedding_dim == rnn.input_size and\
               embedding.input_size == hat.output_size

        self.embedding = embedding
        self.embedding_dim = embedding.embedding_dim
        self.rnn = rnn
        self.hidden_size = rnn.hidden_size
        self.hat = hat
        self.max_encoder_length = max_encoder_length
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.use_cuda = use_cuda
        self.use_attention = use_attention

        if self.use_attention:
            self.attn = nn.Linear(self.hidden_size + self.embedding_dim, self.max_encoder_length, bias=False)
            self.attn_softmax = nn.Softmax(dim=-1)
            self.attn_combine = nn.Linear(self.embedding_dim + self.hidden_size, self.embedding_dim, bias=False)
            self.attn_relu = nn.ReLU()

    def step(self, input, hidden, encoder_outputs):
        """
        input: batch_size
        hidden: n_layers x batch_size x hidden_size
        encoder_outputs: max_length (<= max_length) x batch_size x hidden_size
        embedded: batch_size x embedding_dim
        attn_weights: batch_size x max_length
        attn_applied: batch_size x 1 x hidden_size
        rnn_input: batch_size x embedding_dim
        output: 1 x batch_size x hidden_size

        See figure from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        """
        embedded = self.embedding(input)

        if self.use_attention:
            attn_weights = self.attn_softmax(self.attn(torch.cat((hidden[-1], embedded), dim=1))).unsqueeze(1)
            encoder_length = encoder_outputs.size(0)
            attn_applied = torch.bmm(attn_weights[:, :, :encoder_length], encoder_outputs.transpose(0, 1))
            rnn_input = self.attn_relu(self.attn_combine(torch.cat((embedded, attn_applied.squeeze(1)), 1)))
        else:
            rnn_input = embedded

        output, hidden = self.rnn(rnn_input.unsqueeze(0), hidden)
        output = self.hat(output.squeeze(0))
        return output, hidden

    def init_input(self, batch_size):
        initial_input = torch.full((batch_size,), self.sos_index, dtype=torch.long)
        initial_input = initial_input.cuda() if self.use_cuda else initial_input
        return initial_input

    def forward(self, hidden, encoder_outputs, targets, teacher_forcing_ratio=0.5):
        """
        targets: target_length x batch_size
        """
        input = self.init_input(encoder_outputs.size(1))
        outputs = []
        for t in range(targets.size(0)):
            output, hidden = self.step(input, hidden, encoder_outputs)
            outputs.append(output.unsqueeze(0))
            if np.random.binomial(1, teacher_forcing_ratio):
                input = targets[t]
            else:
                input = torch.topk(output, k=1)[1].squeeze(-1)

        return torch.cat(outputs)


class Seq2Seq(nn.Module):
    def __init__(self, encoder_embedding: PretrainedEmbedding, encoder_rnn,
                 decoder_embedding: PretrainedEmbedding, decoder_rnn, decoder_hat: DecoderHat,
                 max_encoder_length, sos_index, eos_index,
                 use_cuda=False, use_attention=True):
        super(Seq2Seq, self).__init__()
        assert encoder_rnn.hidden_size == decoder_rnn.hidden_size and\
               encoder_rnn.num_layers == decoder_rnn.num_layers

        self.encoder = EncoderRNN(encoder_embedding, encoder_rnn)
        self.decoder = DecoderRNN(decoder_embedding, decoder_rnn, decoder_hat,
                                  max_encoder_length, sos_index, eos_index,
                                  use_cuda, use_attention)

    def forward(self, inputs, targets, teacher_forcing_ratio=0.5):
        encoder_outputs, hidden = self.encoder(inputs)
        decoder_outputs = self.decoder(hidden, encoder_outputs, targets, teacher_forcing_ratio)
        return decoder_outputs, encoder_outputs


class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super(Discriminator, self).__init__()
        self.hidden_size = hidden_size

        self.layers = nn.Sequential(
            nn.Linear(hidden_size, 100), nn.ReLU(),
            nn.Linear(100, 10), nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, encoder_output):
        return self.layers(encoder_output)

