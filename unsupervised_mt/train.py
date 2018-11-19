from unsupervised_mt.models import Embedding, Encoder, DecoderHat, \
    Decoder, Seq2Seq, Discriminator, Attention, identity
from unsupervised_mt.losses import translation_loss, classification_loss
from unsupervised_mt.utils import noise, log_probs2indices

import torch.nn as nn
from torch.optim import SGD


class Trainer:
    def __init__(self, frozen_src2tgt: Seq2Seq, frozen_tgt2src: Seq2Seq,
                 src_embedding: Embedding, tgt_embedding: Embedding,
                 encoder_rnn, decoder_rnn, attention: Attention,
                 src_hat: DecoderHat, tgt_hat: DecoderHat, discriminator: Discriminator,
                 src_sos_index, tgt_sos_index, src_eos_index, tgt_eos_index, src_pad_index, tgt_pad_index,
                 device, lr_core=1e-3, lr_disc=1e-3):
        assert discriminator.hidden_size == encoder_rnn.hidden_size

        self.frozen_src2tgt = frozen_src2tgt
        self.frozen_tgt2src = frozen_tgt2src
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.encoder_rnn = encoder_rnn
        self.decoder_rnn = decoder_rnn
        self.attention = attention
        self.src_hat = src_hat
        self.tgt_hat = tgt_hat
        self.core_model = nn.ModuleList([
            self.encoder_rnn, self.decoder_rnn, self.attention, self.src_hat, self.tgt_hat
        ])
        self.discriminator = discriminator
        self.src_sos_index = src_sos_index
        self.tgt_sos_index = tgt_sos_index
        self.src_eos_index = src_eos_index
        self.tgt_eos_index = tgt_eos_index
        self.src_pad_index = src_pad_index
        self.tgt_pad_index = tgt_pad_index
        self.device = device

        self.core_model.to(device)
        self.discriminator.to(device)

        use_cuda = device.type == 'cuda'
        self.src2src = Seq2Seq(src_embedding, encoder_rnn, src_embedding, attention, decoder_rnn, src_hat, use_cuda)
        self.src2tgt = Seq2Seq(src_embedding, encoder_rnn, tgt_embedding, attention, decoder_rnn, tgt_hat, use_cuda)
        self.tgt2tgt = Seq2Seq(tgt_embedding, encoder_rnn, tgt_embedding, attention, decoder_rnn, tgt_hat, use_cuda)
        self.tgt2src = Seq2Seq(tgt_embedding, encoder_rnn, src_embedding, attention, decoder_rnn, src_hat, use_cuda)

        self.core_optimizer = SGD(self.core_model.parameters(), lr=lr_core)
        self.discriminator_optimizer = SGD(self.discriminator.parameters(), lr=lr_disc)

    def train_step(self, batch, weights=(1, 1, 1), drop_probability=0.1, permutation_constraint=3):
        batch = {l: t.to(self.device) for l, t in batch.items()}

        src2src_dec, src2src_enc = self.src2src(
            noise(batch['src'], self.src_pad_index, drop_probability, permutation_constraint),
            self.src_sos_index, batch['src']
        )
        tgt2tgt_dec, tgt2tgt_enc = self.tgt2tgt(
            noise(batch['tgt'], self.tgt_pad_index, drop_probability, permutation_constraint),
            self.tgt_sos_index, batch['tgt']
        )
        tgt2src_dec, tgt2src_enc = self.tgt2src(
            noise(self.frozen_src2tgt(batch['src']), self.tgt_pad_index, drop_probability, permutation_constraint),
            self.src_sos_index, batch['src']
        )
        src2tgt_dec, src2tgt_enc = self.src2tgt(
            noise(self.frozen_tgt2src(batch['tgt']), self.src_pad_index, drop_probability, permutation_constraint),
            self.tgt_sos_index, batch['tgt']
        )

        # autoencoding
        core_loss = weights[0] * (
                translation_loss(src2src_dec, batch['src']) +
                translation_loss(tgt2tgt_dec, batch['tgt'])
        )

        # translating
        core_loss += weights[1] * (
            translation_loss(tgt2src_dec, batch['src']) +
            translation_loss(src2tgt_dec, batch['tgt'])
        )

        # beating discriminator
        core_loss += weights[2] * (
            classification_loss(self.discriminator(src2src_enc), 'tgt') +
            classification_loss(self.discriminator(tgt2tgt_enc), 'src') +
            classification_loss(self.discriminator(tgt2src_enc), 'src') +
            classification_loss(self.discriminator(src2tgt_enc), 'tgt')
        )

        # training discriminator
        discriminator_loss = classification_loss(self.discriminator(src2src_enc), 'src') + \
                             classification_loss(self.discriminator(tgt2tgt_enc), 'tgt') + \
                             classification_loss(self.discriminator(tgt2src_enc), 'tgt') + \
                             classification_loss(self.discriminator(src2tgt_enc), 'src')

        # update general model's parameters
        self.core_optimizer.zero_grad()
        core_loss.backward(retain_graph=True)
        self.core_optimizer.step()

        # update discriminator parameters
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward(retain_graph=True)
        self.discriminator_optimizer.step()











