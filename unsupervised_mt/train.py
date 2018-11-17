from unsupervised_mt.models import Embedding, Encoder, DecoderHat, \
    Decoder, Seq2Seq, Discriminator, Attention, identity
from unsupervised_mt.losses import translation_loss, classification_loss

import torch
import torch.nn as nn
from torch.optim import SGD


class Trainer:
    def __init__(self, frozen_src2tgt: Seq2Seq, frozen_tgt2src: Seq2Seq,
                 src_embedding: Embedding, tgt_embedding: Embedding,
                 encoder_rnn, decoder_rnn, attention: Attention,
                 src_hat: DecoderHat, tgt_hat: DecoderHat, discriminator: Discriminator,
                 src_sos_index, tgt_sos_index, lr_core=1e-3, lr_disc=1e-3):
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

        self.src2src = Seq2Seq(src_embedding, encoder_rnn, src_embedding, attention, decoder_rnn, src_hat)
        self.src2tgt = Seq2Seq(src_embedding, encoder_rnn, tgt_embedding, attention, decoder_rnn, tgt_hat)
        self.tgt2tgt = Seq2Seq(tgt_embedding, encoder_rnn, tgt_embedding, attention, decoder_rnn, tgt_hat)
        self.tgt2src = Seq2Seq(tgt_embedding, encoder_rnn, src_embedding, attention, decoder_rnn, src_hat)

        self.core_optimizer = SGD(self.core_model.parameters(), lr=lr_core),
        self.discriminator_optimizer = SGD(self.discriminator.parameters(), lr=lr_disc),

    def train_step(self, batch, noise, weights):
        src_batch, tgt_batch = batch['src']['indices'], batch['tgt']['indices']
        core_loss, discriminator_loss = 0, 0

        src2src_dec, src2src_enc = self.src2src(noise(src_batch), src_batch, self.src_sos_index)
        tgt2tgt_dec, tgt2tgt_enc = self.tgt2tgt(noise(tgt_batch), tgt_batch, self.tgt_sos_index)
        tgt2src_dec, tgt2src_enc = self.tgt2src(noise(self._log_probs2indices(self.frozen_src2tgt(src_batch)[0])),
                                                src_batch, self.src_sos_index)
        src2tgt_dec, src2tgt_enc = self.src2tgt(noise(self._log_probs2indices(self.frozen_tgt2src(tgt_batch)[0])),
                                                tgt_batch, self.tgt_sos_index)

        # autoencoding
        core_loss += weights['auto'] * (
                translation_loss(src2src_dec, src_batch) +
                translation_loss(tgt2src_dec, tgt_batch)
        )

        # translating
        core_loss += weights['translate'] * (
            translation_loss(tgt2src_dec, src_batch) +
            translation_loss(src2tgt_dec, tgt_batch)
        )

        # beating discriminator
        core_loss += weights['dicriminator'] * (
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
        core_loss.backward()
        self.core_optimizer.step()

        # update discriminator parameters
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

    @staticmethod
    def _log_probs2indices(decoder_outputs):
        return decoder_outputs.topk(1)[1].squeeze(-1)






