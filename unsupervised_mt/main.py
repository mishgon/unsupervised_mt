from unsupervised_mt.dataset import Dataset
from unsupervised_mt.train import Trainer
from unsupervised_mt.models import Embedding, DecoderHat, Attention, Discriminator
from unsupervised_mt.batch_loader import BatchLoader
from unsupervised_mt.utils import log_probs2indices, noise

import io
from functools import partial
import torch
import torch.nn as nn
from tqdm import tqdm

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device is {}'.format(device))

# dataset
ds = Dataset(corp_paths=('../data/train.lc.norm.tok.en', '../data/train.lc.norm.tok.fr'),
             emb_paths=('../data/wiki.multi.en.vec', '../data/wiki.multi.fr.vec'),
             pairs_paths=('../data/train_test_src2tgt.npy', '../data/train_test_tgt2src.npy'),
             max_length=20, test_size=0.1)
print('finish loading dataset')

# batch iterator
batch_iter = BatchLoader(ds)

# models
hidden_size = 300
num_layers = 3

src_embedding = Embedding(ds.emb_matrix['src']).to(device)
tgt_embedding = Embedding(ds.emb_matrix['tgt']).to(device)
encoder_rnn = nn.GRU(input_size=src_embedding.embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                     bidirectional=True)
decoder_rnn = nn.GRU(input_size=src_embedding.embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                     bidirectional=True)
attention = Attention(src_embedding.embedding_dim, hidden_size, max_length=ds.max_length, bidirectional=True)
src_hat = DecoderHat(2 * hidden_size, ds.vocabs['src'].size)
tgt_hat = DecoderHat(2 * hidden_size, ds.vocabs['tgt'].size)
discriminator = Discriminator(2 * hidden_size)

# trainer
trainer = Trainer(partial(ds.translate_batch_word_by_word, l1='src', l2='tgt'),
                  partial(ds.translate_batch_word_by_word, l1='tgt', l2='src'),
                  src_embedding, tgt_embedding, encoder_rnn, decoder_rnn, attention, src_hat, tgt_hat,
                  discriminator,
                  ds.get_sos_index('src'), ds.get_sos_index('tgt'),
                  ds.get_eos_index('src'), ds.get_eos_index('tgt'),
                  ds.get_pad_index('src'), ds.get_pad_index('tgt'),
                  device, lr_core=1e-3, lr_disc=1e-3)
#trainer.load('../saved_models/final_result1/')
print('finish initializing models')

# training
batch_size = 30
num_steps = 50000

core_losses = []
disc_losses = []
for i in tqdm(range(num_steps)):
    core_loss, disc_loss = trainer.train_step(batch_iter.load_batch(batch_size), weights=(1, 1, 1))
    core_losses.append(core_loss)
    disc_losses.append(disc_loss)

trainer.save('../saved_models/hidden_300/')

# predict
#predictions = trainer.predict_on_test(batch_iter, batch_size=50, visualize=ds.visualize_batch, n_iters=30)
#with io.open('predictions', 'w') as f:
#    print(*predictions, sep='\n', file=f)
#print('finish predicting')
