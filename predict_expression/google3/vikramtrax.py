# Lint as: python3
"""Want to train transformer encoder for DNA interpretation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function
import copy
import os
import random
import gin
import numpy as np
import tensorflow.google as tf
from trax.supervised import inputs as trax_input
from trax import layers as tl
# from trax.backend import numpy as np
def shuffled(x):
  """Functional shuffle."""
  y = copy.copy(x)
  random.shuffle(y)
  return y
@gin.configurable()
def vikram_inputs(
    #    n_devices,
    n_batch=16,
    num_input_timestamps=1001,
    num_embed=117,
    num_output_predictions=57,
    cns_path='/readahead/200M/cns/tm-d/home/vikrama',
    shuffle_files=False,
    batch_shuffle_size=128,
    n_prefetch=4):
  """Prepare inputs."""
  #  if n_batch % n_devices != 0:
  #    tf.logging.fatal(
  #        'n_devices[%d] should divide the first dimension of n_batch[%s]',
  #        n_devices, n_batch)
  # grab filenames from CNS
  train_files = tf.gfile.Glob(os.path.join(cns_path, '*train*'))
  dev_files = tf.gfile.Glob(os.path.join(cns_path, '*dev*'))
  test_files = tf.gfile.Glob(os.path.join(cns_path, '*test*'))
  if shuffle_files:
    train_files = shuffled(train_files)
    dev_files = shuffled(dev_files)
    test_files = shuffled(test_files)
  # tf.example proto parsing
  feature_description = {
      'inputs': tf.VarLenFeature(tf.float32),
      'targets': tf.VarLenFeature(tf.float32),
  }
  def _parse_example(x):
    return tf.parse_example([x], feature_description)
  # reshaping
  input_shape = [-1, num_input_timestamps, num_embed]
  input_dtype = np.float32
  target_shape = [-1, 1, num_output_predictions]
  target_dtype = np.float32
  def _reshape(x):
    inps = x['inputs'].values
    inps = tf.reshape(inps, input_shape)
    inps = inps[:, 499:501, :]
    tgts = x['targets'].values
    tgts = tf.reshape(tgts, target_shape)
    return (inps, tgts)
  # tf.data chain
  def make_dataset_iterator(data_files):
    return (
        tf.data.TFRecordDataset(data_files)
        .map(_parse_example)
        .repeat()
        .shuffle(batch_shuffle_size)
        .batch(n_batch, drop_remainder=True)
        .map(_reshape)
        .prefetch(n_prefetch)
        .as_numpy_iterator()
    )
  # make dataset iterators
  ds_train = lambda: make_dataset_iterator(train_files)
  ds_dev = lambda: make_dataset_iterator(dev_files)
  ds_test = lambda: make_dataset_iterator(test_files)
  # put information in form trax wants
  input_shape_without_batch = list(input_shape)[1:]
  target_shape_without_batch = list(target_shape)[1:]
  return trax_input.Inputs(
      train_stream=ds_train,
      train_eval_stream=ds_dev,
      eval_stream=ds_test,
      input_shape=input_shape_without_batch,
      input_dtype=input_dtype,
      target_shape=target_shape_without_batch,
      target_dtype=target_dtype)
@tl.layer()
def no_padding_mask(x, **kwargs):
  del kwargs
  return np.reshape(np.ones(x.shape[0]*x.shape[-2], dtype=x.dtype),
                    (x.shape[0], 1, 1, x.shape[-2]))
@tl.layer()
def subsetdata(x, startidx, stopidx, **kwargs):
  del kwargs
  x = x[:, startidx:stopidx, :]
  return x
def feed_forward_block(d_model, d_ff, dropout, layer_idx, mode, activation):
  """Returns a list of layers implementing a feed-forward block.
  Args:
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    dropout: float: dropout rate (how much to drop out)
    layer_idx: which layer are we at (for bookkeeping)
    mode: str: 'train' or 'eval'
    activation: the non-linearity in feed-forward layer
  Returns:
    A list of layers which maps vectors to vectors.
  """
  dropout_middle = tl.Dropout(
      rate=dropout, name='ff_middle_%d' % layer_idx, mode=mode)
  dropout_final = tl.Dropout(
      rate=dropout, name='ff_final_%d' % layer_idx, mode=mode)
  return [
      tl.LayerNorm(),
      tl.Dense(d_ff),
      activation(),
      dropout_middle,
      tl.Dense(d_model),
      dropout_final,
  ]
def encoder_block(d_model, d_ff, n_heads, dropout, layer_idx, mode,
                  ff_activation):
  """Returns a list of layers that implements a Transformer encoder block.
  The input to the layer is a pair, (activations, mask), where the mask was
  created from the original source tokens to prevent attending to the padding
  part of the input.
  Args:
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    layer_idx: which layer are we at (for bookkeeping)
    mode: str: 'train' or 'eval'
    ff_activation: the non-linearity in feed-forward layer
  Returns:
    A list of layers that maps (activations, mask) to (activations, mask).
  """
  attention = tl.Attention(
      d_model, n_heads=n_heads, dropout=dropout, mode=mode)
  dropout_ = tl.Dropout(
      rate=dropout, name='dropout_enc_attn', mode=mode)
  feed_forward = feed_forward_block(
      d_model, d_ff, dropout, layer_idx, mode, ff_activation)
  return [
      tl.Residual(
          tl.LayerNorm(),
          attention,
          dropout_,
      ),
      tl.Residual(
          feed_forward
      ),
  ]
@gin.configurable()
def non_tokenizing_transformer(n_classes=57,
                               d_model=512,
                               d_ff=2048,
                               n_layers=2,
                               n_heads=8,
                               dropout=0.1,
                               max_len=1001,
                               mode='train',
                               ff_activation=tl.Relu):
  """Returns a Transformer encoder model.
  The input to the model is a tensor of tokens.
  Args:
    n_classes: how many classes on output
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_layers: int: number of encoder/decoder layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train' or 'eval'
    ff_activation: type of activation function to use
  Returns:
    A Transformer model as a layer that maps from a tensor of tokens to
    activations over a set of output classes.
  """
  positional_encoder = [
      tl.Dense(d_model),
      tl.Dropout(rate=dropout, name='emb_dropout', mode=mode),
      tl.PositionalEncoding(max_len=max_len)
  ]
  encoder_blocks = [
      encoder_block(d_model, d_ff, n_heads, dropout, i, mode, ff_activation)
      for i in range(n_layers)]
  # Assemble and return the model.
  return tl.Serial(                               # toks
      # Encode.
      tl.Branch(positional_encoder, no_padding_mask()),  # vecs masks
      encoder_blocks,                             # vecs masks
      tl.Select([0], n_in=2),                     # vecs
      tl.LayerNorm(),                             # vecs
      # subsetdata(startidx=498, stopidx=501),
      # Map to output categories.
      tl.Mean(axis=1),                            # vecs
      tl.Dense(n_classes),                        # vecs
  )
