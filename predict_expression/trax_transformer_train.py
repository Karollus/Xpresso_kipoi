from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.trax.models import transformer
from tensor2tensor.trax import layers as tl

def TransformerEncoderMod(vocab_size,
                       n_classes=57,
                       d_model=64,
                       d_ff=128,
                       n_layers=6,
                       n_heads=8,
                       dropout=0.1,
                       max_len=1001,
                       mode='train'):
  """Returns a Transformer encoder model.

  The input to the model is a tensor of tokens.

  Args:
    vocab_size: int: vocab size
    n_classes: how many classes on output
    d_model: int:  depth of embedding
    d_ff: int: depth of feed-forward layer
    n_layers: int: number of encoder/decoder layers
    n_heads: int: number of attention heads
    dropout: float: dropout rate (how much to drop out)
    max_len: int: maximum symbol length for positional encoding
    mode: str: 'train' or 'eval'

  Returns:
    A Transformer model as a layer that maps from a tensor of tokens to
    activations over a set of output classes.
  """
  embedder = [
      tl.Embedding(d_model, vocab_size),
      tl.Dropout(rate=dropout, mode=mode),
      tl.PositionalEncoding(max_len=max_len),
  ]
  return tl.Model([                             #      tokens
      tl.Dup(),                                 # toks toks
      tl.Parallel(embedder, tl.PaddingMask()),  # vecs mask
      [EncoderBlock(d_model, d_ff, n_heads, dropout, mode)
       for _ in range(n_layers)],               # vecs mask
      tl.Parallel([], tl.Drop()),               # ____  0
      tl.LayerNorm(),                           # vecs
      tl.Mean(axis=1),  # Average on length.    # vecs
      tl.Dense(n_classes),                      # vecs
      # tl.LogSoftmax(),                          # vecs
  ])

def main():
    input_vocab_size  = 16
    """Run the Transformer forward and check output shape."""
    single_input_shape = [3, 5]
    input_shape = (tuple(single_input_shape), tuple(single_input_shape))
    model = transformer.Transformer(input_vocab_size, d_model=32, d_ff=64, n_layers=2, n_heads=2)
    final_shape = tl.check_shape_agreement(
        model, input_shape, integer_inputs=True)
    print(final_shape)

if __name__ == '__main__':
    main()
