from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.trax.models import transformer
from tensor2tensor.trax import layers as tl
from tensorflow.io import gfile
import pickle
from tensor2tensor.trax import optimizers as trax_opt
from tensor2tensor.trax import learning_rate as lr
from tensor2tensor.trax import trax
from tensor2tensor.trax import inputs as inputs_lib
from tensor2tensor.trax.backend import numpy as np

import h5py
# import numpy as np
# import numpy.random as npr
# import pandas as pd
# from sklearn import preprocessing
# from functools import reduce
# from sklearn import linear_model
from scipy import stats

def main():
    X_test = h5py.File('prepared_data/expr_preds.h5', 'r')['/test_in']
    y_test = h5py.File('prepared_data/expr_preds.h5', 'r')['/test_out']
    #
    # def _select_features(example, feature_list=None):
    #   """Select a subset of features from the example dict."""
    #   feature_list = feature_list or ["inputs", "targets"]
    #   return {f: example[f] for f in feature_list if f in example}
    #
    # def _train_and_eval_dataset_v1(problem_name, data_dir):
    #   """Return train and evaluation datasets, feature info and supervised keys."""
    #   with tf.device("cpu:0"):
    #     problem = t2t_problems.problem(problem_name)
    #     eval_dataset = problem.dataset(tf.estimator.ModeKeys.EVAL, data_dir)
    #     eval_dataset = eval_dataset.map(_select_features)
    #     hparams = problem.get_hparams()
    #     # We take a few training examples to guess the shapes.
    #     input_shapes, target_shapes, examples = [], [], []
    #     if tf.executing_eagerly():
    #       for example in _eager_dataset_iterator(train_dataset.take(3)):
    #         examples.append(example)
    #     else:
    #       example_tensor = train_dataset.make_one_shot_iterator().get_next()
    #       sess = tf.Session()
    #       example1 = sess.run(example_tensor)
    #       example2 = sess.run(example_tensor)
    #       example3 = sess.run(example_tensor)
    #       examples = [example1, example2, example3]
    #   # We use "inputs" as input except for purely auto-regressive tasks like
    #   # language models where "targets" are used as input_key.
    #   input_key = "inputs" if "inputs" in examples[0] else "targets"
    #   supervised_keys = ([input_key], ["targets"])
    #   for example in examples:
    #     input_shapes.append(list(example[input_key].shape))
    #     target_shapes.append(list(example["targets"].shape))
    #   input_vocab_size = hparams.vocab_size[input_key]
    #   target_vocab_size = hparams.vocab_size["targets"]
    #   input_dtype = examples[0][input_key].dtype
    #   target_dtype = examples[0]["targets"].dtype
    #   input_info = _make_info(input_shapes, input_vocab_size, input_dtype)
    #   target_info = _make_info(target_shapes, target_vocab_size, target_dtype)
    #   info = {input_key: input_info, "targets": target_info}
    #   return train_dataset, eval_dataset, info, supervised_keys
    #
    # def shuffle_and_batch_data(dataset,
    #                            target_names,
    #                            features_info,
    #                            training,
    #                            n_devices,
    #                            preprocess_fun=no_preprocess):
    #   """Shuffle and batch the given dataset."""
    #   def append_targets(example):
    #     """Append targets to the example dictionary. Needed for Keras."""
    #     if len(target_names) == 1:
    #       return (example, example[target_names[0]])
    #     targets = {}
    #     for name in target_names:
    #       targets[name] = example[name]
    #     return (example, targets)
    #   dataset = dataset.map(append_targets)
    #   dataset = preprocess_fun(dataset, training)
    #   shapes = {k: features_info[k].shape for k in features_info}
    #   shapes = (shapes, shapes[target_names[0]])
    #   dataset = batch_fun(dataset, training, shapes, target_names, n_devices)
    #   return dataset.prefetch(2)
    #
    # def _train_and_eval_batches(dataset, data_dir, input_name, n_devices):
    #   """Return train and eval batches with input name and shape."""
    #   (train_data, eval_data, features_info, keys) = _train_and_eval_dataset_v1(
    #       dataset, data_dir)
    #   input_names, target_names = keys[0], keys[1]
    #   eval_batches = shuffle_and_batch_data(
    #       eval_data, target_names, features_info, training=False,
    #       n_devices=n_devices)
    #   input_name = input_name or input_names[0]
    #   input_shape = features_info[input_name].shape
    #   input_dtype = features_info[input_name].dtype
    #   return (eval_batches, input_name, list(input_shape), input_dtype)

    print(X_test.shape)

    def make_inputs():
      """Make trax.inputs.Inputs."""
      input_shape = (1001, 117)
      def input_stream():
        yield X_test, y_test

      return inputs_lib.Inputs(
          train_stream=input_stream,
          train_eval_stream=input_stream,
          eval_stream=input_stream,
          input_shape=input_shape,
          input_dtype=np.float32)

    theseinputs = lambda _: make_inputs()

    # trainer = trax.Trainer(
    #   model=model_fn,
    #   loss_fn=trax.loss,
    #   inputs=inputs,
    # )
    # #
    # # Predict with final params
    # inputs = inputs(1).train_stream()
    # model = layers.Serial(model_fn())
    # model(next(inputs)[0], state.opt_state.params)
    #
    # trainer.reset(output_dir1)
    # trainer.evaluate(1)
    # trainer.reset(output_dir2)
    # trainer.evaluate(1)

    """Restore State."""
    with gfile.GFile("model.pkl", "rb") as f:
        (opt_state, step, history, model_state) = pickle.load(f)

    model = tl.Serial(transformer.TransformerEncoder(d_model = 64, d_ff = 128,
      dropout=0.1, max_len=1001, mode='eval', n_classes = 57, n_heads=2,
      n_layers=2))

    # print(len(opt_state[0]))
    # # print(len(params2))
    preds, state = model(X_test[0], params=opt_state[0])
    # print(preds[0])
    # print(X_test[0])
    # print(state)

    trainer = trax.Trainer(
      model=model,
      loss_fn=trax.loss,
      optimizer=trax_opt.SM3,
      lr_schedule=lr.MultifactorSchedule,
      inputs=theseinputs,
    )

    # trainer.evaluate(1)

    #
    # for i in range(0,y_test.shape[1]):
    #      slope, intercept, r_value, p_value, std_err = stats.mstats.linregress(y_test[:,i], y_hat[:,i])
    #      print('Test R^2 %d = %.3f' % (i, r_value**2))

if __name__ == '__main__':
    main()


# # Parameters for TransformerEncoder:
# # ==============================================================================
# TransformerEncoder.d_model = 64
# TransformerEncoder.d_ff = 128
# TransformerEncoder.dropout = 0.1
# TransformerEncoder.max_len = 1001
# TransformerEncoder.mode = 'train'
# TransformerEncoder.n_classes = 57
# TransformerEncoder.n_heads = 2
# TransformerEncoder.n_layers = 2
