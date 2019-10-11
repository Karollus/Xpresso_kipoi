import tensorflow as tf

for example in tf.python_io.tf_record_iterator("expression_level_predict_pca-test-00000-of-00010"):
    example = tf.train.Example.FromString(example)
    print(example)
    # this = tf.reshape(example['inputs'], [1, 1001, 117])
    # print(example['inputs'])
    exit()