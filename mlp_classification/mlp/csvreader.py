"""Reads one or more csv file into TensorFlow reader
   returns tensor containing features and one with labels"""

import tensorflow as tf

def read_csv(batch_size, filenames, record_defaults):
    with tf.name_scope("decoded_CSV_pipeline"):
        filename_queue = tf.train.string_input_producer([filenames])
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)
        decoded = tf.decode_csv(value, record_defaults = record_defaults)
    with tf.name_scope("shuffled_batching"):
        return tf.train.shuffle_batch(decoded,
                                      batch_size=batch_size,
                                      capacity=batch_size * 50,
                                      min_after_dequeue=batch_size)

def features_labels(batch_size, filename, record_defaults, feature_names):
    """assumes features are first columns in file, label is last
       Example:
        features, labels = features_labels(50, "/home/steve/TF_Programs/data/iris_training.csv",  
        [[0.0],[0.0], [0.0], [0.0], [""]], 
        ["sepal_length", "sepal_width", "petal_length", "petal_width"])
       
        record_defaults indicate if each column is a float or string"""

    all_cols =  read_csv(batch_size, filename, record_defaults)
    features, labels = all_cols[:-1], all_cols[-1]

    with tf.name_scope("features"):
        features = tf.squeeze(tf.transpose(tf.pack([features])))
    with tf.name_scope("labels"):
        labels = tf.squeeze(tf.reshape(labels, [batch_size, 1]))

    return features, labels, batch_size

