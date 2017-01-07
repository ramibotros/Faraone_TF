"""Reads one or more csv file into TensorFlow reader
   returns tensor containing features and one with labels"""

import tensorflow as tf
import csv

def read_csv(filename, batch_size):
    temporary_reader = csv.reader(open(filename))
    num_cols = len(next(temporary_reader))
    del temporary_reader

    with tf.name_scope("decoded_CSV_pipeline"):
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)
        decoded = tf.decode_csv(value, record_defaults=[([0.0]) for _ in range(num_cols)])

    with tf.name_scope("shuffled_batching"):
        return tf.train.shuffle_batch(decoded,
                                      batch_size=batch_size,
                                      capacity=batch_size * 50,
                                      min_after_dequeue=batch_size*10)

def data_stream(filename, batch_size):

    all_cols =  read_csv(filename, batch_size)
    all_cols = list(map(tf.squeeze, all_cols))
    return all_cols # list of tensors, one tensor for each CSV column, each tensor with size batch_size