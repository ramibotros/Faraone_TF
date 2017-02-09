"""Reads one or more csv file into TensorFlow reader
   returns tensor containing features and one with labels"""

import csv

import tensorflow as tf
from tensorflow.contrib.training import stratified_sample


def read_csv(filename, batch_size, stratify_task="", config=None):
    temporary_reader = csv.reader(open(filename))
    num_cols = len(next(temporary_reader))
    del temporary_reader

    with tf.name_scope("decoded_CSV_pipeline"):
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)
        decoded = tf.decode_csv(value, record_defaults=[([0.0]) for _ in range(num_cols)])

    if stratify_task:
        with tf.name_scope("stratification"):
            assert stratify_task in config
            assert config.get(stratify_task, "type", fallback="") == "classification"

            slicer = config.get_as_slice(stratify_task, "ground_truth_column")
            labels = tf.to_int64(decoded[slicer])

            num_classes = config.getint(stratify_task, "num_classes")
            target_dist = [1.0 / num_classes] * num_classes

            decoded, _ = stratified_sample(decoded, labels, target_dist, batch_size=1, threads_per_queue=1)

    with tf.name_scope("shuffled_batching"):
        batches = tf.train.shuffle_batch(decoded,
                                         batch_size=batch_size,
                                         capacity=batch_size * 50,
                                         min_after_dequeue=batch_size * 10,
                                         num_threads=4)
        # batches[0] = tf.Print(batches[0], batches)
        all_cols = list(map(tf.squeeze, batches))
        return all_cols  # list of tensors, one tensor for each CSV column, each tensor with size batch_size


def stratified_sampling(batches, batch_size, config, task_name):
    assert task_name in config
    assert config.get(task_name, "type", fallback="") == "classification"

    slicer = config.get_as_slice(task_name, "ground_truth_column")
    labels = tf.to_int64(batches[slicer])

    num_classes = config.getint(task_name, "num_classes")
    target_dist = [1.0 / num_classes] * num_classes

    data_batch, _ = stratified_sample(batches, labels, target_dist, batch_size, enqueue_many=True, threads_per_queue=4)
    return data_batch
