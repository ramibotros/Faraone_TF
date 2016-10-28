import tensorflow as tf

def make_it_hot(labels, num_classes):
    labels_fixed = tf.squeeze(tf.to_int64(labels))
    one_hot_labels = tf.one_hot(labels_fixed, num_classes, on_value=1, off_value=0)

    return one_hot_labels