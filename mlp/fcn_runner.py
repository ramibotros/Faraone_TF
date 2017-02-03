import os
import threading

import numpy as np
import tensorflow as tf

import utils
from mlp.fcn import FCN


class FCNRunner:
    """
    This class acts as a factory and controller for fcn.py
    FullyConnectedNet builds a tensorflow graph that represents the NN and its evaluation ops.
    FCNRunner uses the FullyConnectedNet graph to build two other graphs: one for training and one for validation.
    A good thing is that both the training and testing graphs share the same variables (https://www.tensorflow.org/versions/r0.11/how_tos/variable_scope/index.html)
    So there is no memory duplication of parameters, or duplication of the process of building up the NN twice.
    +----------------------------------------------------------------------------+
    | training                                                                   |
    | data                                                                       |
    | pipeline                                                                   |
    |    +      +----------+                                                     |
    |    +----> | Fully    +-------> train_loss, train_accuracy, optimization_op |
    |           | Connected|                                                     |
    |    +----> | Net      +-------> validation_loss, validation_accuracy        |
    |    +      +----------+                                                     |
    | validation                                                                 |
    | data                                                                       |
    | pipeline                                                                   |
    +----------------------------------------------------------------------------+
    The training output ops (train_loss, etc...) are only concerned with applying the FCN to the training data.
    The validation output ops (validation_loss, etc...) are only concerned with applying the FCN to the validation data.
    """

    def __init__(self, config):

        self.config = config

        # config:
        self.log_folder = config.get("PATHS", "log_folder")
        self.experiment_ID = config.get("PROCESS", "experiment_ID") or utils.date_time_string()
        self.validation_interval = config.getint("PROCESS", "validation_interval", fallback=15)
        self.keep_prob = config.getfloat("TRAINING", "dropout_keep_probability", fallback=1.0)
        self.num_epochs = config.getint("TRAINING", "num_epochs", fallback=0)

        self.network = FCN(config)

    def bind_training_dataqueue(self, train_data_cols):
        config = self.config

        train_batch_size = config.getint("TRAINING", "batch_size")
        with tf.name_scope("Train"):
            self.network.bind_graph("TRAIN", train_data_cols, train_batch_size, reuse=False, with_training_op=True)
        self.train_op = self.network.train_op
        self.train_loss = self.network.loss
        self.train_str_accu = self.network.streaming_accu_op
        self.train_accuracy = self.network.accuracy

        self.train_summaries_merged = self.network.get_summaries("TRAIN")

    def bind_validation_dataqueue(self, valid_data_cols):
        config = self.config

        # now reuse the graph to bind new OPs that handle the validation data:
        valid_batch_size = config.getint("TRAINING", "validation_batch_size")
        with tf.name_scope("Valid"):
            self.network.bind_graph("VALID", valid_data_cols, valid_batch_size, reuse=True, with_training_op=False)
        self.valid_loss = self.network.loss
        self.valid_str_accu = self.network.streaming_accu_op
        self.valid_accuracy = self.network.accuracy

        self.valid_summaries_merged = self.network.get_summaries("VALID")

    def bind_test_dataqueue(self, test_data_cols):
        config = self.config

        # now resuse the graph to bind new OPS that handle the test data:
        test_batch_size = config.getint("TEST", "batch_size")
        with tf.name_scope("Test"):
            self.network.bind_graph("TEST", test_data_cols, test_batch_size, reuse=True, with_training_op=False)
        self.test_loss = self.network.loss
        self.test_str_accu = self.network.streaming_accu_op
        self.test_accuracy = self.network.accuracy
        self.test_summaries_merged = self.network.get_summaries("TEST")
        self.test_predictions = self.network.predictions
        self.test_pred_path = config["TEST"]["write_predictions_to"]

    def initialize(self):
        config = self.config
        self.session = tf.Session()

        self.saver = tf.train.Saver(tf.global_variables())
        self.checkpoint_every = config.getint("PROCESS", "checkpoint_every")
        self.checkpoint_path = config.get("PATHS", "checkpoint_dir") + "/training.ckpt"

        load_checkpoint = config.get("PROCESS", "initialize_with_checkpoint") or None
        if load_checkpoint:
            self.load_checkpoint(load_checkpoint)

        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())  # for streaming metrics

        self.create_summary_writers()

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=self.session, coord=coord)
        # start_queue_runners has to be called for any Tensorflow graph that uses queues.


        tensorboard_thread = threading.Thread(target=self.start_tensorboard, args=())
        tensorboard_thread.start()

    def create_summary_writers(self):

        if hasattr(self, "train_summaries_merged"):
            self.train_summary_writer = tf.summary.FileWriter("%s/%s_train" % (self.log_folder, self.experiment_ID),
                                                              self.session.graph)

        if hasattr(self, "valid_summaries_merged"):
            self.valid_summary_writer = tf.summary.FileWriter("%s/%s_valid" % (self.log_folder, self.experiment_ID))

        if hasattr(self, "test_summaries_merged"):
            self.test_summary_writer = tf.summary.FileWriter("%s/%s_test" % (self.log_folder, self.experiment_ID))

    def close_session(self):
        self.session.close()

    def test(self, test_features, test_labels):
        pass

    def train_once(self, i):
        _, train_loss, training_summary, training_accuracy, train_streaming_accuracy = self.session.run(
            [self.train_op, self.train_loss, self.train_summaries_merged, self.train_accuracy, self.train_str_accu],
            feed_dict={self.network.keep_prob: self.keep_prob, self.network.is_training: True})

        self.train_summary_writer.add_summary(training_summary, i)

        print("Training at the end of iteration %i:\tAccuracy:\t%f\tStreaming Accu:\t%f\tloss:\t%f" % (
            i, training_accuracy, train_streaming_accuracy, train_loss))
        self.train_summary_writer.flush()

    def load_checkpoint(self, path):
        self.saver.restore(self.session, path)
        print("Checkpoint loaded from %s" % path)

    def validate_once(self, i):
        validation_summary, validation_accuracy, validation_streaming_accuracy, validation_loss = self.session.run(
            [self.valid_summaries_merged, self.valid_accuracy, self.valid_str_accu, self.valid_loss],
            feed_dict={self.network.keep_prob: 1, self.network.is_training: False})

        self.valid_summary_writer.add_summary(validation_summary, i)

        print("\n\n" + "*" * 80)
        print("Validation after iteration %i:\tAccuracy:\t%f\tStreaming Accu:\t%f\tloss:\t%f" % (
            i, validation_accuracy, validation_streaming_accuracy, validation_loss))
        print("*" * 80 + "\n\n")
        self.valid_summary_writer.flush()

    def test_once(self):
        test_summary, test_loss, test_predictions, test_accuracy = self.session.run(
            [self.test_summaries_merged, self.test_loss, self.test_predictions, self.test_accuracy],
            feed_dict={self.network.keep_prob: 1, self.network.is_training: False})

        self.test_summary_writer.add_summary(test_summary, 1)

        print("\n\n" + "*" * 80)
        print("Test accuracy at the end:\t%f\tloss:\t%f" % (
            test_accuracy, test_loss))
        print("*" * 80 + "\n\n")
        self.test_summary_writer.flush()

        np.savetxt(self.test_pred_path, test_predictions, '%.7f')
        print("Test predictions/scores saved in %s " % self.test_pred_path)

    def start_tensorboard(self):
        log_dir_abs_path = os.path.abspath(self.log_folder)
        print("tensorboard --logdir=%s\n" % (log_dir_abs_path))
        # Popen(["tensorboard", "--logdir=%s" %(log_dir_abs_path)])
        # print("\n")

        utils.background_process(["tensorboard", "--logdir=%s" % (log_dir_abs_path)])

    def run_training(self):

        self.newest_checkpoint_path = ""
        self.last_train_iteration = 0

        print("\n")
        for i in range(1, self.num_epochs + 1):

            self.train_once(i)
            self.last_train_iteration = i

            if i % self.checkpoint_every == 0:
                self.newest_checkpoint_path = self.saver.save(self.session, self.checkpoint_path, i)
                print("\nCheckpoint saved in %s\n" % self.newest_checkpoint_path)

            if i % self.validation_interval == 0:
                self.validate_once(i)

    def run_test(self):
        print("TESTING")
        self.test_once()
