from mlp_classification.mlp.ClassifyBaseModel import ClassifyBaseModel
import tensorflow as tf
from mlp_classification.mlp import utils
from tensorflow.contrib.layers import fully_connected,  l1_l2_regularizer
from tensorflow.contrib.slim import batch_norm
from tensorflow.python.ops import control_flow_ops


class FullyConnectedNet(ClassifyBaseModel):
    """
        This employs a Softmax Classifier for multilabel classification
        We will employ a simple Softmax Model to classify the captions
    """

    def __init__(self, config):
        """
        Args:
            config: dictionary containing the hyperparameters
                    batch_size
                    learning_rate
                    optimiser is one of: ["vanilla", "adam", "adagrad", "rmsprop"]
                    log_folder - Folder into which the logs need to be written
                    num_epochs
                    num_hidden_units - The number of neurons in hidden layers
                    reg - regularization strength
                    test_log_folder   ----
                                          |--->  This is for storing logs  that needs to be separated into
                                          |--->  test and train. For eg. Training versus validation accuracies
                    train_log_folder  ----
        """
        super(FullyConnectedNet, self).__init__(config)
        self.l1_reg = [float(config["l1_reg"])]
        self.l2_reg = [float(config["l2_reg"])]
        self.num_hidden_units = config["num_hidden_units"]
        self.num_layers = int(config["num_layers"])
        self.learning_rate = config["learning_rate"]

        self.batch_size = config["batch_size"]
        self.optimizer = config["optimizer"]
        self.keep_prob = config["keep_prob"]
        self.num_dimensions = config["num_dimensions"]

        self.add_placeholder()

    def bind_graph(self, input_features, input_labels, reuse=False, with_training_op=False):
        # Builds all ops that correspond to the NN graph and its evaluators and optimizers.
        # Needs the input data Tensors/Queues as argument
        # Any of the built ops, e.g. self.loss,
        # operates on the input data (given as arguments to this function) every time it is called.
        # If reuse=True , the TF graph is not built, but simply reused from the memory with the most recent parameters.




        self.X = input_features
        self.labels = tf.to_int64(input_labels)

        with tf.variable_scope("network", reuse=reuse):

            self.Y_logits = self.make_FN_layers()

            if with_training_op:
                self.updates_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                self.loss = control_flow_ops.with_dependencies(tf.tuple(self.updates_op) ,self.add_loss())
                self.train_op = self.add_optimizer(type=self.optimizer)
            else:
                self.loss = self.add_loss()

        self.calculate_accuracy_op = self.calculate_accuracy()
        self.merged = self.add_summaries_operation()

    def add_placeholder(self):
        """
        Returns:
        """
        with tf.name_scope("Hypers"):
            self.keep_prob = tf.placeholder(dtype=tf.float32)
            self.is_training = tf.placeholder(dtype=tf.bool , shape=[])

    def add_loss(self):

        with tf.name_scope("loss") as scope:
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.Y_logits, self.labels))

            utils.variable_summaries(loss, "loss_summary")
            return loss

    def add_optimizer(self, type="vanilla"):
        """
        Add the optimizer function to perform Gradient Descent
        Args:
            type: The type of update that is needed
                  ["vanilla", "adam", "adagrad", "rmsprop"]
        Returns: None
        """
        if type not in ["vanilla", "adam", "adagrad", "rmsprop"]:
            raise ValueError("Please provide any of [vanilla, adam, adagrad, rmsprop] for optimisation")

        with tf.name_scope("gradient_descent"):
            train_op = None
            if type == "vanilla":
                train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
            elif type == "adam":
                train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            elif type == "adagrad":
                train_op = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
            elif type == "rmsprop":
                train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            return train_op

    def add_summaries_operation(self):
        return tf.merge_all_summaries()

    def calculate_accuracy(self):

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.Y_logits, 1), self.labels)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100
            utils.variable_summaries(accuracy, "accuracy_summary")
            return accuracy

    def make_FN_layers(self):

        previous_out = self.X

        with tf.variable_scope("layers"):
            for i in range(1, self.num_layers + 1):
                with tf.variable_scope("layer%d" % i) as layer_scope:
                    previous_out = fully_connected(previous_out, self.num_hidden_units, activation_fn=tf.nn.elu,
                                                   normalizer_fn=batch_norm,
                                                   normalizer_params={"scale": i == self.num_layers,
                                                                      "is_training": self.is_training,
                                                                      "decay" : 0.9},
                                                   weights_regularizer=l1_l2_regularizer(self.l1_reg, self.l2_reg),
                                                   scope=layer_scope)

                    if i == self.num_layers:
                        previous_out = tf.nn.dropout(previous_out, self.keep_prob)

            with tf.variable_scope("layer%d" % (self.num_layers + 1)) as layer_scope:
                previous_out = fully_connected(previous_out, self.num_classes, activation_fn=tf.identity,
                                               weights_regularizer=l1_l2_regularizer(self.l1_reg, self.l2_reg),
                                               scope=layer_scope)
        return previous_out
