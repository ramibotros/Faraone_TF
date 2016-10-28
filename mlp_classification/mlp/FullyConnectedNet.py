from mlp_classification.mlp.ClassifyBaseModel import ClassifyBaseModel
import tensorflow as tf
from mlp_classification.mlp import utils

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
                    learn_type
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
        self.reg = config["reg"]
        self.num_hidden_units = config["num_hidden_units"]
        self.num_layers = config["num_layers"]
        self.learning_rate = config["learning_rate"]
        self.learn_type = config["learn_type"]


        self.batch_size = config["batch_size"]
        self.optimizer = config["optimizer"]
        self.keep_prob = config["keep_prob"]
        self.num_dimensions = config["num_dimensions"]

        self.add_placeholder()
        self.initialize_parameters()


    def bind_graph(self, input_features, input_labels, reuse=False):
        #Builds all ops that correspond to the NN graph and its evaluators and optimizers.
        #Needs the input data Tensors/Queues as argument
        #Any of the built ops, e.g. self.loss,
        # operates on the input data (given as arguments to this function) every time it is called.
        #If reuse=True , the TF graph is not built, but simply reused from the memory with the most recent parameters.

        if reuse:
            tf.get_variable_scope().reuse_variables()


        self.X = input_features
        self.y = utils.make_it_hot(input_labels, self.num_classes)

        self.yhat = self.calculate_scores()
        self.loss, self.loss_summary = self.add_loss()
        self.train_op = self.add_optimizer(type=self.learn_type)
        self.calculate_accuracy_op, self.accuracy_summary = self.calculate_accuracy()
        self.merged = self.add_summaries_operation()


    def add_placeholder(self):
        """
        Returns:
        """
        with tf.name_scope("Inputs"):

            self.keep_prob = tf.placeholder(dtype=tf.float32)

    def add_loss(self):
        """
        THE TRUE DISTRIBUTION WILL BE A PLACEHOLDER
        Args:
            yhat: prediction of shape (N, C)
                  N - number of training examples
                  C - Number of classes
        Returns: loss - scalar
        """
        # IN WHAT CASES WOULD tf.log() REACH A 0 VALUE
        # IF IT REACHES 0, IT WILL BE GIVE A NAN
        with tf.name_scope("loss") as scope:
            loss = tf.reduce_mean(-tf.reduce_sum(tf.to_float(self.y) * tf.log(self.yhat), reduction_indices=[1]))
            with tf.variable_scope("layers"):
                for i in xrange(1, self.num_layers):
                    with tf.variable_scope("layer_%s" % i, reuse=True):
                        loss = loss + self.reg * tf.nn.l2_loss(tf.get_variable("W"))

            loss_summary = tf.scalar_summary("loss_summary", loss)
            return loss, loss_summary

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
        return tf.merge_summary([self.loss_summary, self.accuracy_summary])

    def calculate_accuracy(self):

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.yhat, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100
            accuracy_summary = tf.scalar_summary("accuracy", accuracy)
            return accuracy, accuracy_summary

    def calculate_scores(self):
        """
        Return the scores of the model
        If it is convolutional neural networks or neural networks
        calculate the activations and final scores before the softmax loss is
        added
        """
        previous_out = self.X
        with tf.variable_scope("layers", reuse=True):
            for i in range(1, self.num_layers):
                with tf.variable_scope("layer_%s" % i, reuse=True):
                    weights = tf.get_variable("W")
                    biases = tf.get_variable("b")
                    previous_out = tf.nn.relu(tf.matmul(previous_out, weights) + biases)

            with tf.variable_scope("layer_%s" % (self.num_layers), reuse=True):
                weights = tf.get_variable("W")
                biases = tf.get_variable("b")
                scores = tf.nn.softmax(tf.matmul(previous_out, weights) + biases)

            return scores

    def initialize_parameters(self):
        """
        Initialize the parameters of the model
        Returns:
        """
        indices_list = []
        indices_list.append(self.num_dimensions)
        indices_list.extend([self.num_hidden_units] * (self.num_layers - 1))
        indices_list.append(self.num_classes)
        with tf.variable_scope("layers"):
            for i in range(1, self.num_layers + 1):
                with tf.variable_scope("layer_%s" % i) as scope:
                    W = tf.get_variable("W", shape=[indices_list[i - 1], indices_list[i]],
                                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.get_variable("b", shape=indices_list[i], initializer=tf.constant_initializer(0.0))
