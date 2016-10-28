import tensorflow as tf
from mlp_classification.mlp.FullyConnectedNet import FullyConnectedNet

class FCNRunner:
    """
    This class acts as a factory and controller for FullyConnectedNet.py
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

    def __init__(self, train_features, train_labels, valid_features, valid_labels, config):
        """
        Args:
            train/valid_features/labels: Tensor objects, queues acceptable.
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

        self.session = tf.Session()
        self.num_classes = config["num_classes"]
        self.num_epochs = config["num_epochs"]

        self.log_folder = config["log_folder"]
        self.train_log_folder = config["train_log_folder"]
        self.test_log_folder = config["test_log_folder"]


        network = FullyConnectedNet(config)
        self.network = network

        network.bind_graph(train_features, train_labels, reuse=False)
        self.train_op = network.train_op
        self.train_loss = network.loss
        self.train_merged = network.merged
        self.train_accurancy = network.calculate_accuracy_op


        #now reuse the graph to bind new OPs that handle the validation data:
        network.bind_graph(valid_features, valid_labels, reuse=True)
        self.valid_loss = network.loss
        self.valid_merged = network.merged
        self.valid_accurancy = network.calculate_accuracy_op


        self.summary_writer = tf.train.SummaryWriter(self.log_folder)
        self.train_summary_writer = tf.train.SummaryWriter(self.train_log_folder)
        self.valid_summary_writer = tf.train.SummaryWriter(self.test_log_folder)

    def close_session(self):
        self.session.close()


    def test(self, test_features, test_labels):
        self.network.bind_graph(test_features, test_labels, reuse=True)
        return self.session.run(self.network.calculate_accuracy_op, feed_dict={self.network.keep_prob: 0.5})


    def train(self, print_every=1, log_summary=True):
        """
        Args:
            print_every: Print the loss and the epoch number every 10 iterations
        Returns:
            pass
        """
        init_operation = tf.initialize_all_variables()
        self.session.run(init_operation)

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=self.session, coord=coord)
        #start_queue_runners has to be called for any Tensorflow graph that uses queues.

        # 1. For every epoch
        #     USED TO BE : Pass once through all the training examples
        #     NOW : train on one batch.
        # 2. At the end of every epoch get the validation and train accuracy
        for i in range(self.num_epochs):
            if i == 0:
                print("*" * 80)
                loss = self.session.run(self.train_loss,
                                        feed_dict={self.network.keep_prob: 0.5})
                print("initial loss %f" % (loss,))
                print("*" * 80)

            _, loss_summary, loss = self.session.run([self.train_op, self.train_merged, self.train_loss],
                                                     feed_dict={self.network.keep_prob: 0.5})
            if i % print_every == 0 and log_summary == True:
                print("loss at iteration %d is %f" % (i, loss))

            self.summary_writer.add_summary(loss_summary, i)

            # PRINT THE TRAINING AND THE VALIDATION ACCURACY
            training_summary, training_accuracy = self.session.run([self.train_merged, self.train_accurancy],
                                                                   feed_dict={self.network.keep_prob: 0.5})

            validation_summary, validation_accuracy = self.session.run([self.valid_merged, self.valid_accurancy],
                                                                       feed_dict={self.network.keep_prob: 0.5})

            print("*" * 80)
            print("Training accuracy at the end of epoch %i: %f" % (i, training_accuracy))
            print("Validation accuracy at the end of epoch %i %f" % (i, validation_accuracy))
            print("*" * 80)

            self.train_summary_writer.add_summary(training_summary, i)
            self.valid_summary_writer.add_summary(validation_summary, i)

            # TODO: STORE THE BEST PARAMETERS

        return validation_accuracy

