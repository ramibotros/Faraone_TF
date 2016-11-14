import tensorflow as tf
from mlp_classification.mlp.FullyConnectedNet import FullyConnectedNet
import time
import threading




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
        self.keep_prob = config["keep_prob"]
        self.log_folder = config["log_folder"]
        self.validation_interval = config["validation_interval"]

        network = FullyConnectedNet(config)
        self.network = network

        network.bind_graph(train_features, train_labels, reuse=False, with_training_op=True)
        self.train_op = network.train_op
        self.train_loss = network.loss
        self.train_accuracy = network.calculate_accuracy_op


        # now reuse the graph to bind new OPs that handle the validation data:
        network.bind_graph(valid_features, valid_labels, reuse=True, with_training_op=False)
        self.valid_loss = network.loss
        self.valid_accuracy = network.calculate_accuracy_op

        self.train_summary_writer = tf.train.SummaryWriter(self.log_folder + "/train", self.session.graph)
        self.valid_summary_writer = tf.train.SummaryWriter(self.log_folder + "/valid", self.session.graph)

        self.summaries_merged = network.merged

        self.saver = tf.train.Saver(tf.all_variables())
        self.checkpoint_every = config["checkpoint_every"]
        self.checkpoint_path = config["checkpoint_folder"] + "/training.ckpt"




    def close_session(self):
        self.session.close()

    def test(self, test_features, test_labels):
        pass

    def train_once(self, i):
        _, train_loss, training_summary, training_accuracy = self.session.run([self.train_op, self.train_loss, self.summaries_merged, self.train_accuracy],
                                                               feed_dict={self.network.keep_prob: self.keep_prob, self.network.is_training:True})
        self.train_summary_writer.add_summary(training_summary, i)
        print("Training accuracy at the end of iteration %i:\t\t%f\t,\tloss:\t%f" % (i, training_accuracy, train_loss))

    def validate_once_and_sleep(self):
        while True:
            if not self.last_train_iteration:
                time.sleep(self.validation_interval)

            #self.saver.restore(self.session, self.newest_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            #global_step = int(self.newest_checkpoint_path.split('/')[-1].split('-')[-1])

            validation_summary, validation_accuracy, validation_loss = self.session.run([self.summaries_merged, self.valid_accuracy, self.valid_loss],
                                                                       feed_dict={self.network.keep_prob: 1, self.network.is_training:False})

            self.valid_summary_writer.add_summary(validation_summary, self.last_train_iteration)

            print("\n\n"+"*" * 80)
            print("Validation accuracy at the end of iteration %i:\t\t%f\tloss:\t%f" % (self.last_train_iteration, validation_accuracy, validation_loss))
            print("*" * 80 + "\n\n")

            time.sleep(self.validation_interval)


    def run_training(self):

        init_operation = tf.initialize_all_variables()
        self.session.run(init_operation)


        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=self.session, coord=coord)
        # start_queue_runners has to be called for any Tensorflow graph that uses queues.

        self.newest_checkpoint_path = ""
        self.last_train_iteration = 0
        valid_thread = threading.Thread(target=self.validate_once_and_sleep, args=())
        valid_thread.start()


        for i in range(1, self.num_epochs + 1):

            self.train_once(i)
            self.last_train_iteration = i

            if i % self.checkpoint_every == 0:
                self.newest_checkpoint_path = self.saver.save(self.session, self.checkpoint_path, i)
                print ("\nCheckpoint saved in %s\n" % self.newest_checkpoint_path)




