import tensorflow as tf

class ClassifyBaseModel(object):
    """
        This models the base class that is required for
        all the classifications in Tensor flow
        Tensorflow classifications model
        Inputs as placeholders
        Output labels as placeholder
    """

    def __init__(self, config):
        self.config = config
        self.num_classes = config["num_classes"]

    def add_loss(self):
        """
        THE TRUE DISTRIBUTION WILL BE A PLACEHOLDER
        Returns: loss - scalar
        """
        pass

    def add_summaries_operation(self):
        pass

    def add_placeholder(self):
        """
        Add all the placeholder that are required for the model here
        Returns:
        """
        pass

    def calculate_accuracy(self):
        """
        Add the operation to calculate the accuracy for your model
        Returns:
        """
        pass

    def calculate_scores(self):
        """
        Return the scores of the model
        If it is convolutional neural networks or neural networks
        calculate the activations and final scores before the softmax loss is
        added
        """
        pass


    def get_placeholder(self, size, dtype=tf.float32):
        """
        SHOULD NOT BE CHANGED IN THE INHERITED CLASS
        Args:
            size: The size of the placeholder to be created
            dtype: Data type for the placeholder
        Returns: tf.placeholder
        """
        return tf.placeholder(dtype, size)

    def initialize_parameters(self):
        """
        Initialize the parameters of the model
        Returns:
        """
        pass



