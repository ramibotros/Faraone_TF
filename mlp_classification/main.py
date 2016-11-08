from mlp_classification.mlp import FCNRunner
from mlp_classification.mlp import csvreader as csv
import os
import utils
import tensorflow as tf

rec_defaults = [([0.0]) for i in range(161)]  # This
training_file_name = "data/enigma_train.csv"
feature_list =   [("var%d" % i) for i in range(160)]
with tf.name_scope("train_data"):
    tfeatures, tlabels, tsize = csv.features_labels(100, training_file_name, rec_defaults, feature_list)
validation_file_name =  "data/enigma_validation.csv"
with tf.name_scope("validation_data"):
    vfeatures, vlabels, vsize = csv.features_labels(1744, validation_file_name, rec_defaults, feature_list)


config = {"l1_reg" : 0, #no regularization = 0;
          "l2_reg" : 0.8,  #no regularization = 0;
          "num_hidden_units": 80,
          "num_layers" : 3,
          "learning_rate" : 0.001,
          "log_folder" : "log/TF_logs",
          "checkpoint_folder" : "checkpoints/enigma",
          "num_epochs" : 1000,  #train on 1000 batches, then stop.
          "batch_size" : tsize,
          "optimizer" : "adam",
          "keep_prob" : 0.5,
          "num_classes": 2, #new: number of possible classes in classification
          "num_dimensions": 160, #new: number of feature dimensions
          "checkpoint_every": 10, # in number of iterations
          "validation_interval": 15, # in seconds
          }

if not os.path.isdir(config["checkpoint_folder"]):
    utils.mkdir_recursive(config["checkpoint_folder"])
#num_classes and num_dimnsions used to be inferred from the data. But the data is now coming into the model
#as a continuous stream, after the model graph is built.


iris_runner = FCNRunner.FCNRunner(tfeatures, tlabels, vfeatures, vlabels, config)
iris_runner.run_training()
