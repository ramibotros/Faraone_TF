from mlp_classification.mlp import FCNRunner
from mlp_classification.mlp import csvreader as csv

rec_defaults = [([0.0]) for i in range(161)]  # This
training_file_name = "data/enigma_train.csv"
feature_list =   [("var%d" % i) for i in range(160)]
tfeatures, tlabels, tsize = csv.features_labels(50, training_file_name, rec_defaults, feature_list)
validation_file_name =  "data/enigma_validation.csv"
vfeatures, vlabels, vsize = csv.features_labels(500, validation_file_name, rec_defaults, feature_list) # 50 might be
# too small for a validation batch. It is common to test validation scores on the complete validation data
# every time. So batch_size = complete set, i.e. no true "batching" of the data.
# Note : the Training accuracy and validation accuracy being printed while the training is running only express the
# accuracy on the *current batch* that the network was given.


config = {"reg" : 0, #L2 regularization = 1; no regularization = 0; try changing this last.
          "num_hidden_units": 100,
          "num_layers" : 5,
          "learning_rate" : .01,
          "learn_type" : "adam",
          "log_folder" : "log/TF_logs",
          "train_log_folder" : "log/TF_train_logs",
          "test_log_folder" : "log/TF_test_logs",
          "num_epochs" : 1000,  #train on 1000 batches, then stop.
          "batch_size" : tsize,
          "optimizer" : "adam",
          "keep_prob" : 0.05,
          "num_classes": 2, #new: number of possible classes in classification
          "num_dimensions": 160}  #new: number of feature dimensions
#num_classes and num_dimnsions used to be inferred from the data. But the data is now coming into the model
#as a continuous stream, after the model graph is built.


iris_runner = FCNRunner.FCNRunner(tfeatures, tlabels, vfeatures, vlabels, config)
iris_runner.train()
