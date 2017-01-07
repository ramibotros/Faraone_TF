import FCNRunner
import csvreader as csv
import os
import utils
import tensorflow as tf
from configreader import read_config

config = read_config("config/default.ini")


with tf.name_scope("train_data"):
    trows = csv.data_stream(config["PATHS"]["training_file"], int(config["TRAINING"]["batch_size"]))

with tf.name_scope("validation_data"):
    vrows = csv.data_stream(config["PATHS"]["validation_file"], int(config["TRAINING"]["validation_batch_size"]))


test_rows = None
if "TEST" in config:
    test_path = config["TEST"]["test_file"]
    test_rows = csv.data_stream(test_path, int(config["TEST"]["batch_size"]))



if not os.path.isdir(config["PATHS"]["checkpoint_dir"]):
    utils.mkdir_recursive(config["PATHS"]["checkpoint_dir"])




iris_runner = FCNRunner.FCNRunner(trows, vrows, test_rows, config)
iris_runner.run_training()
iris_runner.run_test()
