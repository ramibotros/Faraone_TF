import os

import tensorflow as tf

import config_reader
import csv_reader
import mlp
import utils

config = config_reader.read_config("config/default.ini")

if not os.path.isdir(config.get("PATHS", "checkpoint_dir")):
    utils.mkdir_recursive(config.get("PATHS", "checkpoint_dir"))

iris_runner = mlp.FCNRunner(config)  # trows, vrows, test_rows, config)
if "TRAINING" in config:
    with tf.name_scope("train_data"):
        train_batch_size = config.getint("TRAINING", "batch_size")
        stratified_task = config.get("TRAINING", "stratified_sampling", fallback="")
        trows = csv_reader.read_csv(config.get("PATHS", "training_file"), train_batch_size, stratified_task, config)

    with tf.name_scope("validation_data"):
        vrows = csv_reader.read_csv(config.get("PATHS", "validation_file"),
                                    config.getint("TRAINING", "validation_batch_size"))

    iris_runner.bind_training_dataqueue(trows)
    iris_runner.bind_validation_dataqueue(vrows)

if "TEST" in config:
    test_path = config["TEST"]["test_file"]
    with tf.name_scope("test_data"):
        test_rows = csv_reader.read_csv(test_path, int(config["TEST"]["batch_size"]))
    iris_runner.bind_test_dataqueue(test_rows)

iris_runner.initialize()

if "TRAINING" in config:
    iris_runner.run_training()
if "TEST" in config:
    iris_runner.run_test()
