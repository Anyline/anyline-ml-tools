import os
import glob
import tensorflow as tf


def init_device():
    """Initialize GPUs"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print("Init GPUs:")
            print(" - Number of physical GPUs: %d" % len(gpus))
            print(" - Number of logical GPUs: %d" % len(logical_gpus))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def find_best_weights(path):
    """Find best weights in the folder. The filename pattern is "some-name-epoch_id-loss.format"

    Args:
        path: path to the folder

    Returns:
        filename with the smallest loss record

    """
    min_loss = float("inf")
    min_filename = None
    for filename in glob.glob(os.path.join(path, "*")):
        if (os.path.isdir(filename)):
            continue
        try:
            name, _ = os.path.splitext(os.path.basename(filename))
            _, _, iter, loss = name.split("-")
        except Exception as e:
            continue
        loss = float(loss)
        if loss < min_loss:
            min_loss = loss
            min_filename = filename
    return min_filename
