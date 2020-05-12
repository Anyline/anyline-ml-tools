import tensorflow as tf
from datetime import datetime


class EpochTimeCallback(tf.keras.callbacks.Callback):
    """
    Keras callback, which saves execution time of each epoch

    """
    def __init__(self):
        self.timestamps = list()
        self.epoch_start_time = None

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = datetime.now()

    def on_epoch_end(self, epoch, logs={}):
        self.timestamps.append(datetime.now() - self.epoch_start_time)

    def avg_epoch(self):
        """Returns string with average epoch execution time"""
        if len(self.timestamps) == 0:
            return "unknown"
        total = self.timestamps[0]
        for i in range(1, len(self.timestamps)):
            total += self.timestamps[i]
        return EpochTimeCallback.format_timedelta(total / len(self.timestamps))

    def __str__(self):
        """Overall execution time"""
        if len(self.timestamps) == 0:
            return "unknown"
        total = self.timestamps[0]
        for i in range(1, len(self.timestamps)):
            total += self.timestamps[i]
        return EpochTimeCallback.format_timedelta(total)

    @staticmethod
    def format_timedelta(td):
        hours, remainder = divmod(td.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        hours, minutes, seconds = int(hours), int(minutes), int(seconds)
        if hours < 10:
            hours = '0%s' % int(hours)
        if minutes < 10:
            minutes = '0%s' % minutes
        if seconds < 10:
            seconds = '0%s' % seconds
        return '%sh %sm %ss' % (hours, minutes, seconds)