import time

from keras.callbacks import Callback


class TimeOut(Callback):
    def __init__(self, t0, timeout):
        super().__init__()
        self.t0 = t0
        self.timeout = timeout  # time in minutes

    def on_train_batch_end(self, batch, logs=None):
        if time.time() - self.t0 > self.timeout * 60:  # 58 minutes
            print(f"\nReached {(time.time() - self.t0) / 60:.3f} minutes of training, stopping")
            self.model.stop_training = True


