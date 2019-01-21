import tensorflow as tf
import process
import model
import time
import datetime


LOG_FREQUENCY = 1
BATCH_SIZE = 5
TRAIN_DIRECTORY = '/tmp/train'
MAX_STEPS = 100000
LOG_DEVICE_PLACEMENT = False


def train():
    """Train for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
            train_X, train_T, test_X, test_T = process.get_train_and_test_data()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = model.inference(train_X)

        # Calculate loss.
        loss = model.loss(logits, train_T)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = model.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % LOG_FREQUENCY == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = LOG_FREQUENCY * BATCH_SIZE / duration
                    sec_per_batch = float(duration / LOG_FREQUENCY)

                    format_str = (
                        '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                    )

                    print(format_str % (datetime.datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=TRAIN_DIRECTORY,
            hooks=[
                tf.train.StopAtStepHook(last_step=MAX_STEPS),
                tf.train.NanTensorHook(loss),
                _LoggerHook()
            ],
            config=tf.ConfigProto(
                log_device_placement=LOG_DEVICE_PLACEMENT
            )
        ) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
    process.convert_raw_to_wavs()

    if tf.gfile.Exists(TRAIN_DIRECTORY):
        tf.gfile.DeleteRecursively(TRAIN_DIRECTORY)
    tf.gfile.MakeDirs(TRAIN_DIRECTORY)

    train()


if __name__ == '__main__':
    tf.app.run()
