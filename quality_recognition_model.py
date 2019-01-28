from convert_data import SPECTROGRAM_DATA_DIR_PATH, get_chord_data_paths
from fastai.vision import *
from fastai.metrics import error_rate
import os
import atexit


MODEL_NAME = 'model-1'


def get_learner(bs=5):
    """
    Returns the quality recognition model
    """

    # Load data
    data = get_chord_data_paths(parent_path=SPECTROGRAM_DATA_DIR_PATH)
    data = ImageDataBunch.from_lists(
        path=SPECTROGRAM_DATA_DIR_PATH,
        fnames=data[:, 2], labels=data[:, 1],
        bs=bs
    ).normalize(imagenet_stats)

    print('Data loaded, available classes: ', data.classes)

    learn = create_cnn(data, models.resnet34, metrics=error_rate)

    # load if model exists
    if os.path.isfile(SPECTROGRAM_DATA_DIR_PATH + '/models/%s.pth' % MODEL_NAME):
        learn.load(MODEL_NAME)

    return learn


if __name__ == '__main__':
    learn = get_learner()
    learn.fit_one_cycle(50)

    interp = ClassificationInterpretation.from_learner(learn)
    print(interp.most_confused())

    atexit.register(lambda: learn.save(MODEL_NAME))
