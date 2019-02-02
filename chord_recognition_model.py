from convert_data import SPECTROGRAM_DATA_DIR_PATH, get_chord_data_paths
from fastai.vision import *
from fastai.metrics import error_rate
import os
import atexit
import numpy as np

MODEL_NAME = 'model-1'


def get_learner(bs=5, seed=None):
    """
    Returns the quality recognition model
    """

    if seed is not None:
        np.random.seed(seed)

    # Load data
    # data = get_chord_data_paths(
    #     parent_path=SPECTROGRAM_DATA_DIR_PATH, exempt_qualities=['ringy'])
    # data = ImageDataBunch.from_lists(
    #     path=SPECTROGRAM_DATA_DIR_PATH,
    #     fnames=data[:, 2], labels=data[:, 0],
    #     bs=bs
    # ).normalize(imagenet_stats)
    data = ImageDataBunch.from_folder('cp_spec_data')
    print('Data loaded, available classes: ', data.classes)

    learn = create_cnn(data, models.resnet34, metrics=(error_rate, accuracy))
    # load if model exists
    # if os.path.isfile(SPECTROGRAM_DATA_DIR_PATH + '/models/%s.pth' % MODEL_NAME):
    #     learn.load(MODEL_NAME)

    return learn


if __name__ == '__main__':
    learn = get_learner()
    atexit.register(lambda: learn.save(MODEL_NAME))
    learn.fit_one_cycle(5)

    # interp = ClassificationInterpretation.from_learner(learn)
    # print(interp)
