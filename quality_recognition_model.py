from convert_data import SPECTROGRAM_DATA_DIR_PATH, get_chord_data_paths
from fastai.vision import *
from fastai.metrics import error_rate
import os

MODEL_NAME = 'model-1'

bs = 5
epochs = 50

# Load data
data = np.array(get_chord_data_paths(parent_path=SPECTROGRAM_DATA_DIR_PATH))
data = ImageDataBunch.from_lists(
    path=SPECTROGRAM_DATA_DIR_PATH,
    fnames=data[:, 2], labels=data[:, 1],
    bs=bs
).normalize(imagenet_stats)

learn = create_cnn(data, models.resnet34, metrics=error_rate)

# load if model exists
if os.path.isfile(SPECTROGRAM_DATA_DIR_PATH + '/models/%s.pth' % MODEL_NAME):
    learn.load(MODEL_NAME)

learn.fit_one_cycle(epochs)

interp = ClassificationInterpretation.from_learner(learn)
print(interp.most_confused())

learn.save(MODEL_NAME)
