from quality_recognition_model import get_learner, SPECTROGRAM_DATA_DIR_PATH, MODEL_NAME
from convert_data import get_chord_data_paths
import numpy as np
import os
from fastai.vision import *

if __name__ == '__main__':
    bd = SPECTROGRAM_DATA_DIR_PATH

    img = open_image('spec_data/Dm/clear/Dm0.png')
    classes = ['A', 'Am', 'C', 'Cadd9', 'D', 'Dm', 'E', 'Em', 'G']
    data = ImageDataBunch.single_from_classes(bd, classes)
    learn = create_cnn(data, models.resnet34)
    learn.load(MODEL_NAME)

    print('Making prediction...')
    y, idx, outputs = learn.predict(img)
    print('\nPrediction: %s \t Actual: %s' % (str(y), 'Dm'))
    st = ''
    for i, output in enumerate(outputs):
        st += classes[i] + ': %.1f' % (output * 100) + '%\n'
    print('Confidences:')
    print(st)
