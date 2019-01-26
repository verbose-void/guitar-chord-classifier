import numpy as np
from tensorflow import cast, float32
from librosa.feature import mfcc
from pydub import AudioSegment
from loading_utils import for_each_chord_file, DATA_DIR_PATH, RAW_DATA_DIR_PATH, convert_raw_to_wavs, AUDIO_LENGTH
import os
import math
import shutil

TRAIN_TO_TEST_RATIO = 0.8


chord_name_map = [
    'A',
    'B',
    'C',
    'D',
    'E',
    'F',
    'G'
]

quality_map = [
    'clear',
    'ringy',
    'muted'
]


def one_hot_encode(Y) -> np.array:
    """
    First index MUST be chord name.
    Second index MUST be quality.

    Returns a flattened NumPy array.
    """

    new_Y = []
    for y in Y:
        assert len(y) == 2, 'Y MUST have a dimensionality of 2.'
        inner_Y = []

        # Chord names
        for chord in chord_name_map:
            if y[0].lower() == chord.lower():
                inner_Y.append(1)
            else:
                inner_Y.append(0)

        # Qualities
        for quality in quality_map:
            if y[1].lower() == quality.lower():
                inner_Y.append(1)
            else:
                inner_Y.append(0)

        assert sum(inner_Y) == 2, \
            'Each output MUST Have 1 selection for each category.'
        new_Y.append(inner_Y)

    return np.array(new_Y)


class DataContainer:
    def __init__(self):
        self.data = []

    def append_data(self, chord_name, quality, path):
        input_format = path.split('.')[-1]
        sound = AudioSegment.from_file(path, format=input_format)

        assert len(sound) == AUDIO_LENGTH, \
            'Input sample %s MUST be %i miliseconds long.' % \
            (path, AUDIO_LENGTH)

        sound = np.array(sound.get_array_of_samples(), dtype='float32')
        mfcc_processed = mfcc(sound)
        # compensate for audio being a single channel
        mfcc_processed = np.expand_dims(mfcc_processed, axis=2)

        self.data.append([
            mfcc_processed,  # X
            (chord_name, quality)  # Y
        ])

    def get_data(self):
        data = np.array(self.data)
        X = np.stack(data[:, 0])
        # Normalize Xs
        X = X / X.sum(keepdims=1)
        Y = one_hot_encode(data[:, 1])
        assert len(X) == len(Y), 'EVERY input MUST have a resulting output.'
        return X, Y


def get_data():
    dc = DataContainer()
    for_each_chord_file(dc.append_data)
    return dc.get_data()


def get_train_and_test_data():
    """
    Breaks up the data into training & test sets.

    Returns:
        train_X: training inputs
        train_T: training labels
        test_X: testing inputs
        test_T: testing labels
    """

    X, T = get_data()

    N, D, D2, CHANNELS = X.shape
    train_D = math.floor(N * TRAIN_TO_TEST_RATIO)

    # split into test/train sets
    train_X = cast(X[:train_D, :, :], float32)
    train_T = cast(T[:train_D, :], float32)
    test_X = cast(X[train_D-D:, :, :], float32)
    test_T = cast(T[train_D-D:, :], float32)

    # make sure all dimensionalities are equal.
    assert train_X.get_shape()[1:] == test_X.get_shape()[1:]
    assert train_T.get_shape()[1:] == test_T.get_shape()[1:]

    return train_X, train_T, test_X, test_T


if __name__ == '__main__':
    convert_raw_to_wavs()
    # X, Y = get_data()
    # print('\n\nInputs:\n')
    # print(X)
    # print('\n\nOutputs:\n')
    # print(Y)
