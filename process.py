import numpy as np
import tensorflow as tf
import librosa.feature as lf
from pydub import AudioSegment
import os
import math
import shutil

DATA_DIR_PATH = 'chord_data'
RAW_DATA_DIR_PATH = 'raw_data'
AUDIO_LENGTH = 2 * 1000
TRAIN_TO_TEST_RATIO = 0.8


def for_each_chord_file(function, parent_path=DATA_DIR_PATH):
    """
    For each file in the 'DATA_DIR_PATH' subdirectories, call the function arg.

    Args:
        function (function): this function is called at each .wav file.
            arg1: chord_name
            arg2: quality
            arg3: path_to_file

        parent_path (str): the parent path for the data subdirectories.
    """

    jp = os.path.join

    # Loop through each chord type folder
    for chord_name in os.listdir(parent_path):
        chord_path = jp(parent_path, chord_name)

        if os.path.isdir(chord_path):

            # Loop through each quality type folder
            for quality_type in os.listdir(chord_path):

                chord_and_quality_type_dir = jp(chord_path, quality_type)
                if os.path.isdir(chord_and_quality_type_dir):

                    for file_path in os.listdir(chord_and_quality_type_dir):
                        if file_path == '.DS_Store' and not os.path.isdir(file_path):
                            continue

                        file_path = jp(chord_and_quality_type_dir, file_path)
                        function(chord_name, quality_type, file_path)


def convert_to_wav_and_move(chord_name, quality, path):
    """
    Converts the given file from it's current file type (whatever it may be)
    into a .wav file, and copies it from the RAW_DATA_DIR_PATH path to the
    DATA_DIR_PATH path.
    """

    input_format = path.split('.')[-1]
    export_path = DATA_DIR_PATH + '/'
    export_path += '/'.join(path.split('/')[1:])
    dir_name = os.path.split(export_path)[0]

    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    sound = AudioSegment.from_file(path, format=input_format)
    # truncate to be AUDIO_LENGTH miliseconds long
    sound = sound[:AUDIO_LENGTH]
    sound.export(export_path, format='wav')


def convert_raw_to_wavs():
    """
    Takes all files from 'RAW_DATA_DIR_PATH' and moves them to 'DATA_DIR_PATH', also
    converting them to the proper file type.
    """
    if not os.path.isdir(RAW_DATA_DIR_PATH):
        return print('No data to process, skipping...')

    convert = input(
        'Would you like to convert the data inside raw_data? (y, N): ')

    if convert.lower() == 'n':
        return print('Skipping...')

    print('Converting raw files to real data...')

    for_each_chord_file(convert_to_wav_and_move, parent_path=RAW_DATA_DIR_PATH)
    print('Conversion complete!')

    delete = input('Would you like to delete all raw data? (y, N): ')
    if delete.lower() == 'y':
        shutil.rmtree(RAW_DATA_DIR_PATH)


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
        mfcc = lf.mfcc(sound)
        # compensate for audio being a single channel
        mfcc = np.expand_dims(mfcc, axis=2)

        self.data.append([
            mfcc,  # X
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
    train_X = tf.cast(X[:train_D, :, :], tf.float32)
    train_T = tf.cast(T[:train_D, :], tf.float32)
    test_X = tf.cast(X[train_D-D:, :, :], tf.float32)
    test_T = tf.cast(T[train_D-D:, :], tf.float32)

    # make sure all dimensionalities are equal.
    assert train_X.get_shape()[1:] == test_X.get_shape()[1:]
    assert train_T.get_shape()[1:] == test_T.get_shape()[1:]

    return train_X, train_T, test_X, test_T


if __name__ == '__main__':
    convert_raw_to_wavs()
    X, Y = get_data()
    print('\n\nInputs:\n')
    print(X)
    print('\n\nOutputs:\n')
    print(Y)
