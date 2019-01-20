import tensorflow as tf
import numpy as np
import librosa.feature as lf
from pydub import AudioSegment
import os
import shutil

DATA_DIR_PATH = 'chord_data'
RAW_DATA_DIR_PATH = 'raw_data'


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
    sound.export(export_path, format='wav')


def convert_raw_to_wavs():
    """
    Takes all files from 'RAW_DATA_DIR_PATH' and moves them to 'DATA_DIR_PATH', also
    converting them to the proper file type.
    """

    if not os.path.isdir(RAW_DATA_DIR_PATH):
        return print('No data to process, skipping...')

    for_each_chord_file(convert_to_wav_and_move, parent_path=RAW_DATA_DIR_PATH)
    shutil.rmtree(RAW_DATA_DIR_PATH)


class DataContainer:
    def __init__(self):
        self.data = []

    def append_data(self, chord_name, quality, path):
        input_format = path.split('.')[-1]
        sound = AudioSegment.from_file(path, format=input_format)
        sound = np.array(sound.get_array_of_samples(), dtype='float32')
        mfcc = lf.mfcc(sound)

        self.data.append([
            mfcc,  # X
            (chord_name, quality)  # Y
        ])

    def get_data(self):
        """
        TODO: Encode qualities & chord names etc.
        """

        data = np.array(self.data)
        return data[:, 0], data[:, 1]


def get_data():
    dc = DataContainer()
    for_each_chord_file(dc.append_data)
    return dc.get_data()


if __name__ == '__main__':
    convert_raw_to_wavs()
    X, Y = get_data()
    print('\n\nInputs:\n')
    print(X)
    print('\n\nOutputs:\n')
    print(Y)
