import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pydub import AudioSegment
import scipy.signal as signal

AUDIO_LENGTH = 2  # seconds
SPECTROGRAM_DATA_DIR_PATH = 'spec_data'
DATA_DIR_PATH = 'chord_data'
RAW_DATA_DIR_PATH = 'raw_data'


def get_chord_data_paths(parent_path=DATA_DIR_PATH):
    """
    Args:
        parent_path (str): the parent path for the data subdirectories.

    Returns:
        None if directory doesn't exist, or...

        A list of triples containing respectively:
            chord_name: The chord's name.
            quality_type: The quality of the strum.
            file_path:  Path to the sound file.
    """
    if not os.path.isdir(parent_path):
        return None

    paths = []
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
                        paths.append([chord_name, quality_type, file_path])
        elif chord_name == 'not_a_chord':
            for file_path in os.listdir(chord_path):
                if file_path == '.DS_Store' and not os.path.isdir(file_path):
                    continue
                file_path = jp(chord_and_quality_type_dir, file_path)
                paths.append([chord_name, 'none', file_path])

    return np.array(paths)


def spectrogramify(input_path: str = DATA_DIR_PATH, output_path: str = SPECTROGRAM_DATA_DIR_PATH, length_in_seconds: int = AUDIO_LENGTH, plot_image_count: int = 0):
    """
    Takes all the data from 'input_path' & converts it to a spectrogram image, dumping it to 'output_path'.

    Arguments:
        input_path (str): Input data directory.
        output_path (str): Output data directory. -- spectrogram images stored here.
        plot_image_count (int): Determines how many spec images will be plotted while saving.
    """

    paths = get_chord_data_paths(RAW_DATA_DIR_PATH)
    plotted = 0

    for chord_name, quality, path in paths:
        file_name = ''.join(os.path.basename(path).split('.')[:-1])

        path_to_file = '%s/%s/%s' % (
            output_path, chord_name, quality
        )

        os.makedirs(path_to_file, exist_ok=True)

        sound = AudioSegment.from_file(path)
        sound = sound[:length_in_seconds * 1000]  # trim to length
        sound = sound.set_channels(1)  # convert to single channel

        # get array & convert to freq, times, & amplitudes
        array = np.array(sound.get_array_of_samples())
        freqs, times, amplitudes = signal.spectrogram(
            array, sound.frame_rate, scaling='spectrum')

        # amplify amplitudes
        amplitudes = 10 * np.log10(amplitudes + 1e-9)

        plt.pcolormesh(times, freqs, amplitudes)

        plt.axis('off')
        ax = plt.gca()
        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        plt.savefig(
            path_to_file + '/%s.png' % file_name,
            pad_inches=0, bbox_inches='tight', transparent=True
        )

        if plotted < plot_image_count:
            print('plotting a %s %s' % (quality, chord_name))
            plt.show()
            plotted += 1


if __name__ == '__main__':
    spectrogramify(input_path=RAW_DATA_DIR_PATH, plot_image_count=5)
