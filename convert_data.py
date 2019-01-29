import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pydub import AudioSegment, silence
import scipy.signal as signal

AUDIO_LENGTH = 1.2  # seconds
SPECTROGRAM_DATA_DIR_PATH = 'spec_data'
DATA_DIR_PATH = 'chord_data'
RAW_DATA_DIR_PATH = 'raw_data'
RAW_CONTINUOUS_DIR_PATH = 'continuous_raw'
CONTINUOUS_DIR_PATH = 'continuous_split'

EXEMPT_FILES = ['.DS_Store']

MIN_SILENCE_LEN = 1000
SILENCE_THRESH = -38


def split_silence(input_path: str = RAW_CONTINUOUS_DIR_PATH, output_dir: str = CONTINUOUS_DIR_PATH):
    """
    Takes all audio/video clips contained in 'input_path' & splits the clips into the loud spikes.
    This effectively converts the clip into many subclips of potential chord strums.

    If a file is contained in a sub-directory of 'input_path', it will be put into a respectively named file in the 'output_dir'.

    Arguments:
        input_path (str): The folder containing the audio slice(s).
        output_dir (str): The folder where the clips will be saved to.
    """

    if os.path.isdir(input_path):
        for fname in os.listdir(input_path):
            f_out_dir = output_dir

            if os.path.isdir(input_path + '/' + fname):
                f_out_dir += '/' + fname

            split_silence(input_path + '/' + fname,
                          f_out_dir)
        return

    if os.path.split(input_path)[1] not in EXEMPT_FILES:
        print('Splitting ' + input_path)
        sound = AudioSegment.from_file(input_path)
        chunks = silence.split_on_silence(
            sound, min_silence_len=MIN_SILENCE_LEN, silence_thresh=SILENCE_THRESH)

        os.makedirs(output_dir, exist_ok=True)

        used = 0

        for i, chunk in enumerate(chunks):
            # Too short
            if len(chunk) < AUDIO_LENGTH * 1000:
                continue

            name = os.path.splitext(input_path)[0].split('/')[-1]
            chunk.export(output_dir + '/' + name + '%i.wav' % i)
            used += 1

        print('Chunked: %i/%i\n' % (used, len(chunks)))


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
                        if file_path in EXEMPT_FILES and not os.path.isdir(file_path):
                            continue
                        file_path = jp(chord_and_quality_type_dir, file_path)
                        paths.append([chord_name, quality_type, file_path])
        elif chord_name == 'not_a_chord':
            for file_path in os.listdir(chord_path):
                if file_path in EXEMPT_FILES and not os.path.isdir(file_path):
                    continue
                file_path = jp(chord_and_quality_type_dir, file_path)
                paths.append([chord_name, 'none', file_path])

    return np.array(paths)


def spectrogramify(input_path: str = DATA_DIR_PATH, output_path: str = SPECTROGRAM_DATA_DIR_PATH, length_in_seconds: int = AUDIO_LENGTH, plot_image_count: int = 0, remove_silence=True):
    """
    Takes all the data from 'input_path' & converts it to a spectrogram image, dumping it to 'output_path'.

    Arguments:
        input_path (str): Input data directory.
        output_path (str): Output data directory. -- spectrogram images stored here.
        plot_image_count (int): Determines how many spec images will be plotted while saving.
    """

    len_milis = length_in_seconds * 1000
    paths = get_chord_data_paths(input_path)
    plotted = 0

    for chord_name, quality, path in paths:
        file_name = ''.join(os.path.basename(path).split('.')[:-1])

        path_to_file = '%s/%s/%s' % (
            output_path, chord_name, quality
        )

        os.makedirs(path_to_file, exist_ok=True)

        sound = AudioSegment.from_file(path)

        if remove_silence:
            sound = silence.split_on_silence(
                sound, min_silence_len=MIN_SILENCE_LEN, silence_thresh=SILENCE_THRESH)[0]

        if len(sound) < len_milis:
            print('\nClip too short.. (silence removal?: %b) min length: %.1fs: %s\n' %
                  (remove_silence, length_in_seconds, path))
            continue

        sound = sound[:int(len_milis)]  # trim to length
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

        save_to = path_to_file + '/%s.png' % file_name

        plt.savefig(save_to, pad_inches=0,
                    bbox_inches='tight', transparent=True)

        if plotted < plot_image_count:
            plt.show()
            plotted += 1

        print('Spectrogrammed: (%s, %s) -- from %s' %
              (quality, chord_name, path))


if __name__ == '__main__':
    # split_silence()
    spectrogramify(input_path=RAW_DATA_DIR_PATH)
    spectrogramify(input_path=CONTINUOUS_DIR_PATH, remove_silence=False)
