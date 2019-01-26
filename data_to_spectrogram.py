import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

from loading_utils import for_each_chord_file, RAW_DATA_DIR_PATH, convert_raw_to_wavs

SPECTROGRAM_DATA_PATH = 'chord_data-spectrogram'


def convert_wav_to_spectrogram(name: str, quality: str, path: str):
    sample_rate, samples = wavfile.read(path)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def convert_wavs_to_spectrograms(convert_raw: bool = False):
    """
    Gets all .wav files from the 'DATA_DIR_PATH' and converts them
    into spectrogram image files in the directory 'SPECTROGRAM_DATA_PATH'.

    Arguments:
        convert_raw (bool): Flags if the raw data should be converted before this process.
    """

    if convert_raw:
        convert_raw_to_wavs()

    for_each_chord_file(lambda name, quality,
                        path: convert_wav_to_spectrogram(name, quality, path))


if __name__ == '__main__':
