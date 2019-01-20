import tensorflow as tf
import math
from process import get_data, convert_raw_to_wavs

if __name__ == '__main__':
    convert_raw_to_wavs()
    X, Y = get_data()

    N, D, D2 = X.shape
    half_D = math.floor(N/2)

    # split into test/train sets
    train_X = X[:half_D, :, :]
    train_Y = Y[:half_D, :]
    test_X = X[-half_D:, :, :]
    test_Y = Y[-half_D:, :]

    assert train_X.shape == test_X.shape
    assert train_Y.shape == test_Y.shape
