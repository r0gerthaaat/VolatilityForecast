import data_preprocessor as dp
import dataset as ds

import statsmodels.api as sm
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader

from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_absolute_percentage_error

def main():
    # dp.LOOKBACK_WINDOW = 22
    #
    # data = dp.preprocess_data()
    #
    # xs, ys = ds.create_sequences(data, dp.LOOKBACK_WINDOW)
    # print(data)
    # print(xs.shape)
    # print(ys.shape)


if __name__ == '__main__':
    main()