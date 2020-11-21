#Testing code for LSTM-AE

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import keras
import tensorflow

import numpy as np
from PIL import Image

from model import get_single_test, mse, get_model

class Config:
  DATASET_PATH ="/kaggle/input/ucsddataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
  SINGLE_TEST_PATH = "/kaggle/input/ucsddataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test001"
  BATCH_SIZE = 4
  EPOCHS = 20
  MODEL_PATH = "/kaggle/working/model.hdf5"

mod = get_model(True)
test = get_single_test()
sz = test.shape[0]
sequences = np.zeros((sz, 5, 256, 256, 1))
# apply the sliding window technique to get the sequences
for i in range(0, sz-5):
    clip = np.zeros((5, 256, 256, 1))
    for j in range(0, 5):
        clip[j] = test[i + j, :, :, :]
    sequences[i] = clip

print("got data")
# get the reconstruction cost of all the sequences
reconstructed_sequences = mod.predict(sequences,batch_size=4)

sequences_reconstruction_cost = np.array([np.maximum(mse(sequences[i],reconstructed_sequences[i])) for i in range(0,sz-5)])
#sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / np.max(sequences_reconstruction_cost)
np.savetxt('test-scores.csv',sa,delimiter = ',')
# plot the regularity scores
axes = plt.gca()
plt.plot(sa)
plt.ylabel('loss')
plt.xlabel('frame t')
plt.show()

