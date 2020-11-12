#Code for training LSTM-AE Model

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import keras
import tensorflow
from tensorflow.keras.layers import Conv2DTranspose, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LayerNormalization
from os import listdir
from os.path import isfile, join, isdir
import numpy as np
from PIL import Image
from model import *

class Config:
  DATASET_PATH ="/kaggle/input/ucsddataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
  TEST_PATH = "/kaggle/input/ucsddataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test"
  BATCH_SIZE = 4
  EPOCHS = 20
  MODEL_PATH = "/kaggle/working/model.hdf5"
  
  mod = get_model(True)
  
  mod.fit(get_training_set(), get_training_set(),batch_size=Config.BATCH_SIZE, epochs=Config.EPOCHS, shuffle=False)
  
  mod.save(Config.MODEL_PATH)
  
  
  