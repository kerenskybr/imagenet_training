from os import path
import os

cwd = os.getcwd()
#Define the base path to the emotion dataset
BASE_PATH = cwd + "/datasets/fer2013"

#Use the base path to define the path to the input emotions file
INPUT_PATH = path.sep.join([BASE_PATH, "fer2013.csv"])

#Define the number of classes (set to 6 if you are ignoring the
#"disgust " class)
NUM_CLASSES = 6

#Define the path to the output training, validation
#and testing HDF5 files

TRAIN_HDF5 = path.sep.join([BASE_PATH, "hdf5/train.hdf5"])
VAL_HDF5 = path.sep.join([BASE_PATH, "hdf5/val.hdf5"])
TEST_HDF5 = path.sep.join([BASE_PATH, "hdf5/test.hdf5"])

#DEfine the batch size
BATCH_SIZE = 128

#DEfine the path to where output logs will be stored
OUTPUT_PATH = path.sep.join([BASE_PATH, "output"])