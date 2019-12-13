from config import emotion_config as config
from in_out.hdf5datasetwriter import HDF5DatasetWriter
import numpy as np

#Open the input file for reading (skipping header) then
#Initialize the list of data and labels for the training
#validation and testing sets
print("[INFO] loading input data...")
f = open(config.INPUT_PATH)
f.__next__()

(trainImages, trainLabels) = ([], [])
(valImages, valLabels) = ([], [])
(testImages, testLabels) = ([], [])

#Loop over the rows in the input file
for row in f:
	#Extract the label, image and usage from the row
	(label, image, usage) = row.strip().split(",")
	label = int(label)

	#If we are ignoring the "disgust" cllass, there will be 6
	#total class labels instead 7
	if config.NUM_CLASSES == 6:
		#Merge togheter the "anger" and "disgust" classes
		if label == 1:
			label = 0

		#If label has a value greater tha zer, subtract one from
		#it to make all labels sequential (not requiered, but helps
		#when interpreting results)
		if label > 0:
			label -= 1

	#Reshape the flattened pixel list into 48x48
	#(grayscale) image
	image = np.array(image.split(" "), dtype="uint8")
	image = image.reshape((48, 48))

	#Check if we are examining a training image
	if usage == "Training":
		trainImages.append(image)
		trainLabels.append(label)

	#Check if this is a validation image
	elif usage == "PrivateTest":
		valImages.append(image)
		valLabels.append(label)

	#Otherwise, this must be a testing image
	else:
		testImages.append(image)
		testLabels.append(label)

#Construct a list pairing the training, validation and testing
#iages along with their corresponding labels and output HD5 files
datasets = [
	(trainImages, trainLabels, config.TRAIN_HDF5),
	(valImages, valLabels, config.VAL_HDF5),
	(testImages, testLabels, config.TEST_HDF5)]

#Loop over the dataset tuples
for (images, labels, outputPath) in datasets:
	#Create HDF5 writer
	print("[INFO] building {}...".format(outputPath))
	writer = HDF5DatasetWriter((len(images), 48, 48), outputPath)

	#Loop over the image and add them to the dataset
	for (image, label) in zip(images, labels):
		writer.add([image], [label])

	#Close the HDF5 writer
	writer.close()

#Close the input file
f.close()