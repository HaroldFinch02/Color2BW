import tensorflow as tf
import numpy as np
import cv2
import tflearn

num_images = 1530

dataset_source = []
# Train Data
for i in range(1, num_images+1):
    img = cv2.imread(
        "data/train_data/color_images/color_" + str(i) + ".jpg")
    dataset_source.append(np.array(img))

dataset_source = np.asarray(dataset_source)

dataset_tar = []
for i in range(1, num_images+1):
    img = cv2.imread(
        "data/train_data/gray_images/gray_" + str(i) + ".jpg", 0)
    dataset_tar.append(np.array(img))
dataset_target = np.asarray(dataset_tar)

dataset_target = dataset_target[:, :, :, np.newaxis]


testSource = []
# Test Data
num_images = 725
for i in range(1, num_images+1):
    img = cv2.imread(
        "data/testData/test_color_images/color_" + str(i) + ".jpg")
    testSource.append(np.array(img))

testSource = np.asarray(testSource)

testTarget = []
for i in range(1, num_images+1):
    img = cv2.imread(
        "data/testData/test_gray_images/gray_" + str(i) + ".jpg", 0)
    testTarget.append(np.array(img))
testTarget = np.asarray(dataset_tar)

testTarget = testTarget[:, :, :, np.newaxis]


# Encoder
encoder = tflearn.input_data(shape=[None, 128, 128, 3])
encoder = tflearn.conv_2d(encoder, 64, 3, activation='relu')
encoder = tflearn.conv_2d(encoder, 32, 3, activation='relu')

# Decoder
decoder = tflearn.conv_2d(encoder, 16, 3, activation='relu')
decoder = tflearn.conv_2d(decoder, 1, 3, activation='relu')

net = tflearn.regression(decoder, optimizer='adam', learning_rate=0.001, loss='mean_square')


model = tflearn.DNN(net, best_checkpoint_path='SavedModel/bm')
model.fit(dataset_source, dataset_target, validation_set=(testSource, testTarget), batch_size=32, n_epoch=10, show_metric=True)

