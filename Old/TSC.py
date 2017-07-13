# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = y_train.max()+1

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Preprocess the data here.
### Feel free to use as many code cells as needed.

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.

from sklearn.model_selection import train_test_split
import numpy as np

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

print("Updated Image Shape: {}".format(X_train[0].shape))

### Generate additional data (OPTIONAL!)

print((X_train).shape)
print(len(X_train))
print((y_train).shape)
print(len(y_train))

counters = [0 for i in range(43)]

for i in range(43):
    for j in range(len(y_train)):
        if i == y_train[j]:
            counters[i] += 1
max_index = 0
for i in range(len(counters)):
    if counters[i] == max(counters):
        max_index = i

counters_max = max(counters)
# print("\n\nmax no. of element occurency is:",max(counters) , "at index:", data_1[max_index])

import cv2
from scipy import ndimage

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def rotate_90(img):
    #rotation angle in degree
    rotated = ndimage.rotate(img, 90)
    return rotated

def rotate_180(img):
    #rotation angle in degree
    rotated = ndimage.rotate(img, 180)
    return rotated

def rotate_270(img):
    #rotation angle in degree
    rotated = ndimage.rotate(img, 270)
    return rotated

# image_blur = gaussian_blur(image, 3)
# image_gray = grayscale(image_blur)
# canny_image = canny(image_gray, 50, 100)
# canny_image = canny_image.reshape(canny_image.shape + (1,))
# canny_image = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB)
#
# rotate_90_image = rotate_90(image)
# rotate_180_image = rotate_180(image)
# rotate_270_image = rotate_270(image)
#
# print("norm shape:", image.shape)
# print("blurr shape:", image_blur.shape)
# print("gray shape:", image_gray.shape)
# print("canny shape:", canny_image.shape)
# print("rotate_90_image shape:", rotate_90_image.shape)
# print("rotate_180_image shape:", rotate_180_image.shape)
# print("rotate_270_image shape:", rotate_270_image.shape)
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(image,ang_range,shear_range,trans_range):
    # Rotation
    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = image.shape
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])
    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    shear_M = cv2.getAffineTransform(pts1,pts2)

    image = cv2.warpAffine(image,Rot_M,(cols,rows))
    image = cv2.warpAffine(image,Trans_M,(cols,rows))
    image = cv2.warpAffine(image,shear_M,(cols,rows))

    #Brightness augmentation
    image = augment_brightness_camera_images(image)

    return image

#add a blurred image and an edge detected image for every item already in the training dataset
def process_edges(img):
    image_blur = gaussian_blur(img, 3)
    image_gray = grayscale(image_blur)
    canny_image = canny(image_gray, 50, 100)
    canny_image = canny_image.reshape(canny_image.shape + (1,))
    canny_image = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB)

    return canny_image

#add rotation to each of the image in the dataset

# for i in range(len(X_train)):
#     X_train = np.concatenate((X_train, np.expand_dims(rotate_90(X_train[i]), axis=0)), axis=0)
#     y_train = np.append(y_train, y_train[i])
#     X_train = np.concatenate((X_train, np.expand_dims(rotate_180(X_train[i]), axis=0)), axis=0)
#     y_train = np.append(y_train, y_train[i])
#     X_train = np.concatenate((X_train, np.expand_dims(rotate_270(X_train[i]), axis=0)), axis=0)
#     y_train = np.append(y_train, y_train[i])

#     if i%1000 == 0:
#         print(i)

# print((X_train).shape)
# print(len(X_train))
# print((y_train).shape)
# print(len(y_train))
# counter_difference = [0 for i in range(len(counters))]
# processing_indexes = [0 for i in range(len(counters))]
#
# #check how many more elements needed for equal training balance
# for i in range(len(counters)):
#     for j in range((max(counters) - counters[i])):
#         counter_difference[i] += 1
#
# print(counter_difference)
#
# #check an image for each type to use for generation
# for i in range(len(counters)):
#     for j in range(len(y_train)):
#         if i == y_train[j]:
#             processing_indexes[i] = j
#
# print(processing_indexes)
#
# #make up the difference by processing extra data
# for i in range(len(counters)):
#     print("processing type:", i)
#
#     for j in range(counter_difference[i]):
#         X_train = np.concatenate((X_train, np.expand_dims(transform_image(X_train[processing_indexes[i]], 270, 6, 20), axis=0)), axis=0)
#         y_train = np.append(y_train, y_train[i])
#         k = 1
#     print(j)

# print((X_train).shape)
# print(len(X_train))
# print((y_train).shape)
# print(len(y_train))


# counters = [0 for i in range(43)]
#
# for i in range(43):
#     for j in range(len(y_train)):
#         if i == y_train[j]:
#             counters[i] += 1
#
# counter_difference = [0 for i in range(43)]
#
# #check how many more elements needed for equal training balance
# for i in range(len(counters)):
#     for j in range((max(counters) - counters[i])):
#         counter_difference[i] += 1
#
# print(len(counters)/counters_max)

#shuffle the datasets again after training data addition

X_train, y_train = shuffle(X_train, y_train)

### Define your architecture here.
### Feel free to use as many code cells as needed.

#Setup tensorflowAnswer
import tensorflow as tf

from tensorflow.contrib.layers import flatten

def LeNet(x):

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


    ### Train your model here.
    ### Feel free to use as many code cells as needed.

# Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
mu = 0
sigma = 0.1

EPOCHS = 25
BATCH_SIZE = 256

#features and labels
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

#training pipeline
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

#model evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

#model training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet-TSC')
    print("Model saved")

import tensorflow as tf
import pickle

### Feel free to use as many code cells as needed.
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

test_accu = test_accuracy

# with tf.Session() as sess:
#     saver.restore(sess, tf.train.latest_checkpoint('.'))
#
#     for i in range(len(y_test_short)):
#
#         test_accuracy = evaluate(X_test_short[i:i+1], y_test_short[i:i+1])
#
#         print("Test Accuracy = {:.3f}".format(test_accuracy))
#
#
#     test_accuracy = evaluate(X_test_short, y_test_short)
#
#     short_test_accu = test_accuracy
#     print("Test Accuracy = {:.3f}".format(test_accuracy))
