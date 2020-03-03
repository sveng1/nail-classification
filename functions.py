import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD


def load_images(folder):
    """
    Loads images and labels from specified folder
    :param folder: str, directory with image data
    :return: images, list, list with images
    :return: labels, list, list with image class labels
    """
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('jpeg'):
            # Load image
            img = cv2.imread(folder+filename)
            images.append(img)
            # Get label
            if filename.endswith('good.jpeg'):
                labels.append(1)
            elif filename.endswith('bad.jpeg'):
                labels.append(0)
    return images, labels


def generate_augmented_data(images, labels):
    """
    Generates augmented data from original images
    :param images: list, images to use for data augmentation
    :param labels: list, labels corresponding to images
    :return: aug_images, list of generated images
    :return: aug_labels, list of corresponding labels
    """

    aug_images = []
    aug_labels = []

    for i in range(len(images)):
        img = images[i]
        new = []
        # rotate original
        new.append(np.rot90(img, 1))
        new.append(np.rot90(img, 2))
        new.append(np.rot90(img, 3))

        # horizontal flip and rotate
        flipped_h = np.fliplr(img)
        new.append(flipped_h)
        new.append(np.rot90(flipped_h, 1))
        new.append(np.rot90(flipped_h, 2))
        new.append(np.rot90(flipped_h, 3))

        # vertical flip and rotate
        flipped_v = np.flipud(img)
        new.append(flipped_v)
        new.append(np.rot90(flipped_v, 1))
        new.append(np.rot90(flipped_v, 2))
        new.append(np.rot90(flipped_v, 3))

        label = np.argmax(labels[i])

        aug_labels.extend([label]*len(new))
        aug_images.extend(new)
    return aug_images, to_categorical(aug_labels)


def extract_nail(img):
    """
    Extracts image patch around detected nail, using
    cv2.findContours
    :param img: image with nail
    :return patch: image patched cropped to nail
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Crop image to avoid bright areas
    y = 200
    x = 550
    h = 800
    w = 850
    cropped = gray[y:y+h, x:x+w]
    # Blur with gaussian
    blurred = cv2.medianBlur(cropped, 5)
    # Thresholding to get binary image
    ret, th1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Get contours, use them to get center of nail
    contours, hierarchy = cv2.findContours(th1, 1, 3)
    xs = []
    ys = []
    x_ends = []
    y_ends = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        xs.append(x)
        ys.append(y)
        x_ends.append(x+w)
        y_ends.append(y+h)

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(x_ends)
    y_max = max(y_ends)

    mean_x = np.mean([x_min, x_max])
    mean_y = np.mean([y_min, y_max])

    size = 128

    # Crop image with found coordinates
    top = int(mean_y-size)
    left = int(mean_x-size)
    right = int(mean_x+size)
    bottom = int(mean_y+size)

    patch = cropped[top:bottom, left:right]

    return patch


def show_class_examples(class_1, class_2, label_1, label_2):
    """
    Compares images from two different sets with matplotlib
    :param class_1: list of images from one class
    :param class_2: list of images from others class
    :param label_1: label to use as title for class_1
    :param label_2: labels to use as title for class_2
    :return: None
    """
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))

    i = 0
    for y in range(5):
        axes[0, y].imshow(class_1[i])
        axes[0, y].set_title(label_1)
        axes[0, y].get_xaxis().set_visible(False)
        axes[0, y].get_yaxis().set_visible(False)
        i += 1
    i = 0
    for y in range(5):
        axes[1, y].imshow(class_2[i])
        axes[1, y].set_title(label_2)
        axes[1, y].get_xaxis().set_visible(False)
        axes[1, y].get_yaxis().set_visible(False)
        i += 1


def cnn():
    """
    Builds and compiles CNN keras model
    :return: model
    """
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)))
    model.add(Conv2D(32, (3,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['acc'])

    return model
