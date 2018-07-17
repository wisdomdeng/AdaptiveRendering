"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])


# -----------------------------
# new added functions for pix2pix

def load_data(image_path, flip=False, is_test=False):
    style, pose, target = load_image(image_path)
    style, pose, target = preprocess_SPT(style, pose, target, flip=flip, is_test=is_test)

    style = style / 127.5 - 1.
    pose = pose / 127.5 - 1.
    target = target / 127.5 - 1.

    img_SPT = np.concatenate((style, pose, target), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_SPT


def load_image(image_path):
    image_path = image_path.split(',')
    style = imread(image_path[0])
    pose = imread(image_path[1])
    target = imread(image_path[2])

    return style, pose, target


def preprocess_SPT(style, pose, target, load_size=134, fine_size=128, flip=False, is_test=False):
    if is_test:
        style = scipy.misc.imresize(style, [fine_size, fine_size])
        pose = scipy.misc.imresize(pose, [fine_size, fine_size])
        target = scipy.misc.imresize(target, [fine_size, fine_size])
    else:
        style = scipy.misc.imresize(style, [fine_size, fine_size])
        pose = scipy.misc.imresize(pose, [fine_size, fine_size])
        target = scipy.misc.imresize(target, [fine_size, fine_size])
        # style = scipy.misc.imresize(style, [load_size, load_size])
        # pose = scipy.misc.imresize(pose, [load_size, load_size])
        # target = scipy.misc.imresize(target, [load_size, load_size])

        # h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        # w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        # style = style[h1:h1+fine_size, w1:w1+fine_size]
        # pose = pose[h1:h1+fine_size, w1:w1+fine_size]
        # target = target[h1:h1+fine_size, w1:w1+fine_size]

        # if flip and np.random.random() > 0.5:
        #     style = np.fliplr(style)
        #     pose = np.fliplr(pose)
        #     target = np.fliplr(target)

    return style, pose, target


# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale=False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
    return (images + 1.) / 2.
