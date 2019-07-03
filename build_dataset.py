from skimage.feature import hog, canny, local_binary_pattern
from skimage import data, exposure, io, transform
import os
import numpy as np


def compute_hog(image_path):
    image = io.imread(image_path)
    image = transform.resize(image=image, output_shape=(50, 50))

    fd = hog(image, orientations=8, pixels_per_cell=(16, 16),
             cells_per_block=(1, 1), multichannel=False,)
    return fd


def compute_hog_experiments(image_path):
    image = io.imread(image_path)

    height = len(image)
    width = len(image[0])

    image = transform.resize(image=image, output_shape=(100, 100))

    fd = hog(image, orientations=8, pixels_per_cell=(20, 20),
             cells_per_block=(1, 1), multichannel=False,)
    return fd


def build_dataset_hog(type):
    base = './dataset/{}'.format(type)
    out = open(base + '.csv', 'w')
    first = True
    for folder in os.listdir(base):
        for image_path in os.listdir(base+'/'+folder):
            hog_ = compute_hog(base+'/'+folder+'/'+image_path)
            if first:
                out.write(' '.join('f{}'.format(i) for i in range(len(hog_))))
                out.write(' label\n')
                first = False
            x = ' '.join([str(x) for x in hog_])
            out.write(x + ' ' + folder + '\n')
    out.close()


def build_dataset_experiments(type):
    base = './dataset/{}'.format(type)
    out = open(base + '.csv', 'w')
    first = True
    for folder in os.listdir(base):
        for image_path in os.listdir(base+'/'+folder):
            hog_ = compute_hog_experiments(base+'/'+folder+'/'+image_path)
            if first:
                out.write(' '.join('f{}'.format(i) for i in range(len(hog_))))
                out.write(' label\n')
                first = False
            x = ' '.join([str(x) for x in hog_])
            out.write(x + ' ' + folder + '\n')
    out.close()


build_dataset_experiments('Train')
build_dataset_experiments('Test')
