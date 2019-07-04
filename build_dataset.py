from skimage.feature import hog, canny, local_binary_pattern
from skimage.filters import threshold_otsu
from skimage import data, exposure, io, transform
import os
import numpy as np


def compute_hog_parametrized(image_path,  image_size, pixels_per_cell, ):
    image = io.imread(image_path)
    image = transform.resize(
        image=image, output_shape=(image_size, image_size))

    fd = hog(image, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
             multichannel=False, )
    return fd


def build_dataset_hog_parametrized(image_size, pixels_per_cell, ):
    base = './dataset/Train'
    out = open(base + '.csv', 'w')
    first = True
    for folder in os.listdir(base):
        for image_path in os.listdir(base+'/'+folder):
            hog_ = compute_hog_parametrized(base+'/'+folder+'/'+image_path,
                                            image_size, pixels_per_cell, )
            if first:
                out.write(' '.join('f{}'.format(i) for i in range(len(hog_))))
                out.write(' label\n')
                first = False
            x = ' '.join([str(x) for x in hog_])
            out.write(x + ' ' + folder + '\n')
    out.close()

    base = './dataset/Test'
    out = open(base + '.csv', 'w')
    first = True
    for folder in os.listdir(base):
        for image_path in os.listdir(base+'/'+folder):
            hog_ = compute_hog_parametrized(base+'/'+folder+'/'+image_path,
                                            image_size, pixels_per_cell, )
            if first:
                out.write(' '.join('f{}'.format(i) for i in range(len(hog_))))
                out.write(' label\n')
                first = False
            x = ' '.join([str(x) for x in hog_])
            out.write(x + ' ' + folder + '\n')
    out.close()
