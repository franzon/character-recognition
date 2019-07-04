from skimage.feature import hog, canny, local_binary_pattern
from skimage.filters import threshold_otsu
from skimage import data, exposure, io, transform
import os
import numpy as np


def compute_hog_parametrized(image_path, orientations, image_size, pixels_per_cell, filter_):
    image = io.imread(image_path)
    image = transform.resize(
        image=image, output_shape=(image_size, image_size))

    if filter_:
        thresh = threshold_otsu(image)
        binary = (image > thresh).astype(np.float32)

        fd = hog(binary, orientations=orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                 cells_per_block=(1, 1), multichannel=False,)
    else:
        fd = hog(image, orientations=orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                 cells_per_block=(1, 1), multichannel=False,)
    return fd


def build_dataset_hog_parametrized(orientations, image_size, pixels_per_cell, filter_):
    base = './dataset/Train'
    out = open(base + '.csv', 'w')
    first = True
    for folder in os.listdir(base):
        for image_path in os.listdir(base+'/'+folder):
            hog_ = compute_hog_parametrized(base+'/'+folder+'/'+image_path,
                                            orientations, image_size, pixels_per_cell, filter_)
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
                                            orientations, image_size, pixels_per_cell, filter_)
            if first:
                out.write(' '.join('f{}'.format(i) for i in range(len(hog_))))
                out.write(' label\n')
                first = False
            x = ' '.join([str(x) for x in hog_])
            out.write(x + ' ' + folder + '\n')
    out.close()
