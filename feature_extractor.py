from skimage import data, io, filters, feature, exposure, color
import numpy as np
import cv2
import os


def histogram():
    for f in ['Train', 'Valid']:
        csvfile = open('histogram-{}.csv'.format(f), 'w')

        first = True
        for filename in os.listdir('./dataset/{}'.format(f)):
            if filename.endswith('.bmp'):
                image = io.imread(
                    './dataset/{}/{}'.format(f, filename), as_gray=True)

                fd, hist_centers = exposure.histogram(image)
                if first:
                    csvfile.write(' '.join(['feature{}'.format(i)
                                            for i in range(len(fd))] + ['label\n']))
                    first = False

                csvfile.write(' '.join(str(f) for f in fd))
                if filename.startswith('bart'):
                    csvfile.write(' bart')
                elif filename.startswith('homer'):
                    csvfile.write(' homer')
                elif filename.startswith('lisa'):
                    csvfile.write(' lisa')
                elif filename.startswith('maggie'):
                    csvfile.write(' maggie')
                elif filename.startswith('marge'):
                    csvfile.write(' marge')
                else:
                    print(filename)
                csvfile.write('\n')

        csvfile.close()


def sift(vector_size):
    for f in ['Train', 'Valid']:
        csvfile = open('sift-{}.csv'.format(f), 'w')

        first = True
        for filename in os.listdir('./dataset/{}'.format(f)):
            if filename.endswith('.bmp'):
                image = cv2.imread(
                    './dataset/{}/{}'.format(f, filename), cv2.IMREAD_GRAYSCALE)

                sift = cv2.xfeatures2d.SIFT_create(vector_size)
                kps, fd = sift.detectAndCompute(image, None)
                fd = fd.flatten()
                print(fd.size)
                # needed_size = (vector_size * 1)
                # if fd.size < needed_size:
                #     print(fd.size)
                #     fd = np.concatenate([fd, np.zeros(needed_size-fd.size)])
                #     print(fd.size)
                if first:
                    csvfile.write(' '.join(['feature{}'.format(i)
                                            for i in range(len(fd))] + ['label\n']))
                    first = False

                csvfile.write(' '.join(str(f) for f in fd))
                if filename.startswith('bart'):
                    csvfile.write(' bart')
                elif filename.startswith('homer'):
                    csvfile.write(' homer')
                elif filename.startswith('lisa'):
                    csvfile.write(' lisa')
                elif filename.startswith('maggie'):
                    csvfile.write(' maggie')
                elif filename.startswith('marge'):
                    csvfile.write(' marge')
                else:
                    print(filename)
                csvfile.write('\n')

        csvfile.close()


def surf():
    for f in ['Train', 'Valid']:
        csvfile = open('surf-{}.csv'.format(f), 'w')

        first = True
        for filename in os.listdir('./dataset/{}'.format(f)):
            if filename.endswith('.bmp'):
                image = cv2.imread(
                    './dataset/{}/{}'.format(f, filename), cv2.IMREAD_GRAYSCALE)

                vector_size = 32
                alg = cv2.xfeatures2d.SURF_create()
                kps = alg.detect(image)
                kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
                kps, dsc = alg.compute(image, kps)
                dsc = dsc.flatten()
                needed_size = (vector_size * 64)
                if dsc.size < needed_size:
                    dsc = np.concatenate(
                        [dsc, np.zeros(needed_size - dsc.size)])
                else:

                    dsc = dsc[:needed_size]
                # needed_size = (vector_size * 1)
                # if fd.size < needed_size:
                #     print(fd.size)
                #     fd = np.concatenate([fd, np.zeros(needed_size-fd.size)])
                #     print(fd.size)
                if first:
                    csvfile.write(' '.join(['feature{}'.format(i)
                                            for i in range(len(dsc))] + ['label\n']))
                    first = False

                csvfile.write(' '.join(str(f) for f in dsc))
                if filename.startswith('bart'):
                    csvfile.write(' bart')
                elif filename.startswith('homer'):
                    csvfile.write(' homer')
                elif filename.startswith('lisa'):
                    csvfile.write(' lisa')
                elif filename.startswith('maggie'):
                    csvfile.write(' maggie')
                elif filename.startswith('marge'):
                    csvfile.write(' marge')
                else:
                    print(filename)
                csvfile.write('\n')

        csvfile.close()


# sift(32)
# kaze()
surf()
