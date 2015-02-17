# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Here we visualize filters and outputs using the network architecture proposed by Krizhevsky et al. for ImageNet and implemented in `caffe`.
# 
# (This page follows DeCAF visualizations originally by Yangqing Jia.)

# <markdowncell>

# First, import required modules and set plotting parameters

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# <markdowncell>

# Run `./scripts/download_model_binary.py models/bvlc_reference_caffenet` to get the pretrained CaffeNet model, load the net, specify test phase and CPU mode, and configure input preprocessing.

# <codecell>

caffe.set_phase_test()
caffe.set_mode_cpu()
net = caffe.Classifier(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                       caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
net.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))  # ImageNet mean
net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# <markdowncell>

# Run a classification pass

# <codecell>

scores = net.predict([caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')])

# <markdowncell>

# The layer features and their shapes (10 is the batch size, corresponding to the the ten subcrops used by Krizhevsky et al.)

# <codecell>

[(k, v.data.shape) for k, v in net.blobs.items()]

# <markdowncell>

# The parameters and their shapes (each of these layers also has biases which are omitted here)

# <codecell>

[(k, v[0].data.shape) for k, v in net.params.items()]

# <markdowncell>

# Helper functions for visualization

# <codecell>

# take an array of shape (n, height, width) or (n, height, width, channels)
#  and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)

# <markdowncell>

# The input image

# <codecell>

# index four is the center crop
plt.imshow(net.deprocess('data', net.blobs['data'].data[4]))

# <markdowncell>

# The first layer filters, `conv1`

# <codecell>

# the parameters are a list of [weights, biases]
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))

# <markdowncell>

# The first layer output, `conv1` (rectified responses of the filters above, first 36 only)

# <codecell>

feat = net.blobs['conv1'].data[4, :36]
vis_square(feat, padval=1)

# <markdowncell>

# The second layer filters, `conv2`
# 
# There are 256 filters, each of which has dimension 5 x 5 x 48. We show only the first 48 filters, with each channel shown separately, so that each filter is a row.

# <codecell>

filters = net.params['conv2'][0].data
vis_square(filters[:48].reshape(48**2, 5, 5))

# <markdowncell>

# The second layer output, `conv2` (rectified, only the first 36 of 256 channels)

# <codecell>

feat = net.blobs['conv2'].data[4, :36]
vis_square(feat, padval=1)

# <markdowncell>

# The third layer output, `conv3` (rectified, all 384 channels)

# <codecell>

feat = net.blobs['conv3'].data[4]
vis_square(feat, padval=0.5)

# <markdowncell>

# The fourth layer output, `conv4` (rectified, all 384 channels)

# <codecell>

feat = net.blobs['conv4'].data[4]
vis_square(feat, padval=0.5)

# <markdowncell>

# The fifth layer output, `conv5` (rectified, all 256 channels)

# <codecell>

feat = net.blobs['conv5'].data[4]
vis_square(feat, padval=0.5)

# <markdowncell>

# The fifth layer after pooling, `pool5`

# <codecell>

feat = net.blobs['pool5'].data[4]
vis_square(feat, padval=1)

# <markdowncell>

# The first fully connected layer, `fc6` (rectified)
# 
# We show the output values and the histogram of the positive values

# <codecell>

feat = net.blobs['fc6'].data[4]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)

# <markdowncell>

# The second fully connected layer, `fc7` (rectified)

# <codecell>

feat = net.blobs['fc7'].data[4]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)

# <markdowncell>

# The final probability output, `prob`

# <codecell>

feat = net.blobs['prob'].data[4]
plt.plot(feat.flat)

# <markdowncell>

# Let's see the top 5 predicted labels.

# <codecell>

# load labels
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
try:
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
except:
    !../data/ilsvrc12/get_ilsvrc_aux.sh
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

# sort top k predictions from softmax output
top_k = net.blobs['prob'].data[4].flatten().argsort()[-1:-6:-1]
print labels[top_k]

# <headingcell level=1>

# From here on it's stuff that we've written in order to extract features from the Leeds Sports Pose dataset

# <codecell>

# load labels
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
try:
   labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
except:
   !../data/ilsvrc12/get_ilsvrc_aux.sh
   labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

# <codecell>

import scipy.io as sio # Need this for 'savemat'
features = []

for i in xrange(100):
    image_filename = '/home/omer/Downloads/lsp_dataset_original/im' + "%04d" % (i+1) +'.jpg'
    print "Extracting features for image " + image_filename
    scores = net.predict([caffe.io.load_image(image_filename)])
    # # Plot the image
    # plt.imshow(net.deprocess('data', net.blobs['data'].data[4]))
    feat = net.blobs['fc7'].data[4]
    features.append(feat.copy())
    # # Plot the features
    # plt.subplot(2, 1, 1)
    # plt.plot(feat.flat)
    # plt.subplot(2, 1, 2)
    # _ = plt.hist(feat.flat[feat.flat > 0], bins=100)
    
    ## Get the classification of the image
    # feat = net.blobs['prob'].data[4]
    # plt.plot(feat.flat)
    
    # Write the class labels (need to load the labels for this to work):
    # sort top k predictions from softmax output
    top_k = net.blobs['prob'].data[4].flatten().argsort()[-1:-6:-1]
    print ", ".join(labels[top_k])

# Save the results to the mat file
sio.savemat('/home/omer/Downloads/lsp_dataset_original/features/features.mat',{'features':features})

