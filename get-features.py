import numpy as np # needed for reshape
import scipy.io as sio # needed for savemat
import sys

if len(sys.argv) != 4:
    print "Usage: python get-features.py START_INDEX END_INDEX LABEL"
    print "e.g.   python get-features.py 30001 35887 30k"
    sys.exit(1)

START_INDEX = int(sys.argv[1]) #30001
END_INDEX   = int(sys.argv[2]) #35887
START_LABEL = sys.argv[3] # '30k'

# Make sure that caffe is on the python path:
# caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
caffe_root = '/home/omer/code/3rdparty/caffe/'
sys.path.insert(0, caffe_root + 'python')

import caffe

caffe.set_phase_test()
caffe.set_mode_cpu()
net = caffe.Classifier(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                       caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
net.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))  # ImageNet mean
net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

features = []

for i in xrange(START_INDEX, END_INDEX): # xrange(30000,35887): ##
    image_filename = '/home/omer/Downloads/fer2013/parsed/face' + "%06d" % i +'.jpg'
    print "Extracting features for image " + image_filename
    scores = net.predict([caffe.io.load_image(image_filename)])

    feat = net.blobs['fc7'].data[4]
    features.append(feat.copy())
    if i % 5000 == 0: # save checkpoint ##
        sio.savemat('/home/omer/Downloads/fer2013/parsed/features' + START_LABEL + '_' + "%d" % (i/5000) + '.mat',{'features':features})
        print "Saved checkpoint " + '/home/omer/Downloads/fer2013/parsed/features_checkpoint_' + "%d" % (i/5000) + '.mat'
    print i

# Save the results to the mat file
sio.savemat('/home/omer/Downloads/fer2013/parsed/features_' + START_LABEL + '.mat',{'features_from' + START_LABEL :features}) ##
