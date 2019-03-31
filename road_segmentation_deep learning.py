"""
Copyright@2017 by Yi Liu and Jan Krepl 


Code for road segmentation based on Tensorflow .

The model consists of a 16 layers CNN layers with a soft-max loss.
We use AdamOptimizer optimization in the tensorflow.
This version gives 0.88 score and after postprocessing it gives 0.91

"""


import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image
import code
import tensorflow.python.platform
import numpy
import tensorflow as tf




NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 100
VALIDATION_SIZE = 10  # Size of the validation set.
SEED = None  # Set to None for random seed.
BATCH_SIZE = 1
NUM_EPOCHS = 1
RESTORE_MODEL = False # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000


############## FROM JAN #################
# Reuses the image to generate additional training data
OFFSET_DATA = [False, [2,4,11,13,3]] 
# OFFSET_DATA[0] - active/nonactive
# OFFSET_DATA[1] - offset sizes (number of skipped pixels at the start) 
#                - has to be smaller than IMG_PATCH_SIZE   


OFFSET_TEST = [False, 2]  # IMPLEMENTED FOR PATCH SIZE 16
# OFFSET_TEST[0] - active/inactive
# OFFSET_TEST[1] - offset_size

##########################################


# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!

IMG_PATCH_SIZE = 16
tensorboard_dir='tmp_deep_11_final'


tf.app.flags.DEFINE_string('train_dir', tensorboard_dir,
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS



# Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


# Image crop with an offset - FROM JAN
def img_crop_augmented(im, w, h, offset): # offset < w=h
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
        
    for i in range(offset,imgheight - h,h):
        for j in range(offset,imgwidth - w,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)        
    return list_patches


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            #print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]

    ######## ADD OFFSET
    if OFFSET_DATA[0]:
        for offset_size in OFFSET_DATA[1]:
            img_patches_offset = [img_crop_augmented(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE,offset_size) for i in range(num_images)]    
            img_patches += img_patches_offset
    
    ######

    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return numpy.asarray(data)
        


# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:
        return [0, 1] 
    else:
        return [1, 0]



# Extract label images
def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            #print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]

    ######## ADD OFFSET
    if OFFSET_DATA[0]:
        for offset_size in OFFSET_DATA[1]:
            gt_patches_offset = [img_crop_augmented(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE,offset_size) for i in range(num_images)]    
            gt_patches += gt_patches_offset
    
    ######


    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)



def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])


# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()


# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    print (str(max_labels) + ' ' + str(max_predictions))


# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if labels[idx][0] > 0.5:   # here  others labled as l=1
                l = 1
            else:
                l = 0
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels


def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / (numpy.max(rimg)+1e-5) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg


def concatenate_images(img, gt_img): 
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


def img_crop_with_offset_patches(img, w, h, offset_size):
    # we discard edge patches and for each middle patch we also right away generate all 8 of its offset patches
    # since we will apply it on test images, we know its a 3d image
    # Initialize output
    list_patches = []
    list_patches_1 = []
    list_patches_2 = []
    list_patches_3 = []
    list_patches_4 = []
    list_patches_5 = []
    list_patches_6 = []
    list_patches_7 = []
    list_patches_8 = []
    
    imgwidth = img.shape[0]
    imgheight = img.shape[1]
    
    # We populate in column direction first 
    for c in range(w,imgwidth - w, w):
        for r in range(h, imgheight - h, h):
            patch = img[r:r+h, c:c+w,:]
            patch_1 = img[r:r+h, c-offset_size:c+w-offset_size,:]
            patch_2 = img[r-offset_size:r+h-offset_size, c-offset_size:c+w-offset_size,:]
            patch_3 = img[r-offset_size:r+h-offset_size, c:c+w,:]
            patch_4 = img[r-offset_size:r+h-offset_size, c+offset_size:c+w+offset_size,:]
            patch_5 = img[r:r+h, c+offset_size:c+w+offset_size,:]
            patch_6 = img[r+offset_size:r+h+offset_size, c+offset_size:c+w+offset_size,:]
            patch_7 = img[r+offset_size:r+h+offset_size, c:c+w,:]
            patch_8 = img[r+offset_size:r+h+offset_size, c-offset_size:c+w-offset_size,:]
            
            list_patches.append(patch)
            list_patches_1.append(patch_1)
            list_patches_2.append(patch_2)
            list_patches_3.append(patch_3)
            list_patches_4.append(patch_4)
            list_patches_5.append(patch_5)
            list_patches_6.append(patch_6)
            list_patches_7.append(patch_7)
            list_patches_8.append(patch_8)
    
    return [list_patches,list_patches_1,list_patches_2,list_patches_3,list_patches_4,list_patches_5,list_patches_6,list_patches_7,list_patches_8]




##########################
#START MAIN FUNCTION BELOW
##############################





def main(argv=None):  # pylint: disable=unused-argument

    data_dir = 'data/training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/' 

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, TRAINING_SIZE)
    train_labels = extract_labels(train_labels_filename, TRAINING_SIZE)

    num_epochs = NUM_EPOCHS

    
    train_size = train_labels.shape[0]

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.


    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS))
    train_all_data_node = tf.constant(train_data)


    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}

    """
    ^^^^^^^^^^^^^^^^^^^^^^^^^^
    Initialize weights for CNN
    ^^^^^^^^^^^^^^^^^^^^^^^^^^
    """
###################
    conv1_weights = tf.Variable(
        tf.truncated_normal([3, 3, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([32]))
###################
    conv2_weights = tf.Variable(
        tf.truncated_normal([3, 3, 32, 32],
                            stddev=0.1,
                            seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[32]))
###################
    conv3_weights = tf.Variable(
        tf.truncated_normal([3, 3, 32, 64],
                            stddev=0.1,
                            seed=SEED))
    conv3_biases = tf.Variable(tf.constant(0.1, shape=[64]))

################### 
    conv4_weights = tf.Variable(
        tf.truncated_normal([3, 3, 64, 64],
                            stddev=0.1,
                            seed=SEED))
    conv4_biases = tf.Variable(tf.constant(0.1, shape=[64]))
###################
    conv5_weights = tf.Variable(
        tf.truncated_normal([3, 3, 64, 64],
                            stddev=0.1,
                            seed=SEED))
    conv5_biases = tf.Variable(tf.constant(0.1, shape=[64]))
###################
    conv6_weights = tf.Variable(
        tf.truncated_normal([3, 3, 64, 128],
                            stddev=0.1,
                            seed=SEED))
    conv6_biases = tf.Variable(tf.constant(0.1, shape=[128]))
###################
    conv7_weights = tf.Variable(
        tf.truncated_normal([3, 3, 128, 256],
                            stddev=0.1,
                            seed=SEED))
    conv7_biases = tf.Variable(tf.constant(0.1, shape=[256]))
###################
    conv8_weights = tf.Variable(
        tf.truncated_normal([3, 3, 256, 256],
                            stddev=0.1,
                            seed=SEED))
    conv8_biases = tf.Variable(tf.constant(0.1, shape=[256]))
###################
    conv9_weights = tf.Variable(
        tf.truncated_normal([1, 1, 256, 512],
                            stddev=0.1,
                            seed=SEED))
    conv9_biases = tf.Variable(tf.constant(0.1, shape=[512]))
###################
    conv10_weights = tf.Variable(
        tf.truncated_normal([1, 1, 512, 512],
                            stddev=0.1,
                            seed=SEED))
    conv10_biases = tf.Variable(tf.constant(0.1, shape=[512]))
###################
    conv11_weights = tf.Variable(
        tf.truncated_normal([1, 1, 512, 512],
                            stddev=0.1,
                            seed=SEED))
    conv11_biases = tf.Variable(tf.constant(0.1, shape=[512]))
###################

    fc1_weights = tf.Variable(  # fully connected, depth 512. ... assumes input layer 4096
        tf.truncated_normal([int(512), 1000],
                            stddev=0.1,
                            seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[1000]))
####################
    fc2_weights = tf.Variable(
        tf.truncated_normal([1000, NUM_LABELS],
                            stddev=0.1,
                            seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))
###################

    

    # Make an image summary for 4d tensor image with index idx
    def get_image_summary(img, idx = 0):
        V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        min_value = tf.reduce_min(V)
        V = V - min_value
        max_value = tf.reduce_max(V)
        V = V / (max_value*PIXEL_DEPTH)
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V
    

    # Make an image summary for 3d tensor image with index idx
    def get_image_summary_3d(img):
        V = tf.slice(img, (0, 0, 0), (1, -1, -1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V

    # Get prediction for given input image 
    def get_prediction(img):
        data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
        data_node = tf.constant(data)
        output = tf.nn.softmax(model(data_node))
        output_prediction = s.run(output)

        if OFFSET_TEST[0]:
            zipped = img_crop_with_offset_patches(img,IMG_PATCH_SIZE,IMG_PATCH_SIZE,OFFSET_TEST[1])
            
            # Get all offset patches
            data_inner = numpy.asarray(zipped[0])
            data_1 = numpy.asarray(zipped[1])
            data_2 = numpy.asarray(zipped[2])
            data_3 = numpy.asarray(zipped[3])
            data_4 = numpy.asarray(zipped[4])
            data_5 = numpy.asarray(zipped[5])
            data_6 = numpy.asarray(zipped[6])
            data_7 = numpy.asarray(zipped[7])
            data_8 = numpy.asarray(zipped[8])
            
        
            
            data_node_inner = tf.constant(data_inner)
            data_node_1 = tf.constant(data_1)
            data_node_2 = tf.constant(data_2)
            data_node_3 = tf.constant(data_3)
            data_node_4 = tf.constant(data_4)
            data_node_5 = tf.constant(data_5)
            data_node_6 = tf.constant(data_6)
            data_node_7 = tf.constant(data_7)
            data_node_8 = tf.constant(data_8)
            
            output_inner = tf.nn.softmax(model(data_node_inner)) 
            output_1 = tf.nn.softmax(model(data_node_1)) 
            output_2 = tf.nn.softmax(model(data_node_2)) 
            output_3 = tf.nn.softmax(model(data_node_3)) 
            output_4 = tf.nn.softmax(model(data_node_4)) 
            output_5 = tf.nn.softmax(model(data_node_5)) 
            output_6 = tf.nn.softmax(model(data_node_6)) 
            output_7 = tf.nn.softmax(model(data_node_7)) 
            output_8 = tf.nn.softmax(model(data_node_8)) 
            
            output_prediction_inner = s.run(output_inner)
            output_prediction_1 = s.run(output_1)
            output_prediction_2 = s.run(output_2)
            output_prediction_3 = s.run(output_3)
            output_prediction_4 = s.run(output_4)
            output_prediction_5 = s.run(output_5)
            output_prediction_6 = s.run(output_6)
            output_prediction_7 = s.run(output_7)
            output_prediction_8 = s.run(output_8)
            
            
            for ix in range(1,9):
                output_prediction_inner += eval('output_prediction_' + str(ix))
                
            output_prediction_inner /= 9

            
            # Embedding inner data into the entire data..designed for patch size 16
            I_inner = [i for i in range(1444) if (i > 38) and (i < 1444 - 38) and (i%38 != 0) and ((i+1)%38 != 0)]
            
            output_prediction[I_inner] = output_prediction_inner               



        img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

        return img_prediction

    #END get_prediction(img)




    # Get a concatenation of the prediction and groundtruth for given input file
    def get_prediction_with_groundtruth(filename, image_idx):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)
        img_prediction = get_prediction(img)
        cimg = concatenate_images(img, img_prediction)

        return cimg



    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay(filename, image_idx):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)
        img_prediction = get_prediction(img)
        oimg = make_img_overlay(img, img_prediction)

        return oimg


    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.


#############################
#Start Model
#############################


    def model(data, train=True):

        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

        conv2 = tf.nn.conv2d(conv,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

        pool1 = tf.nn.max_pool(relu2,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')


        conv3 = tf.nn.conv2d(pool1,
                            conv3_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))


        conv4 = tf.nn.conv2d(relu3,
                            conv4_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))


        conv5= tf.nn.conv2d(relu4,
                            conv5_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))

        
        pool2 = tf.nn.max_pool(relu5,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        
        
        conv6 = tf.nn.conv2d(pool2,
                            conv6_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        relu6 = tf.nn.relu(tf.nn.bias_add(conv6, conv6_biases))

        conv7 = tf.nn.conv2d(relu6,
                            conv7_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        relu7 = tf.nn.relu(tf.nn.bias_add(conv7, conv7_biases))


        conv8 = tf.nn.conv2d(relu7,
                            conv8_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        relu8 = tf.nn.relu(tf.nn.bias_add(conv8, conv8_biases))

        
        pool3 = tf.nn.max_pool(relu8,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        conv9 = tf.nn.conv2d(pool3,
                            conv9_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        relu9 = tf.nn.relu(tf.nn.bias_add(conv9, conv9_biases))

        conv10 = tf.nn.conv2d(relu9,
                            conv10_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        relu10 = tf.nn.relu(tf.nn.bias_add(conv10, conv10_biases))

        conv11 = tf.nn.conv2d(relu10,
                            conv11_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        relu11 = tf.nn.relu(tf.nn.bias_add(conv11, conv11_biases))


        pool4 = tf.nn.max_pool(relu11,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        
        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool4.get_shape().as_list()
        reshape = tf.reshape(
            pool4,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)


        out = tf.matmul(hidden, fc2_weights) + fc2_biases


        if True:
            summary_id = '_0'
            s_data = get_image_summary(data)
            filter_summary0 = tf.summary.image('summary_data' + summary_id, s_data)
            #s_conv = get_image_summary(conv)
            #filter_summary2 = tf.summary.image('summary_conv' + summary_id, s_conv)
            #s_pool1 = get_image_summary(pool1)
            #filter_summary3 = tf.summary.image('summary_pool' + summary_id, s_pool1)
            s_conv10 = get_image_summary(conv10)
            filter_summary1 = tf.summary.image('summary_conv10' + summary_id, s_conv10)
            #s_pool4 = get_image_summary(pool4)
            #filter_summary5 = tf.summary.image('summary_pool2' + summary_id, s_pool4)

        return out




########################
#End Model
########################




    # Training computation: logits + cross-entropy loss.

    logits = model(train_data_node, True) # BATCH_SIZE*NUM_LABELS
    # 
    #print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())


    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=train_labels_node,logits=logits))
    tf.summary.scalar('loss', loss)



    all_params_node = [conv1_weights, conv1_biases,conv2_weights, conv2_biases, conv3_weights, conv3_biases, 
                       conv4_weights, conv4_biases,conv5_weights, conv5_biases, conv6_weights, conv6_biases, 
                       conv7_weights, conv7_biases,conv8_weights, conv8_biases, conv9_weights, conv9_biases, 
                       conv10_weights, conv10_biases,conv11_weights, conv11_biases,
                       fc1_weights, fc1_biases, fc2_weights, fc2_biases]
    all_params_names = ['conv1_weights', 'conv1_biases','conv2_weights', 'conv2_biases','conv3_weights', 'conv3_biases',
                        'conv4_weights', 'conv4_biases','conv5_weights', 'conv5_biases','conv6_weights', 'conv6_biases',
                        'conv7_weights', 'conv7_biases','conv8_weights', 'conv8_biases','conv9_weights', 'conv9_biases',
                        'conv10_weights', 'conv10_biases','conv11_weights', 'conv11_biases',
                        'fc1_weights', 'fc1_biases', 'fc2_weights', 'fc2_biases']

    all_grads_node = tf.gradients(loss, all_params_node)

    all_grad_norms_node = []


    for i in range(0, len(all_grads_node)):
        norm_grad_i = tf.global_norm([all_grads_node[i]])
        all_grad_norms_node.append(norm_grad_i)
        tf.summary.scalar(all_params_names[i], norm_grad_i)
    

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers




    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)

    tf.summary.scalar('learning_rate', learning_rate)
    

    # Use simple momentum for the optimization.
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=0.001,use_locking=False,name='Adam').minimize(loss,global_step=batch)
    #(learning_rate,initial_accumulator_value=0.1,use_locking=False, name='Adagrad').minimize(loss,global_step=batch)

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.
    train_all_prediction = tf.nn.softmax(model(train_all_data_node))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()


    # Create a local session to run this computation.
    with tf.Session() as s:


        if RESTORE_MODEL:
            # Restore variables from disk.
            saver.restore(s,"tmp_deep_11_v"+ "/model.ckpt")
            print("Model restored.")


        else:

            # Run all the initializers to prepare the trainable parameters.
            
            #tf.global_variables_initializer().run()
            saver.restore(s, "tmp_deep_11_v"+ "/model.ckpt")
            print("Model restored and continued")

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir,graph=s.graph)
    

            print ('Initialized!')

            # Loop through training steps.
            print ('Total number of iterations = ' + str(int(num_epochs * train_size / BATCH_SIZE)))


            training_indices = range(train_size)

            for iepoch in range(num_epochs):

                # Permute training indices
                perm_indices = numpy.random.permutation(training_indices)


                for step in range (1):#int(train_size / BATCH_SIZE)):

                    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                    batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = train_data[batch_indices, :, :, :]
                    batch_labels = train_labels[batch_indices]
                    # This dictionary maps the batch data (as a numpy array) to the
                    # node in the graph is should be fed to.
                    feed_dict = {train_data_node: batch_data,
                                 train_labels_node: batch_labels}

                    if step % RECORDING_STEP == 0:

                        summary_str, _, l, lr, predictions = s.run(
                            [summary_op,optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()


                        #print_predictions(predictions, batch_labels)

                        print ('Epoch %.2f' % (float(step) * float(BATCH_SIZE )/ float(train_size)))
                        print ('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                        print ('Minibatch error: %.1f%%' % error_rate(predictions,
                                                                     batch_labels))
                        sys.stdout.flush()

                    else:
                        # Run the graph and fetch some of the nodes.
                        _, l, lr, predictions = s.run(
                            [optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)

                # Save the variables to disk.
                save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt")
                print("Model saved in file: %s" % save_path)



        print ("Running prediction on test set")
        prediction_test_dir = "predictions_submit/"
        if not os.path.isdir(prediction_test_dir):
            os.mkdir(prediction_test_dir)
        test_dir='data/test_set_images/'



        #for i in range(1, 50+1):


        #    testid = "test_%d/" % i
        #    test_data_filename=test_dir+testid+"test_%d" % i+".png"
        #    img = mpimg.imread(test_data_filename)
        #    img_prediction = get_prediction(img)
        #    img_prediction=img_float_to_uint8(img_prediction) 
        #    Image.fromarray(img_prediction).save(prediction_test_dir + "prediction_"+ str(i) + ".png")


"""
###################
END MAIN  FUNCTION
######################
"""

if __name__ == '__main__':
    tf.app.run()

   
