# -*- coding: utf-8 -*-

import os
import datetime
import configparser

import numpy as np
from PIL import Image
import imageio

import tensorflow as tf

from keras import backend
from keras.applications.vgg16 import VGG16

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imresize

from skimage.transform import resize, rotate
from skimage import exposure

configParser = configparser.RawConfigParser()
configParser.read('transfer.cfg')

# Params
ITERATIONS = int(configParser.get('params', 'ITERATIONS'))
CHANNELS = 3
IMAGE_SIZE = int(configParser.get('params', 'IMAGE_SIZE'))
IMAGE_WIDTH = IMAGE_SIZE
IMAGE_HEIGHT = IMAGE_SIZE
IMAGENET_MEAN_RGB_VALUES = [123.68, 116.779, 103.939]
CONTENT_WEIGHT = float(configParser.get('params', 'CONTENT_WEIGHT'))
STYLE_WEIGHT_01 = float(configParser.get('params', 'STYLE_WEIGHT_01'))
STYLE_WEIGHT_02 = float(configParser.get('params', 'STYLE_WEIGHT_02'))
TOTAL_VARIATION_WEIGHT = float(configParser.get('params', 'TOTAL_VARIATION_WEIGHT'))
TOTAL_VARIATION_LOSS_FACTOR = float(configParser.get('params', 'TOTAL_VARIATION_LOSS_FACTOR'))

POSTPROCESSING = bool(configParser.get('params', 'POSTPROCESSING'))
PP_brightness_min = float(configParser.get('params', 'PP_brightness_min'))
PP_brightness_mult = float(configParser.get('params', 'PP_brightness_mult'))
PP_gamma = float(configParser.get('params', 'PP_gamma'))

DEBUG = bool(configParser.get('params', 'DEBUG'))
MASK = True

# Files
input_image_file = configParser.get('files', 'input_image')
style_image_01_file = configParser.get('files', 'style_image_01')
style_image_02_file = configParser.get('files', 'style_image_02')

saliency_model = "models/saliency_detection"

CUSTOM_NAME = ""

INPUT = os.path.basename(input_image_file)[:-4]
STYLE1 = os.path.basename(style_image_01_file)[:-4]
STYLE2 = os.path.basename(style_image_02_file)[:-4]

output_image_path = "output/{}_{}_{}_{}_{}{}/".format(INPUT, STYLE1, STYLE2, ITERATIONS, IMAGE_SIZE, CUSTOM_NAME)

if (not os.path.isdir(output_image_path)):
    os.mkdir(output_image_path)

def preprocess(img):
    res = Image.open(img)
    res = res.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    res = np.asarray(res, dtype="float32")
    res = np.expand_dims(res, axis=0)
    res[:, :, :, 0] -= IMAGENET_MEAN_RGB_VALUES[2]
    res[:, :, :, 1] -= IMAGENET_MEAN_RGB_VALUES[1]
    res[:, :, :, 2] -= IMAGENET_MEAN_RGB_VALUES[0]
    return res[:, :, :, ::-1]

input_image_array = preprocess(input_image_file)
style_image_array_01 = preprocess(style_image_01_file)
style_image_array_02 = preprocess(style_image_02_file)

# Model
input_image = backend.variable(input_image_array)
style_image_01 = backend.variable(style_image_array_01)
style_image_02 = backend.variable(style_image_array_02)
combination_image = backend.placeholder((1, IMAGE_HEIGHT, IMAGE_SIZE, 3))

input_tensor = backend.concatenate([input_image,style_image_01,style_image_02,combination_image], axis=0)
model = VGG16(input_tensor=input_tensor, include_top=False)

def content_loss(content, combination):
    return backend.sum(backend.square(combination - content))

layers = dict([(layer.name, layer.output) for layer in model.layers])

content_layer = "block2_conv2"
layer_features = layers[content_layer]
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[3, :, :, :]

loss = backend.variable(0.)
loss += CONTENT_WEIGHT * content_loss(content_image_features,
                                      combination_features)

def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram

def compute_style_loss(style, combination):
    style = gram_matrix(style)
    combination = gram_matrix(combination)
    size = IMAGE_HEIGHT * IMAGE_WIDTH
    return backend.sum(backend.square(style - combination)) / (4. * (CHANNELS ** 2) * (size ** 2))

def get_test_mask():
    xs = np.linspace(0, 1, IMAGE_SIZE)
    ys = np.linspace(0, 1, IMAGE_SIZE)
    [mask, _] = np.meshgrid(xs, ys)
    return mask

mask = get_test_mask()

if MASK:
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(saliency_model + ".meta")
        saver.restore(sess, saliency_model)
        
        graph = tf.get_default_graph()
        
        x = graph.get_tensor_by_name("X:0")
        is_training = graph.get_tensor_by_name("is_training:0")
        
        resized_input_image = resize(input_image_array, (1, 180, 320, 3))
        pred_saliency = sess.run("loss/prediction:0", feed_dict={x: resized_input_image, is_training: False})
        
        mask = resize(pred_saliency[0,:,:,:], output_shape=(IMAGE_SIZE,IMAGE_SIZE,1))
        
        if POSTPROCESSING:
            mask[mask > 0.05] *= 10
            mask = exposure.adjust_gamma(mask, 2)  
            mask[mask > 1.0] = 1.0
            
            print("Mask MAX: {}".format(np.amax(mask)))
            print("Mask MIN: {}".format(np.amin(mask)))
        
        imageio.imsave(output_image_path + "mask_" + os.path.basename(input_image_file), mask)
        
        mask = np.squeeze(mask)
   
    print("\n=== GENERATED MASK ===\n")

style_layers = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]
for layer_name in style_layers:
    layer_features = layers[layer_name]
    
    style_features_01 = layer_features[1, :, :, :]
    style_features_02 = layer_features[2, :, :, :]
    
    combination_features = layer_features[3, :, :, :]
    
    layer_size = style_features_01.shape[0]
    layer_depth = style_features_01.shape[2]
    
    if MASK:
        style_mask = imresize(mask, (layer_size, layer_size)) / 255.0

        #normalize 0-1 
        style_mask = (style_mask - np.amin(style_mask)) / (np.amax(style_mask) - np.amin(style_mask))
        
        style_mask_inv = 1 - style_mask
        
        print("Style_Mask MAX: {}".format(np.amax(style_mask)))
        print("Style_Mask MIN: {}".format(np.amin(style_mask)))
        print("Style_Mask_Inv MAX: {}".format(np.amax(style_mask_inv)))
        print("Style_Mask_Inv MIN: {}".format(np.amin(style_mask_inv)))
        
        #debugging
        if DEBUG:
            imageio.imsave(output_image_path + layer_name + "_mask.jpg", style_mask)
            imageio.imsave(output_image_path + layer_name + "_mask_inv.jpg", style_mask_inv)
            print(np.amax(style_mask))
            print(np.amin(style_mask))
        
        #normalize mask [0,1]
        style_mask = (style_mask - np.amin(style_mask)) / (np.amax(style_mask) - np.amin(style_mask))
        style_mask_inv = 1 - style_mask
        
        #for some reason the mask gets rotated later, so counter-rotate now 
        style_mask = rotate(style_mask, 90)
        style_mask_inv = rotate(style_mask_inv, 90)
        
        #for some reason the mask gets flipped-LR, so undo that, too...
        style_mask = np.flipud(style_mask)
        style_mask_inv = np.flipud(style_mask_inv)
    
        # stack mask layer to fit number of feature maps
        style_mask = np.array([style_mask for _ in range(layer_depth)]).transpose()
        style_mask_inv = np.array([style_mask_inv for _ in range(layer_depth)]).transpose()
        
        #apply mask
        style_features_01 = tf.multiply(style_features_01, style_mask)
        style_features_02 = tf.multiply(style_features_02, style_mask_inv)
        
        combination_features_01 = tf.multiply(combination_features, style_mask)
        combination_features_02 = tf.multiply(combination_features, style_mask_inv)
        
        style_loss_01 = compute_style_loss(style_features_01, combination_features_01)
        loss += (STYLE_WEIGHT_01 / len(style_layers)) * style_loss_01
        
        style_loss_02 = compute_style_loss(style_features_02, combination_features_02)   
        loss += (STYLE_WEIGHT_02 / len(style_layers)) * style_loss_02
    
    else:
        style_loss = compute_style_loss(style_features_02, combination_features)   
        loss += (STYLE_WEIGHT_02 / len(style_layers)) * style_loss
        
#    print("{} shape: {}".format(layer_name, style_features_01.shape))

def total_variation_loss(x):
    a = backend.square(x[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - x[:, 1:, :IMAGE_WIDTH-1, :])
    b = backend.square(x[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - x[:, :IMAGE_HEIGHT-1, 1:, :])
    return backend.sum(backend.pow(a + b, TOTAL_VARIATION_LOSS_FACTOR))

loss += TOTAL_VARIATION_WEIGHT * total_variation_loss(combination_image)

outputs = [loss]
outputs += backend.gradients(loss, combination_image)

def evaluate_loss_and_gradients(x):
    x = x.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    outs = backend.function([combination_image], outputs)([x])
    loss = outs[0]
    gradients = outs[1].flatten().astype("float64")
    return loss, gradients

class Evaluator:

    def loss(self, x):
        loss, gradients = evaluate_loss_and_gradients(x)
        self._gradients = gradients
        return loss

    def gradients(self, x):
        return self._gradients

evaluator = Evaluator()

x = np.random.uniform(0, 255, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)) - 128.

print("=== START STYLE TRANSFER ===\n\n")

logged_loss = []

for i in range(ITERATIONS):
    i += 1
    x, loss, _ = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.gradients, maxfun=20)
    print("=== ITERATION {} COMPLETED: LOSS {} ===".format(i, loss))
    
    logged_loss.append(loss)
    
    x = x.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    x = x[:, :, ::-1]
    x[:, :, 0] += IMAGENET_MEAN_RGB_VALUES[2]
    x[:, :, 1] += IMAGENET_MEAN_RGB_VALUES[1]
    x[:, :, 2] += IMAGENET_MEAN_RGB_VALUES[0]
    x = np.clip(x, 0, 255).astype("uint8")
    output_image = Image.fromarray(x)
    output_image.save(output_image_path + "{}.png".format(i))
    #output_image
    

# Visualizing and saving combined results
"""
combined = Image.new("RGB", (IMAGE_WIDTH*5, IMAGE_HEIGHT))
x_offset = 0

images = [output_image_path + os.path.basename(input_image_file),
          output_image_path + "mask_" + os.path.basename(input_image_file),
          output_image_path + os.path.basename(style_image_01),
          output_image_path + os.path.basename(style_image_02),
          output_image_path + "{}.png".format(ITERATIONS)]

for image in map(Image.open, images):
    combined.paste(image, (x_offset, 0))
    x_offset += IMAGE_WIDTH
combined.save(output_image + "combined.png")
"""

if DEBUG:
    # Write Logfile with settings
    with open(output_image_path + "settings.txt", "w+") as file:
        file.write("DATE: {}\n".format(datetime.datetime.now()))
        file.write("ITERATIONS: {}\n".format(ITERATIONS))
        file.write("IMAGE_SIZE: {}\n".format(IMAGE_SIZE))
        file.write("CONTENT_WEIGHT({}): {}\n".format(INPUT, CONTENT_WEIGHT))
        file.write("STYLE_WEIGHT_01({}): {}\n".format(STYLE1, STYLE_WEIGHT_01))
        file.write("STYLE_WEIGHT_02({}): {}\n".format(STYLE2, STYLE_WEIGHT_02))
        file.write("TOTAL_VARIATION_WEIGHT: {}\n".format(TOTAL_VARIATION_WEIGHT))
        file.write("TOTAL_VARIATION_LOSS_FACTOR: {}\n".format(TOTAL_VARIATION_LOSS_FACTOR))
        file.write("\nMASK: {}\n".format(MASK))
        file.write("\nPP: {}\n".format(POSTPROCESSING))
        file.write("PP brightness min : {}\n".format(PP_brightness_min))
        file.write("PP brightness mul: {}\n".format(PP_brightness_mult))
        file.write("PP gamma: {}\n".format(PP_gamma))
        
        for i, loss in enumerate(logged_loss):
            file.write("\nLoss in Iteration {}: {}".format(i, loss))