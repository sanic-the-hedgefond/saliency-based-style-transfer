# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 12:18:01 2019
"""
# Imports
import os
import datetime

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

# Hyperparams
ITERATIONS = 5
CHANNELS = 3
IMAGE_SIZE = 250
IMAGE_WIDTH = IMAGE_SIZE
IMAGE_HEIGHT = IMAGE_SIZE
IMAGENET_MEAN_RGB_VALUES = [123.68, 116.779, 103.939]
CONTENT_WEIGHT = 0.4
STYLE_WEIGHT_01 = 4.5
STYLE_WEIGHT_02 = 2.5
TOTAL_VARIATION_WEIGHT = 0.995
TOTAL_VARIATION_LOSS_FACTOR = 1.25

POSTPROCESSING = False
PP_brightness_min = 0.05
PP_brightness_mult = 10
PP_gamma = 2

NO_MASK = True

# Paths
input_image_path = "content_images/portrait1.jpg"
style_image_path_01 = "style_images/opart.jpg"
style_image_path_02 = "style_images/camouflage.jpg"

saliency_model = "saliency_model/model_30000b_20bsize_0.001lr"

CUSTOM_NAME = "_no_mask_02"

INPUT = os.path.basename(input_image_path)[:-4]
STYLE1 = os.path.basename(style_image_path_01)[:-4]
STYLE2 = os.path.basename(style_image_path_02)[:-4]

output_image_path = "output/{}_{}_{}_{}_{}{}/".format(INPUT, STYLE1, STYLE2, ITERATIONS, IMAGE_SIZE, CUSTOM_NAME)

if (not os.path.isdir(output_image_path)):
    os.mkdir(output_image_path)

#Input visualization 
input_image = Image.open(input_image_path)
input_image = input_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
input_image.save(output_image_path + os.path.basename(input_image_path))

# Style visualization 
style_image_01 = Image.open(style_image_path_01)
style_image_01 = style_image_01.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
style_image_01.save(output_image_path + os.path.basename(style_image_path_01))

style_image_02 = Image.open(style_image_path_02)
style_image_02 = style_image_02.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
style_image_02.save(output_image_path + os.path.basename(style_image_path_02))

# Data normalization and reshaping from RGB to BGR
input_image_array = np.asarray(input_image, dtype="float32")
input_image_array = np.expand_dims(input_image_array, axis=0)
input_image_array[:, :, :, 0] -= IMAGENET_MEAN_RGB_VALUES[2]
input_image_array[:, :, :, 1] -= IMAGENET_MEAN_RGB_VALUES[1]
input_image_array[:, :, :, 2] -= IMAGENET_MEAN_RGB_VALUES[0]
input_image_array = input_image_array[:, :, :, ::-1]

style_image_array_01 = np.asarray(style_image_01, dtype="float32")
style_image_array_01 = np.expand_dims(style_image_array_01, axis=0)
style_image_array_01[:, :, :, 0] -= IMAGENET_MEAN_RGB_VALUES[2]
style_image_array_01[:, :, :, 1] -= IMAGENET_MEAN_RGB_VALUES[1]
style_image_array_01[:, :, :, 2] -= IMAGENET_MEAN_RGB_VALUES[0]
style_image_array_01 = style_image_array_01[:, :, :, ::-1]

style_image_array_02 = np.asarray(style_image_02, dtype="float32")
style_image_array_02 = np.expand_dims(style_image_array_02, axis=0)
style_image_array_02[:, :, :, 0] -= IMAGENET_MEAN_RGB_VALUES[2]
style_image_array_02[:, :, :, 1] -= IMAGENET_MEAN_RGB_VALUES[1]
style_image_array_02[:, :, :, 2] -= IMAGENET_MEAN_RGB_VALUES[0]
style_image_array_02 = style_image_array_02[:, :, :, ::-1]

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
        
    if NO_MASK:
        mask[mask > 0.0] = 0.0
    
    imageio.imsave(output_image_path + "mask_" + os.path.basename(input_image_path), mask)
    
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
    
    style_mask = imresize(mask, (layer_size, layer_size)) / 255.0
    
    if not NO_MASK:
        #normalize 0-1 
        style_mask = (style_mask - np.amin(style_mask)) / (np.amax(style_mask) - np.amin(style_mask))
        
    style_mask_inv = 1 - style_mask
    
    print("Style_Mask MAX: {}".format(np.amax(style_mask)))
    print("Style_Mask MIN: {}".format(np.amin(style_mask)))
    print("Style_Mask_Inv MAX: {}".format(np.amax(style_mask_inv)))
    print("Style_Mask_Inv MIN: {}".format(np.amin(style_mask_inv)))
    
    #debugging
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
    
#    print("{} shape: {}".format(layer_name, style_features_01.shape))
    
    style_loss_01 = compute_style_loss(style_features_01, combination_features_01)
    style_loss_02 = compute_style_loss(style_features_02, combination_features_02)
#    style_loss_01 = compute_style_loss(style_features_01, combination_features)
#    style_loss_02 = compute_style_loss(style_features_02, combination_features)
    
    loss += (STYLE_WEIGHT_01 / len(style_layers)) * style_loss_01
    loss += (STYLE_WEIGHT_02 / len(style_layers)) * style_loss_02
    
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
combined = Image.new("RGB", (IMAGE_WIDTH*5, IMAGE_HEIGHT))
x_offset = 0

images = [output_image_path + os.path.basename(input_image_path),
          output_image_path + "mask_" + os.path.basename(input_image_path),
          output_image_path + os.path.basename(style_image_path_01),
          output_image_path + os.path.basename(style_image_path_02),
          output_image_path + "{}.png".format(ITERATIONS)]

for image in map(Image.open, images):
    combined.paste(image, (x_offset, 0))
    x_offset += IMAGE_WIDTH
combined.save(output_image_path + "combined.png")

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
    
    for i, loss in enumerate(logged_loss):
        file.write("\nLoss in Iteration {}: {}".format(i, loss))