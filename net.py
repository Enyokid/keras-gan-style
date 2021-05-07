import os, sys

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import cv2 as cv
import time
import tensorflow as tf
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from utils import InstanceNormalization
import argparse

h,w = 256, 256
inshape = (h, w, 3)

def Conv2DNormLReLU(x, k_num, k_size, padding_type, name_id):
    x = layers.Conv2D(k_num, k_size, strides=1, padding=padding_type,use_bias=None,kernel_initializer='he_normal',
               activation=None, name=name_id, trainable=True)(x)
    x = InstanceNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x

def SeparableConv2D(x, k_num, k_size, strides,padding_type, name_id):
    x = layers.SeparableConv2D(k_num,k_size,strides=strides,padding=padding_type,dilation_rate=(1, 1),depth_multiplier=1,name=name_id)(x)
    x = InstanceNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x

def InvertedRes_block(x, k_num, k_size, padding_type, name_id):
    x1 = Conv2DNormLReLU(x, k_num*2, 3, "same", None)
    x2 = layers.DepthwiseConv2D(k_size,strides=(1, 1),padding="same")(x1)
    x2 = InstanceNormalization()(x2)
    x2 = layers.LeakyReLU(alpha=0.2)(x2)

    x3 = layers.Conv2D(k_num, k_size, strides=1, padding= "same",use_bias=None,kernel_initializer='he_normal')(x2)
    x3 = InstanceNormalization()(x3)
    
    y = layers.Add(name = name_id)([x,x3])
    return y

def vgg_net(inshape):
    rawvgg=VGG16(weights='imagenet',include_top=False,input_shape=inshape)
    vgg_model=tf.keras.models.Model(inputs=rawvgg.input,outputs=rawvgg.layers[13].output)
    return vgg_model

def G_net(inshape):
    inputlayer = layers.Input(shape=inshape)
    #block 1
    conv1 = Conv2DNormLReLU(inputlayer, 64, 3, "same", "conv1")
    conv2 = Conv2DNormLReLU(conv1, 64, 3, "same", "conv2")
    downcon1 = SeparableConv2D(conv2, 128, 3, 2,"same", "downcon1")
    
    #block 2
    conv3 = Conv2DNormLReLU(downcon1, 128, 3, "same", "conv3")
    conv4 = SeparableConv2D(conv3, 128, 3, 1,"same", "conv4")
    downcon2 = SeparableConv2D(conv4, 256, 3, 2,"same", "downcon2")

    #core
    conv5 = Conv2DNormLReLU(downcon2, 256, 3, "same", "conv5")
    irb1 = InvertedRes_block(conv5, 256, 3, "same", "irb1")
    irb2 = InvertedRes_block(irb1, 256, 3, "same", "irb2")
    irb3 = InvertedRes_block(irb2, 256, 3, "same", "irb3")
    irb4 = InvertedRes_block(irb3, 256, 3, "same", "irb4")
    donv5 = Conv2DNormLReLU(irb4, 256, 3, "same", "donv5")

    #up_block2
    upcon2 = layers.UpSampling2D(size=2, interpolation="nearest", name='upcon2')(donv5)
    donv4 = SeparableConv2D(upcon2, 128, 3, 1,"same", "donv4")
    donv3 = Conv2DNormLReLU(donv4, 128, 3, "same", "donv3")

    #up_block1
    upcon1 = layers.UpSampling2D(size=2, interpolation="nearest", name='upcon1')(donv3)
    donv2_0 = SeparableConv2D(upcon1, 128, 3, 1,"same", "donv2_0")
    donv2 = Conv2DNormLReLU(upcon1, 64, 3, "same", "donv2")
    donv1 = Conv2DNormLReLU(donv2, 64, 3, "same", "donv1")

    output = layers.Conv2D(3, 3, strides=1, padding= "same",activation='tanh',kernel_initializer='he_normal',name='output')(donv1)
    return tf.keras.models.Model(inputs=inputlayer, outputs=output)


def inited_G_net(inshape):
    G_net_model = G_net(inshape)#load the gen_net

    real_img = G_net_model.input#get the input_img
    gen_img = G_net_model.output#layers[75] is the G_net's output
    
    vgg_model = vgg_net(inshape)#layers[78] is the vggmodel 

    #get the vgg_maps
    real_img_255 = layers.Lambda(lambda x: (x + 1) * 127.5)(real_img)
    gen_img_255 = layers.Lambda(lambda x: (x + 1) * 127.5)(gen_img)    
    real_vgg_map = vgg_model(real_img_255)
    gen_vgg_map = vgg_model(gen_img_255)
    
    output=layers.concatenate([real_vgg_map, gen_vgg_map], axis=-1)
    return tf.keras.models.Model(inputs=real_img, outputs=output)

def inited_G_loss(_, y_pred):    
    real_vgg_map = y_pred[:,:,:,0:512]
    gen_vgg_map = y_pred[:,:,:,512:]    
    mae = tf.keras.losses.MeanAbsoluteError()
    loss = mae(real_vgg_map, gen_vgg_map)
    return loss

def train_G_net(inshape):
    G_net_model = G_net(inshape)#load model

    real_img = G_net_model.input
    anie_gray = layers.Input(shape=inshape,name = 'anie_gray')
    gen_img = G_net_model(real_img)

    vgg_model = vgg_net(inshape)
    #get the vgg_maps
    real_img_255 = layers.Lambda(lambda x: (x + 1) * 127.5)(real_img)
    anie_gray_255 = layers.Lambda(lambda x: (x + 1) * 127.5)(anie_gray)    
    gen_img_255 = layers.Lambda(lambda x: (x + 1) * 127.5)(gen_img)

    real_vgg_map = vgg_model(real_img_255)
    anie_vgg_map = vgg_model(anie_gray_255)
    gen_vgg_map = vgg_model(gen_img_255)

    output=layers.concatenate([real_vgg_map,gen_vgg_map,anie_vgg_map], axis=-1)

    gen_img_logit = layers.Input((inshape[0]//4,inshape[1]//4,1),name = 'gen_img_logit')
    return tf.keras.models.Model(inputs=[real_img,anie_gray,gen_img_logit], outputs=[output,gen_img_logit])

def gram(x):
    shape_x = tf.shape(x)
    b = shape_x[0]
    c = shape_x[3]
    x = tf.reshape(x, [b, -1, c])
    return tf.matmul(tf.transpose(x, [0, 2, 1]), x) / tf.cast((tf.size(x) // b), tf.float32)

def train_G_loss(_, y_pred):    
    real_vgg_map = y_pred[:,:,:,:512]
    gen_vgg_map = y_pred[:,:,:,512:1024]
    gray_vgg_map = y_pred[:,:,:,1024:]
    #
    mae = tf.keras.losses.MeanAbsoluteError()
    c_loss = mae(real_vgg_map, gen_vgg_map)
    s_loss = mae(gram(gray_vgg_map), gram(gen_vgg_map))

    return 1.5*c_loss + 2.8*s_loss

def Gan_loss(y_true, y_pred):    
    #
    mse = tf.keras.losses.MeanSquaredError()
    loss = mse(y_true, y_pred)

    return 300*loss


def D_net(inshape):
    inputlayer = layers.Input(shape=inshape)
    #block 1
    conv1 = layers.Conv2D(32, 3, strides=1, padding= "same",use_bias=None,kernel_initializer='he_normal')(inputlayer)
    conv1 = layers.LeakyReLU(alpha=0.2)(conv1)
    #block 2
    conv2 = layers.Conv2D(64, 3, strides=2, padding= "same",use_bias=None,kernel_initializer='he_normal')(conv1)
    conv2 = layers.LeakyReLU(alpha=0.2)(conv2)
    conv3 = Conv2DNormLReLU(conv2, 128, 3, "same", "desr_conv3")

    #block 3
    conv4 = layers.Conv2D(128, 3, strides=2, padding= "same",use_bias=None,kernel_initializer='he_normal')(conv3)
    conv4 = layers.LeakyReLU(alpha=0.2)(conv4)
    conv5 = Conv2DNormLReLU(conv4, 256, 3, "same", "desr_conv5")

    #last block
    conv6 = Conv2DNormLReLU(conv5, 256, 3, "same", "desr_conv6")

    output = layers.Conv2D(1, 3, strides=1, padding= "same",activation=None,kernel_initializer='he_normal',name='output')(conv6)

    return tf.keras.models.Model(inputs=inputlayer, outputs=output)


#Dnet
def train_D_net(inshape):
    D_net_model = D_net(inshape)#load the des_net
    anie_img = layers.Input(shape=inshape)
    anie_gray = layers.Input(shape=inshape)
    gen_img = layers.Input(shape=inshape)
    anie_smooth = layers.Input(shape=inshape)

    anie_logit = D_net_model(anie_img)
    anie_gray_logit = D_net_model(anie_gray)
    gen_img_logit = D_net_model(gen_img)
    anie_smooth_logit = D_net_model(anie_smooth)

    output=layers.concatenate([anie_logit, anie_gray_logit,gen_img_logit,anie_smooth_logit], axis=-1)

    return tf.keras.models.Model(inputs=[anie_img,anie_gray,anie_smooth,gen_img], outputs=output)

def train_D_loss(_, y_pred):
    anie_logit = y_pred[:,:,:,0]
    anie_gray_logit = y_pred[:,:,:,1]
    anie_smooth_logit = y_pred[:,:,:,2]
    gen_img_logit  = y_pred[:,:,:,3]

    real_loss = tf.reduce_mean(tf.square(anie_logit - 1.0))
    gray_loss = tf.reduce_mean(tf.square(anie_gray_logit))
    real_blur_loss = tf.reduce_mean(tf.square(anie_smooth_logit))
    gen_loss = tf.reduce_mean(tf.square(gen_img_logit))
    
    loss = 1.7 * real_loss + 1.7 * gen_loss + 1.7 * gray_loss  +  0.8 * real_blur_loss
    return loss*300
