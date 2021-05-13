import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from net import *
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
import time
from tensorflow.keras.utils import multi_gpu_model


h,w = 256, 256
inshape = (h, w, 3)
epochs = 100
batch_size = 4

def COMPILE(inshape,D_model,G_model):
    real_img = G_model.input
    anie_gray = layers.Input(shape=inshape,name = 'anie_gray')
    gen_img = G_model(real_img)
    gen_img_logit = D_model(gen_img)

    vgg_model = vgg_net(inshape)

    real_vgg_map = vgg_model(real_img)
    anie_vgg_map = vgg_model(anie_gray)
    gen_vgg_map = vgg_model(gen_img)

    output=layers.concatenate([real_vgg_map,gen_vgg_map,anie_vgg_map], axis=-1)
    output2=layers.concatenate([real_img,gen_img], axis=-1)
    return tf.keras.models.Model(inputs=[real_img,anie_gray], outputs=[output,output2,gen_img_logit])

#compile model
with tf.device('/cpu:0'):
    train_D_model = train_D_net(inshape)
    train_D_model.compile(Adam(lr=4e-5), loss=['mse','mse','mse','mse'],loss_weights=[2,2,1,2])
    D_model = tf.keras.models.Model(inputs=train_D_model.layers[4].input,outputs=train_D_model.layers[4].output)

    G_model = G_net(inshape)
    G_model.load_weights('models/G_weights_pre.h5')
    gan_model = COMPILE(inshape,D_model,G_model)
    gan_model.summary()
    gan_model.layers[3].trainable=False#VGG-net
    gan_model.layers[5].trainable=False#D—net
    gan_model.compile(Adam(lr=2e-5), loss=[style_loss,color_loss,'mse'],loss_weights=[1,3,5])
    
    D_model.summary()

#img_list
realdir = 'dataset/train_photo'
aniedir = 'dataset/Target/style'
smoothdir = 'dataset/Target/smooth'
real_list = os.listdir(realdir)
anie_list = os.listdir(aniedir)
smooth_list = os.listdir(smoothdir)

#output
gen_logit_mask = np.ones((batch_size, h//4, w//4 ,1))
g_out_mask1 = np.zeros((batch_size, h//4, w//4 ,1536))
g_out_mask2 = np.zeros((batch_size, h, w ,6))

d_out_mask1 = np.ones((batch_size, h//4, w//4 ,1))
d_out_mask2 = np.zeros((batch_size, h//4, w//4 ,1))
d_out_mask3 = np.zeros((batch_size, h//4, w//4 ,1))
d_out_mask4 = np.zeros((batch_size, h//4, w//4 ,1))

#g_input
real_img = np.zeros((batch_size, h, w ,3))
anie_gray = np.zeros((batch_size, h, w ,3))
anie_img = np.zeros((batch_size, h, w ,3))
anie_smooth = np.zeros((batch_size, h, w ,3))

gen_img_logit = np.zeros((batch_size, h//4, w//4 ,1))

for epoch in range(epochs):
    for i in range(0,len(anie_list)-5,batch_size):
        start_time = time.time()
        real_list = shuffle(real_list)

        #img data load
        for j in range(batch_size):
            real_path = realdir + '/' + real_list[i+j]
            real_src = cv.imread(real_path)

            anie_path = aniedir + '/' + anie_list[i+j]
            anie_src = cv.imread(anie_path)
            anie_src_gray = cv.cvtColor(anie_src, cv.COLOR_BGR2GRAY)

            anie_src = anie_src.astype(np.float64)
                        
            gray_src = cv.merge([anie_src_gray,anie_src_gray,anie_src_gray])

            smooth_path = smoothdir + '/' + smooth_list[i+j]
            smooth_src = cv.imread(smooth_path)

            #load to [-1,1]
            real_src = 1/127.5 * real_src -1
            anie_src = 1/127.5 * anie_src -1
            gray_src = 1/127.5 * gray_src -1
            smooth_src = 1/127.5 * smooth_src -1
            real_img[j,...] = real_src
            anie_img[j,...] = anie_src
            anie_gray[j,...] = gray_src
            anie_smooth[j,...] = smooth_src
        
        
        #  Train D
        D_model.trainable=True
        gen_img = G_model.predict(real_img)
        d_loss = train_D_model.train_on_batch([anie_img,anie_gray,anie_smooth,gen_img], [d_out_mask1,d_out_mask2,d_out_mask3,d_out_mask4])
        # ---------------------
        
        #  Train G
        D_model.trainable=False
        g_loss = gan_model.train_on_batch([real_img,anie_gray], [g_out_mask1,g_out_mask2, gen_logit_mask])

        # -----------------
        elapsed_time = time.time() - start_time

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] time: %s" 
        % (epoch,epochs,i, len(anie_list),d_loss[0],g_loss[0],elapsed_time))

        
    D_model.save_weights('models/D_weights_' + str(epoch) + '.h5')
    G_model.save_weights('models/G_weights_' + str(epoch) + '.h5')