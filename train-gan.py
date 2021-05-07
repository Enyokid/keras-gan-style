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
#compile model
with tf.device('/cpu:0'):
    train_D_model = train_D_net(inshape)
    train_D_model.compile(Adam(lr=0.00016), loss=train_D_loss)

    train_G_model = train_G_net(inshape)
    train_G_model.compile(Adam(lr=0.00008), loss=[train_G_loss,Gan_loss])
    #freeze the vgg map
    train_G_model.layers[6].trainable=False
    train_G_model.summary()
    #get the G_net
    G_model = tf.keras.models.Model(inputs=train_G_model.layers[0].input,outputs=train_G_model.layers[1].output)
    G_model.load_weights('models/G_weights_pre.h5')

#img_list
realdir = 'dataset/train_photo'
aniedir = 'dataset/Target/style'
smoothdir = 'dataset/Target/smooth'
real_list = os.listdir(realdir)
anie_list = os.listdir(aniedir)
smooth_list = os.listdir(smoothdir)

#output
gen_logit_mask = np.ones((batch_size, h//4, w//4 ,1))
g_out_mask = np.zeros((batch_size, h//4, w//4 ,1536))
d_out_mask = np.zeros((batch_size, h//4, w//4 ,4))

#g_input
real_img = np.zeros((batch_size, h, w ,3))
anie_gray = np.zeros((batch_size, h, w ,3))
anie_img = np.zeros((batch_size, h, w ,3))
anie_smooth = np.zeros((batch_size, h, w ,3))

gen_img_logit = np.zeros((batch_size, h//4, w//4 ,1))

for epoch in range(epochs):
    for i in range(0,len(real_list),batch_size):
        start_time = time.time()
        real_list = shuffle(real_list)
        anie_list = shuffle(anie_list)
        smooth_list = shuffle(smooth_list)
        #img data load
        for j in range(batch_size):
            real_path = realdir + '/' + real_list[j]
            real_src = cv.imread(real_path)

            anie_path = aniedir + '/' + anie_list[j]
            anie_src = cv.imread(anie_path)
            anie_src_gray = cv.cvtColor(anie_src, cv.COLOR_BGR2GRAY)

            anie_src = anie_src.astype(np.float64)
            
            
            gray_src = cv.merge([anie_src_gray,anie_src_gray,anie_src_gray])

            smooth_path = smoothdir + '/' + smooth_list[j]
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
        gen_img = G_model.predict(real_img)
        d_loss = train_D_model.train_on_batch([anie_img,anie_gray,anie_smooth,gen_img], d_out_mask)
        # ---------------------
        
        #  Train G        
        all_logit = train_D_model.predict([anie_img,anie_gray,anie_smooth,gen_img])
        gen_img_logit = np.expand_dims(all_logit[..., 3],axis=3)
        g_loss = train_G_model.train_on_batch([real_img,anie_gray,gen_img_logit], [g_out_mask, gen_logit_mask])
        # -----------------
        elapsed_time = time.time() - start_time

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] time: %s" 
        % (epoch,epochs,i, len(real_list),d_loss,g_loss[0],elapsed_time))

        
    if epoch % 5 == 0:
        train_D_model.save_weights('models/D_weights_' + str(epoch) + '.h5')
        G_model.save_weights('models/G_weights_' + str(epoch) + '.h5')