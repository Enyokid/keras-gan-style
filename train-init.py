import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import cv2 as cv
import tensorflow as tf
tf.config.experimental.set_memory_growth = True
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from net import *
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle

def inited_G_generator(data_path,h,w,batch_size,val):
    data_list = os.listdir(data_path)    
    while True:       
        X = np.zeros((batch_size, h, w ,3))
        Y = np.zeros((batch_size, h//8, w//8 ,1024))
        #
        if val:
            get_list = data_list[int(len(data_list)*0.8):]
        else:
            get_list = data_list[:int(len(data_list)*0.8)]
        load_list = shuffle(get_list)

        for i in range(batch_size):
            img_path = data_path + '/' + load_list[i]
            rawsrc = cv.imread(img_path)
            #处理与写入
            src = 1/255.0 * rawsrc
            X[i,...] = src
        yield (X,Y)

h,w = 256, 256
inshape = (h, w, 3)

inited_G_model = inited_G_net(inshape)
#freez the vgg model
print(len(inited_G_model.layers))
inited_G_model.layers[78].trainable=False
inited_G_model.summary()
inited_G_model.compile(Adam(lr=0.0001), loss=inited_G_loss)


#train
data_path = 'dataset/train_photo'
nb_epochs = 1
bh_size = 4
steps_per_epoch = int(len(os.listdir(data_path))/bh_size)
inited_G_model.fit_generator(generator=inited_G_generator(data_path,h,w,batch_size=bh_size,val=False),
                                        epochs=nb_epochs, 
                                        steps_per_epoch=steps_per_epoch, validation_steps=steps_per_epoch//4,
                                        validation_data=inited_G_generator(data_path,h,w,batch_size=bh_size,val=True),
                                        verbose=1)
G_model = tf.keras.models.Model(inputs=inited_G_model.input,outputs=inited_G_model.layers[75].output)
G_model.summary()
G_model.save_weights('models/G_weights_pre.h5')