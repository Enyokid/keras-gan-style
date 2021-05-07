import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import cv2 as cv
import tensorflow as tf
tf.config.experimental.set_memory_growth = True
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from net import *
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
import time



test_path = "test_img"
save_path = 'results'

data_list = os.listdir(test_path)

for p in range(len(data_list)):
    img_path = test_path + os.sep + data_list[p]
    rawsrc = image.load_img(img_path)
    rawsrc = image.img_to_array(rawsrc)
    h, w = rawsrc.shape[:2]
    h = h //16 * 16
    w = w //16 * 16

    inshape = (h, w, 3)

    #compile model
    train_G_model = train_G_net(inshape)
    G_model = tf.keras.models.Model(inputs=train_G_model.layers[0].input,outputs=train_G_model.layers[1].output)
    G_model.load_weights('models/G_weights.h5')
    #读取图片
    src = cv.imread(img_path)
    rawsrc = cv.resize(src,(w,h))
    src = 1/127.5 * rawsrc - 1
    X = np.expand_dims(src,axis=0)
    gen_img_logit = np.ones((1, h//4, w//4 ,1))

    #推理
    gen_img = G_model.predict([X,X,gen_img_logit])
    print(np.shape(gen_img))

    gen_img = gen_img[0,:,:,:]

    Y = (gen_img + 1)*127.5
    print(np.shape(Y))
    Y = Y.astype(np.uint8)
    cv.imwrite(save_path + os.sep + data_list[p],Y)


