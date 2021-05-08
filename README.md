# keras-gan-style
A simple code for build and train a GAN model to change photo's style.

This code is an unofficial implementation of the paper [AnimeGAN: A Novel Lightweight GAN for Photo Animation](https://www.researchgate.net/publication/341634830_AnimeGAN_A_Novel_Lightweight_GAN_for_Photo_Animation), the code is not a complete reproduction but more like a baseline with some issue I still working on it. I tried to construct the GAN structure with my own understanding and experiment, acctually it can be use to transfer images in any styles depending on the training and target data that been used. however, effective training techniques and dataset production are very important for good results.

Requirements
-----
python 3.6  

tensorflow-gpu>=2.0 (ubuntu, GPU 2080Ti , cuda 9.0, cudnn 7.1.3)  

opencv  

numpy  


Usage
-----
1. Prepare dataset, the directory structure including train_photo, Target style and smooth images.It is recommended to use at least 4000 training images and 2000 style images.
2. run edge_smooth.py to generate the edage smooth images of the style images.
3. run train-init.py to initialize the weight of the G_net.
4. run train-gan.py to train the GAN model.
5. After training, run test.py to convert the image style in the test_img catalog.

Results
-----
Using different datasets and different parameter to train the models, you can get different image styles. Here is a simple example
![image](https://github.com/Enyokid/keras-gan-style/blob/main/results/sample.jpg)

Acknowledgment
-----
Thanks to the contributors of [AnimeGAN](https://github.com/TachibanaYoshino/AnimeGAN). This work is based on it.
