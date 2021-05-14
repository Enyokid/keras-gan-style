# keras-gan-style
A simple code for build and train a GAN model to convert photo's style.

This repo is an unofficial implementation of the paper [AnimeGAN: A Novel Lightweight GAN for Photo Animation](https://www.researchgate.net/publication/341634830_AnimeGAN_A_Novel_Lightweight_GAN_for_Photo_Animation).I tried to construct the GAN structure with my own understanding and still working on it. Acctually it can be use to transfer images in any styles depending on the training and target data that been used. however, effective training techniques and dataset production are very important for good results.

Requirements
-----
python 3.6  
tensorflow-gpu>=2.0(ubuntu, GPU 2080Ti)  
opencv  
numpy  


Usage
-----
1. Download or clone this repo .
2. Put the photos in the 'test_img' folder,run test.py, the script will use the pre-trained model to convert the photos into anime style.
3. See the results in the 'results' folder.

Train your own model
-----
1. Prepare dataset, the directory structure including train_photo, Target style and smooth images.It is recommended to use at least 4000 training images and 2000 style images.
2. run edge_smooth.py to generate the edage smooth images of the style images.
3. run train-init.py to initialize the weight of the G_net.
4. run train-gan.py to train the model.

Results
-----
Using different datasets and different parameter to train the models, you can get different image styles.  
The effect of the model is as follows(convert photo to Anime Style).
![image](https://github.com/Enyokid/keras-gan-style/blob/main/test_img/1.jpg)
![image](https://github.com/Enyokid/keras-gan-style/blob/main/results/1.jpg)
![image](https://github.com/Enyokid/keras-gan-style/blob/main/test_img/2.jpg)
![image](https://github.com/Enyokid/keras-gan-style/blob/main/results/2.jpg)
![image](https://github.com/Enyokid/keras-gan-style/blob/main/test_img/3.jpg)
![image](https://github.com/Enyokid/keras-gan-style/blob/main/results/3.jpg)

Acknowledgment
-----
Thanks to the contributors of [AnimeGAN](https://github.com/TachibanaYoshino/AnimeGAN). This work is based on it.
