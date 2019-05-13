# EverybodyDanceNow-Temporal-Smoothing-FaceGAN


Created by Liangjian Chen.

I analysied the video on Windows 10 and train the model on Ubuntu 16.04

## Reference:
Part of the project is inherited from:

[nyoki-mtl](https://github.com/nyoki-mtl) pytorch-EverybodyDanceNow

[Lotayou](https://github.com/Lotayou) everybody_dance_now_pytorch

## OpenPose release　and video pre-processing

### Download OpenPose Source and target video:

* Source video can be download from [1.video](https://drive.google.com/file/d/1AxY1toJOmyy1cuqzCNJtsNAhWF1LBLHG/view?usp=sharing)

* Target video can be download from [2.video](https://drive.google.com/file/d/162f8GdNAga3gFBygA1CBZt6XPG1KqQxx/view?usp=sharing)

* Download OpenPose release from [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases)

### Pre-processing in Windows
Paste 1.mp4 and 2.mp4 under the folder of Open pose release and
Run 

`./build/examples/openpose/openpose.bin --video 1.mp4 --write_json anno_1/ --display 0 --render_pose 0 --face --hand`  

and 

`./build/examples/openpose/openpose.bin --video 2.mp4 --write_json anno_2/ --display 0 --render_pose 0 --face --hand`  

to get the pose annotation from video

### Pre-trained models

* Download vgg19-dcbb9e9d.pth.crdownload [here](https://drive.google.com/file/d/1JG-pLXkPmyx3o4L33rG5WMJKMoOjlXhl/view?usp=sharing) and put it in `./src/pix2pixHD/models/`  <br>

* Download pre-trained vgg_16 for face enhancement [here](https://drive.google.com/file/d/180WgIzh0aV1Aayl_b1X7mIhVhDUcW3b1/view?usp=sharing) and put in `./face_enhancer/`

## Full process

This step is completed in Ubuntu 16.04

### Pose2vid network

#### Make target pictures
* put the `2.mp4` and `anno_2` in `./data/1` and rename it to `video.mp4` and `anno`

* Run `python target.py --name 1`

#### Make source pictures
* put the `1.mp4` and `anno_1` in `./data/1` and rename it to `video.mp4` and `anno`

* Run `python source.py --name 1 --which_train 2`

* `source.py` rescales the label and save it in `./data/2/test/` 

#### Train and use pose2vid network
* Run `python train_pose2vid_temporal.py` and check loss and full training process in `./checkpoints/`

* If you break the traning and want to continue last training, set `load_pretrain = './checkpoints/target/` in `./src/config/train_opt.py`

* Run `transfer.py` and get results in `./result`

#### Face enhancement network

![](/result/pic2.png)
#### Train and use face enhancement network
* Run `python ./Face_GAN/prepare_Data` and check the results in `./Face_GAN/data/`.
* Run `python ./Face_GAN/train_face_gan.py` train face enhancer and run`./Face_GAN/Inference.py` to gain results <br>


#### Gain results
* Run `python transfer_temporal.py` and make result pictures to video


## TODO

- Pose estimation
    - [x] Pose
    - [x] Face
    - [x] Hand
- [x] pix2pixHD
- [x] FaceGAN
- [X] Temporal smoothing

## Environments
Ubuntu 16.04 <br>
Python 3.6.5 <br>
Pytorch 0.4.1  <br>
OpenCV 3.4.4  <br>


