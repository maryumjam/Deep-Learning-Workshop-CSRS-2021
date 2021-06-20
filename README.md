# Deep-Learning-Workshop-CSRS-2021
URL: https://crss-sct.ca/conferences/csrs-2021/workshops-2021/#three <br>
Instructors: Dr. Connie Ko, Maryam Jameela, Andrew Chadwick <br>
Date : June 21, 2021 

<H1> Description </H1>
Deep learning is one of the fastest growing areas of machine learning and has been successfully applied in many applications including speech recognition, object detection and classification for autonomous navigation. In the remote sensing community, deep learning has been applied to a variety of data types (e.g. spectral, hyperspectral, LiDAR )for the applications of image classification, anomaly detection, terrain surface classification, object detection and many more (Zhu et al., 2017).

There are two key parts to deep learning: training, and inference. Training is a process of inputting a large amount of labelled data into the deep learning model. During the training phase, the model understands the data characteristics automatically and memorizes these characteristics. Unlike traditional machine learning where features are designed by the user, deep learning algorithms automatically learn features from the training data. During the inference phase, deep learning algorithms apply features learned during the training phase to make predictions on new data. Because these processes are automated, deep learning is often referred to as an end-to-end solution for tasks that traditionally required user supervision.

The purpose of this workshop is to introduce deep learning approaches through theory, examples, and experiences. The workshop will include two sessions, the first will focus on image datasets (2D) and the second will focus on LiDAR datasets (3D). Each session will include a 40-minute lecture and 80-minute demonstration, and/or hands-on exercise.

The first session will provide an overview of deep learning, explaining some of the important theories and terminologies in convolutional neural networks. Following this, we will review Mask R-CCN (He et al., 2017), which will be the focus of this session’s demonstration. This demonstration will provide a walk-through of adapting Mask R-CNN to the task of individual tree crown delineation. The second session will cover a lecture that discuss LiDAR point-cloud processing and especially precoding. We will be reviewing PointNet (Qi et al., 2017) and Point Pillars (Lang et al., 2018) for the purpose of understanding the demonstration of single tree detection in LiDAR point cloud
<H2> Agenda Workshop </H2>
<H3>Morning</H3>
10:00 am to 12:00 pm Mountain Daylight Time (GMT-6)
Lecture: Deep Learning in 2D <br>
Presenter: Connie Ko <br>
Demo: Mask RCNN for individual tree crown delineation <br>
Presenter: Andrew Chadwick <br>

<H3> Afternoon </H3>
13:00 pm to 15:00 pm Mountain Daylight Time (GMT-6)
Lecture: Deep Learning in 3D <br>
Presenter: Connie Ko<br>
Demo: PointNet and 3D-Unet<br>
Presenter: Maryam Jameela<br>
<H2> System Configuration </H2>
<H3>Operating System </H3>
Ubuntu 18.04
<H3>Environment </H3>
Compatability Table <a href="https://www.tensorflow.org/install/source#linux">Here</a> <br>
CUDA 10.0: Click <a href="https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal">Here</a> to Download <br>
CuDNN 7.4: <a href="https://developer.nvidia.com/cudnn">Download</a> <br>
  Step 1:  Register a nvidia developer account and download cudnn <br>
  Step 2:  Check where your cuda installation is. <br>
  Step 3:  Copy the files:<br>

      $ cd folder/extracted/contents 
      $ sudo cp include/cudnn.h /usr/local/cuda/include 
      $ sudo cp lib64/libcudnn* /usr/local/cuda/lib64 
      $ sudo chmod a+r /usr/local/cuda/lib64/libcudnn* 

Pycharm 2021 <a href="https://www.jetbrains.com/pycharm/download/#section=linux"> Download </a> <br>
<H3>Install</H3>
Tensorflow 1.14<br>

			$ pip3 install tensorflow==1.14.0
     	$ pip3 install tensorflow-gpu==1.14.0 
		 
Numpy: 

     $ pip3 install numpy 
H5py: 

     $ pip3 install h5py 


<H2>Demo 2D Deep Learning </H2>
<H3> Mask RCNN for individual tree crown delineation </H3>
Fork from <a href="https://github.com/charlesq34/pointnet">Original Repository</a> by Authors of MaskRCNN



<H2>Deep Classification and Semantic Segmentation of 3D Point cloud </H2>
<H3> PointNet for 3D Point cloud Classification </H3>
Fork from <a href="https://github.com/charlesq34/pointnet">Original Repository</a> by Authors of Pointnet: Deep Learning on Point Sets for 3d Classification and Segmentation [Qi et al., 2017]


### Usage
```
Configuration 
ModelNet40 70-30 Split Training and Test 
Block size= 2m2xInf 
Batch size=32x2048x6 
1 GPU GTX 2080
Training Time: 12 hours (250 Epochs) 
Inference: 30 seconds
```	

To train: <br>

     $ python3 train.py 

Inspect at tensorboard <br>

     $ tensorboard --logdir log 

To test <br>

    $ python evaluate.py --visu 
		
### 3D U-Net Semantic Segemntation 
```
|__models
   |__
|__preprocessing
   |__data
         |__voxels
|__utils
   |__provider.py
train.py
test.py
```			 


### Usage 
```
Configuration <br>
Incremental K-Cross Validation Training 
Voxel size= 1m^3 
Input grid size = 32x32x32 and Batch size=4 grids 
1 GPU GTX 2080 
Training Time: 48 hours 
Inference: 2-3 mins
```	
To train: <br>
  
	$ python3 train.py
	
	
To test: <br>
 
    $ python3 test.py

		
### Citation 
```
@article{DBLP:journals/corr/HeGDG17,
  author    = {Kaiming He and
               Georgia Gkioxari and
               Piotr Doll{\'{a}}r and
               Ross B. Girshick},
  title     = {Mask {R-CNN}},
  journal   = {CoRR},
  volume    = {abs/1703.06870},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.06870},
  archivePrefix = {arXiv},
  eprint    = {1703.06870},
  timestamp = {Mon, 13 Aug 2018 16:46:36 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/HeGDG17.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}


@article{qi2016pointnet,
  title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
  author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
  journal={arXiv preprint arXiv:1612.00593},
  year={2016}
}


@article{DBLP:journals/corr/CicekALBR16,
  author    = {{\"{O}}zg{\"{u}}n {\c{C}}i{\c{c}}ek and
               Ahmed Abdulkadir and
               Soeren S. Lienkamp and
               Thomas Brox and
               Olaf Ronneberger},
  title     = {3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation},
  journal   = {CoRR},
  volume    = {abs/1606.06650},
  year      = {2016},
  url       = {http://arxiv.org/abs/1606.06650},
  archivePrefix = {arXiv},
  eprint    = {1606.06650},
  timestamp = {Mon, 13 Aug 2018 16:47:29 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/CicekALBR16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

```			


### Acknowledgements
```
3DMMAI (https://gunhosohn.me/3dmmai/  ; https://gunhosohn.me/2020/09/06/3d-mobile-mapping-ai-project/) 
Lab Introduction https://gunhosohn.me/team/
Dr. Connie Ko, Dr. Gunho Sohn, Maryam Jameela, Muhammad Kamran, Sunghwoon Yoo
Our lab will lead the development of deep learning-based 

Object Detection 
Noise Filtering 
Terrain Filtering
Semantic Segmentation 
Building Footprint Extraction
LIDAR and visual SLAM development 
Transmission Network Modeling

Andrew Chadwick
https://irsslab.forestry.ubc.ca/people/graduate-students/




```






