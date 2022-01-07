# third-year-ENSISA-project
Human actions classifier with Graph Convolutional Networks

Study of graphical convolutional neural networks with a real application : worked on physiotherapeutic movement classification for rehabilitation with graph convolutional networks (GCN) using NTU-RGB Dataset which contains different data on the joints of the human skeleton, captured by Kinect. The purpose of this project was to be a spatio-temporal classification.

The file ntu_dataset_gcn.py contains classes to create DGL dataset and to design the struture of graph convolutional network with method for training and validation.

Packages used :
  - Deep Graph Library
  - PyTorch
  - Pandas
  - Numpy
  - Networkx

References :
 - NTU-RGB Dataset : https://rose1.ntu.edu.sg/dataset/actionRecognition/
 - NTU-RGB+D GitHub : https://github.com/shahroudy/NTURGB-D

The NTU RGB+D (or NTU RGB+D 120) Action Recognition Dataset made available by the ROSE Lab at the Nanyang Technological University, Singapore :
  - Amir Shahroudy, Jun Liu, Tian-Tsong Ng, Gang Wang, "NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis", IEEE Conference on Computer Vision and Pattern               Recognition (CVPR), 2016 [PDF].
  - Jun Liu, Amir Shahroudy, Mauricio Perez, Gang Wang, Ling-Yu Duan, Alex C. Kot, "NTU RGB+D 120: A Large-Scale Benchmark for 3D Human Activity Understanding", IEEE Transactions     on Pattern Analysis and Machine Intelligence (TPAMI), 2019. [PDF].
