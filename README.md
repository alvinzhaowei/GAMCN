# GAMCN
This is the Tensorflow Keras implementation of GAMCN in the following paper:\
Jianzhong Qi, Zhuowei Zhao, Egemen Tanin, Tingru Cui, Neema Nassir, and Majid Sarvi. "A Graph and Attentive Multi-Path Convolutional
Network for Traffic Prediction"

## Data
Adjacent matrix files and traffic speed files are included in the data folder. The datasets are from Caltrans Performance Measurement System (PeMS) (https://pems.dot.ca.gov/). The road networks are collected from OpenStreetMap (https://planet.osm.org).

## Model

<p align="center">
  <img src=./model.PNG>
</p>

## Requirements
Python 3.7  <br/>
TensorFlow 2.9.1 <br/>
Numpy

## Run Demo
python demo.py <br/>

The default dataset is PEMS04, if you would like to test other datasets, you can change the dir at line 5 and line 35

## Prepare the raw data

The traffic data is preprocessed into a numpy array whose size is (T,N), where T is the time slot number and N is the node number. The adjacent file is a text file, and for each line, it consists of node1 Id, node2 Id and their normalized distance.

## Citation
If you find this repository useful in your research, please cite the following paper:

```
@inproceedings{qi2022gamcn,
  title={A Graph and Attentive Multi-Path Convolutional
Network for Traffic Prediction},
  author={Jianzhong Qi, Zhuowei Zhao, Egemen Tanin, Tingru Cui, Neema Nassir, and Majid Sarvi},
  booktitle={IEEE Transactions on Knowledge and Data Engineering (TKDE)},
  year={2022}
}
```
