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

## Packages
Python 3.7 \\
TensorFlow 2.9.1

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
