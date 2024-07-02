# VOC to TFRecord Converter
This script converts the data from the VOC dataset to TFRecord, which is required for training TensorFlow models.

## Clone the repository
```sh
git clone https://github.com/sssarana/voc-to-tfrecord.git
```
## Install requirements and TensorFlow's Object Detection API
```sh
pip install -r requirements.txt
```
```sh
git clone https://github.com/tensorflow/models.git
cd models/research
```
Install the Protobuf compiler and compile the Protobuf files:
```sh
apt-get install -y protobuf-compiler
protoc object_detection/protos/*.proto --python_out=.
```
Add models to python path, you might need to add path to research folder too.
```sh
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
Install the package using setup.py file provided:
```sh
pip install .
```

## Download the dataset
Create a directory for the VOC dataset and navigate to it:
```sh
cd /data
mkdir voc
cd voc
```
Download and extract the VOC dataset:
```sh
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
```
## Adjust paths
Find the following paths in the code and replace them according to the needs of your project and depending on where the dataset is located:
```python
data_dir = '/home/object_detection/data/voc/VOCdevkit/VOC2012'
train_output_path = '/home/object_detection/data/voc/voc_train.record'
val_output_path = '/home/object_detection/data/voc/voc_val.record'
```
## Run the script
Run this in your project directory:
```sh
python3 convert_voc_to_tfrecord.py
```
This will generate the TFRecord files needed for training.