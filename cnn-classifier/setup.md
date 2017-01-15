# Make Keras to work with TensorFlow
see requirements.txt
```
h5py==2.6.0
Keras==1.2.0
numpy==1.11.3
protobuf==3.1.0.post1
PyYAML==3.12
scikit-learn==0.18.1
scipy==0.18.1
six==1.10.0
tensorflow-gpu==0.12.1 // это не ставить, команда для установки ниже
Theano==0.8.2
```
Also you need:
```
pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-linux_x86_64.whl
```
Additionally add to .bashrc
```
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
export CUDA_HOME=/usr/local/cuda
```

Configure Keras for TensorFlow backend
```
$ cat .keras/keras.json 
{ 
    "image_dim_ordering": "th",
    "epsilon": 1e-07, 
    "floatx": "float32", 
    "backend": "tensorflow"
}
```
