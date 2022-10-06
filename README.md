## set up environment 
for detailed environment, please see `environment.yaml`, but the libraries are too many

you can also install environment by doing:
```python
conda create -n pvt python=3.8 -y
conda activate pvt
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.3.2 tensorboardX six
pip install wandb
```
and clone the repo by:
```python
git clone https://github.com/yqi19/mycode.git
```
##### note: please remember to `pip install wandb` to successfully run the code
##### And this code supports ImageNet-100 only : https://github.com/danielchyeh/ImageNet-100-Pytorch 

## classification
```python
cd classification
```
### run baseline pvt_v2 on imagenet-100
```python
bash dist_train.sh configs/pvt_v2/pvt_v2_b0.py 4 --data-path /home/yu/dataset/imagenet-100
```

### run implementation of mine on imagenet-100
```python
bash dist_train.sh configs/our/window_new/b0_new_3.py 4 --data-path /home/yu/dataset/imagenet-100
```
