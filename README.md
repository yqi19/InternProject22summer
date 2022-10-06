## set up environment 

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
