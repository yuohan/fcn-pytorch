# fcn-pytorch
Pytorch implementation (Fully Convolutional Network)
Trained with VOC and SBD datasets

# Usage
Train
```bash
# Train a fcn model
python train.py --epochs 10 --use-cuda
```
Segment
```bash
python segment.py sample_image.jpg
```

# Result
After 10 epochs, val acc = 0.9119, val_mean_iou: 0.6396.
Trained model and sample result on validation image set are [here](https://drive.google.com/drive/folders/16xC47TUTlz5bK7bJabL2LT9UIshEs9yw?usp=sharing)