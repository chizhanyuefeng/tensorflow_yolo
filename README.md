# You Only Look Once: Unified, Real-Time Object Detection
Using tensorflow to complete the yolo train and test


## Notice
Now I have no time to add get_batch() fuctions,so I only use one
picture to train the model. When I'm training the model,loss will be descend and accuracy is coming to 100%.
Training own dataset will be coming.

## Requirenments
- python3
- tensorflow 1.4.0
- pandas
- numpy
- matplotlib

## Test yolo using trained model
- [Download the tiny yolo model](https://drive.google.com/file/d/0B2JbaJSrWLpza0FtQlc3ejhMTTA/view?usp=sharing)
- Put the model to ./weights/YOLO_tiny.ckpt
- Run


    python run.py --download_model=True --test_img=IMG_PATH

## Train and test you data

### 1.Train the tiny yolo

    python run.py --train=True

### 2.Test 

    python run.py --test_img=IMG_PATH

## Notice
现在只能train一张图片，因为并没有完善从其他数据集获取batch来进行训练。
有待完善。 2018.6.7记
