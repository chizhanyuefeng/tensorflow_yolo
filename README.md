# yolo_v1(tensorflow)

## 1.Test yolo using trained model
- [Download the tiny yolo model](https://drive.google.com/file/d/0B2JbaJSrWLpza0FtQlc3ejhMTTA/view?usp=sharing)
- Put the model to ./weights/YOLO_tiny.ckpt
- Run


    python run.py --download_model=True

## 2.Train and test you data

### Train the tiny yolo

    python run.py --train=True

### Test 

    python run.py --test_img=IMG_PATH

## 3.Notice
现在只能train一张图片，因为并没有完善从其他数据集获取batch来进行训练。
有待完善。 2018.6.7记
