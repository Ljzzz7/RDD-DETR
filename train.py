import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/RFB_HL_MDF_CWD.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='dataset/SVRDD.yaml',
                cache=False,
                imgsz=640,
                epochs=260,
                batch=16,
                workers=10,
                device='0',
                # resume='', # last.pt path
                project='runs/train',
                name='exp',
                )
