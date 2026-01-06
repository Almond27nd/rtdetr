import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
import torch

if __name__ == '__main__':
    model = RTDETR('BFAM+HSFPN+LKSA.yaml')
    torch.cuda.empty_cache()

    # model.load('') # loading pretrain weights
    model.train(data=r'myData.yaml',
                cache=False,
                imgsz=640,
                epochs=72,
                batch=4,
                workers=0,
                device='0',
                #resume='/tmp/pycharm_project_191/runs/train/exp9/weights/last.pt', # last.pt path
                project='runs/train',
                name='BFAM+HSFPN+LKSA',
                # amp=Tre
                )