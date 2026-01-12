"""
RT-DETR道路缺陷检测改进模型训练脚本
适用于硕士毕业论文实验

作者：[你的名字]
日期：2026-01
"""

import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
import torch
import argparse

# 改进模型配置列表
IMPROVED_MODELS = {
    # 方案1: DCNv2可变形卷积 + EMA注意力 - 适合不规则形状缺陷
    "dcnv2_ema": "rtdetr-resnet18-DCNv2-EMA.yaml",
    
    # 方案2: FasterBlock轻量化 + BiFPN双向融合 - 轻量化且多尺度增强
    "faster_bifpn": "rtdetr-resnet18-FasterBlock-BiFPN.yaml",
    
    # 方案3: LSKA大核注意力 + SDI + DySample - 大感受野+小目标增强
    "lska_sdi_dy": "rtdetr-resnet18-LSKA-SDI-DySample.yaml",
    
    # 方案4: DCNv2 + TripletAttention + BiFPN - 综合改进方案
    "dcnv2_triplet_bifpn": "rtdetr-resnet18-DCNv2-Triplet-BiFPN.yaml",
    
    # 方案5: EMA + CARAFE + RepNCSPELAN4 - 特征增强方案
    "ema_carafe_elan": "rtdetr-resnet18-EMA-CARAFE-RepELAN.yaml",
    
    # 方案6: DLKA可变形大核注意力 + ADown + SDI - 针对裂缝优化
    "dlka_adown_sdi": "rtdetr-resnet18-DLKA-ADown-SDI.yaml",
    
    # 基线模型
    "baseline": "rtdetr-resnet18.yaml",
}

def train_model(model_key, data_yaml="myData.yaml", epochs=72, batch_size=4, 
                imgsz=640, device='0', resume=None):
    """
    训练指定的改进模型
    
    Args:
        model_key: 模型键名，参见 IMPROVED_MODELS
        data_yaml: 数据集配置文件
        epochs: 训练轮数
        batch_size: 批次大小
        imgsz: 输入图像尺寸
        device: GPU设备ID
        resume: 恢复训练的权重路径
    """
    
    if model_key not in IMPROVED_MODELS:
        print(f"可用的模型: {list(IMPROVED_MODELS.keys())}")
        raise ValueError(f"未知的模型: {model_key}")
    
    model_yaml = IMPROVED_MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"训练模型: {model_key}")
    print(f"配置文件: {model_yaml}")
    print(f"{'='*60}\n")
    
    model = RTDETR(model_yaml)
    torch.cuda.empty_cache()
    
    train_args = {
        'data': data_yaml,
        'cache': False,
        'imgsz': imgsz,
        'epochs': epochs,
        'batch': batch_size,
        'workers': 0,
        'device': device,
        'project': 'runs/train',
        'name': model_key,
    }
    
    if resume:
        train_args['resume'] = resume
    
    model.train(**train_args)
    
    return model


def compare_all_models(data_yaml="myData.yaml", epochs=72, batch_size=4):
    """
    训练所有改进模型用于对比实验
    """
    for model_key in IMPROVED_MODELS.keys():
        print(f"\n开始训练: {model_key}")
        try:
            train_model(model_key, data_yaml, epochs, batch_size)
        except Exception as e:
            print(f"训练 {model_key} 失败: {e}")
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RT-DETR道路缺陷检测改进模型训练')
    parser.add_argument('--model', type=str, default='dcnv2_ema', 
                        choices=list(IMPROVED_MODELS.keys()),
                        help='选择要训练的模型')
    parser.add_argument('--data', type=str, default='myData.yaml', help='数据集配置')
    parser.add_argument('--epochs', type=int, default=72, help='训练轮数')
    parser.add_argument('--batch', type=int, default=4, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='0', help='GPU设备')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的权重路径')
    parser.add_argument('--all', action='store_true', help='训练所有模型进行对比')
    
    args = parser.parse_args()
    
    if args.all:
        compare_all_models(args.data, args.epochs, args.batch)
    else:
        train_model(args.model, args.data, args.epochs, args.batch, 
                   args.imgsz, args.device, args.resume)

