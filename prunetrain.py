from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型

    model = YOLO(r'../prune.pt')  # 加载训练模型 (推荐)

    # 指定训练参数开始训练
    model.train(
        data="codshuju.yaml",
        workers=8,
        epochs=300,
        imgsz=640,
        batch=50,
        amp=True,
        optimize=True,
        profile=True,
        cos_lr=True,
        pretrained=False,
        device='0',
    )


