from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    # model = YOLO('zhuyiliyolov8n.yaml')  # 从yaml文件加载
    model = YOLO(r'G:\v8\ultralytics-main\runs\detect\train\prune.pt')  # 加载训练模型 (推荐)
    #model = YOLO('yolov8-c2f-ca.yaml').load('yolov8n.pt')  # 从 YAML加载 然后再加载权重

    # 指定训练参数开始训练
    model.val(
        data="codshuju.yaml",
        workers=8,
        device='0',
    )
