1. Constraint training
约束训练主要是对模型进行BN层进行L1正则化，因此需要在trainer.py文件夹下添加BN层进行L1约束的代码，trainer.py文件位于ultralytics/yolo/engine文件夹下，添加的具体位置在327行，添加的具体内容如下：

# Backward
self.scaler.scale(self.loss).backward()

# ========== 新增 ==========
l1_lambda = 1e-2 * (1 - 0.9 * epoch / self.epochs)
for k, m in self.model.named_modules():
    if isinstance(m, nn.BatchNorm2d):
        m.weight.grad.data.add_(l1_lambda * torch.sign(m.weight.data))
        m.bias.grad.data.add_(1e-2 * torch.sign(m.bias.data))
# ========== 新增 ==========

# Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
if ni - last_opt_step >= self.accumulate:
    self.optimizer_step()
    last_opt_step = ni

将代码修改好后，按照之前提到的Pretrain中将VOC.yaml和default.yaml修改好，点击train.py开始训练即可。
————————————————
版权声明：本文为CSDN博主「爱听歌的周童鞋」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_40672115/article/details/130175558


2. Prune
我们拿到约束训练的模型后就可以开始剪枝了，开工👨‍🏭，本次剪枝使用的是约束训练中的last.pt模型(我们不使用best.pt，通过result.csv你会发现mAP最高的模型在第一个epoch，主要是因为模型在COCO数据集上的预训练泛化性比较强，所以开始的mAP很高，这显然是不真实的)，我们在根目录ultralytics-main文件夹下创建一个prune.py文件，用于我们的剪枝任务，同时将约束训练中的last.pt模型放到根目录下
————————————————
版权声明：本文为CSDN博主「爱听歌的周童鞋」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_40672115/article/details/130175558
我们通过上述代码可以完成剪枝工作并将剪枝好的模型进行保存，用于finetune，有以下几点说明：

在本次剪枝中我们利用factor变量来控制剪枝的保留率

我们用来剪枝的模型一定是约束训练的模型，即对BN层加上L1正则化后训练的模型

约束训练后的b.min().item值非常小，接近于0或者等于0，可以依据此来判断加载的模型是否正确

我们可以选择将yolo.train()取消注释，在剪枝完成直接进入微调训练，博主在这里选择先保存剪枝模型

我们可以选择yolo.export()取消注释，将剪枝完成后的模型导出为ONNX，查看对应的大小和channels是否发生改变，以此确认我们完成了剪枝

yolo.val()用于进行模型验证，建议取消注释进行相关验证，之前梁老师说yolo.val()验证的mAP值完全等于0是不正常的，需要检查下剪枝过程是否存在错误，最好是有一个值，哪怕非常小，博主剪枝后进行验证的结果如下图所示，可以看到mAP值真的是惨不忍睹(🤣)，所以后续需要finetune模型来恢复我们的精度
————————————————
版权声明：本文为CSDN博主「爱听歌的周童鞋」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_40672115/article/details/130175558
3. finetune
拿到剪枝的模型后，我们需要先做两件事情

1.切记！！！在进行finetune之前需要将我们在trainer.py为BN层添加的L1正则化的代码注释掉(也就是我们在第2节添加的内容)
2.切记！！！剪枝后不要从yaml导入结构。如果我们直接将剪枝后的模型prune.pt放到v8/detect目录下修改default.yaml文件，然后点击train.py是会存在问题的，此时模型结构是通过yolov8.yaml加载的，而我们并没有修改yaml文件，因此finetune的模型其实并不是剪枝的模型
因此，正常finetune训练的步骤如下：

1.在yolo/engine/trainer.py中注释掉为BN层加L1正则化的代码

2.修改yolo/engine/model.py代码，让其不要从yaml导入网络结构，具体修改内容是BaseTrainer类中的setup_model方法中，代码大概在443行左右，新增一行代码即可，如下所示

# ========== yolo/engine/trainer.py的443行 ==========
self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)

# ========== 新增该行代码 ==========
self.model = weights

return ckpt
1
2
3
4
5
6
7
3.将剪枝完保存的模型放到yolo/v8/detect文件夹下

4.修改default.yaml文件，主要修改model为prune.pt即剪枝完的模型，具体修改如下：

model: prune.pt  # path to model file, i.e. yolov8n.pt, yolov8n.yaml
1
5.点击train.py开始训练即可，博主在这里选择的是微调50个epoch，大家根据自己的实际情况来，尽可能的多finetune几个epoch

微调50个epoch后模型的表现如下图所示，可以看到精度恢复得还可以，可以训练更多epoch使其精度更加稳定。
————————————————
版权声明：本文为CSDN博主「爱听歌的周童鞋」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_40672115/article/details/130175558
4. 剪枝后不从yaml导入结构(补充细节)
剪枝完成后的模型该如何正确的加载并训练呢？这里再提供另外一种方案，供大家借鉴参考，主要修改两个地方.

修改1：修改网络加载的地方，让其不要从yaml导入结构

具体修改代码在 ultralytics/yolo/engine/model.py
具体位置在 YOLO 类的 train 方法中，大概是 363 行的位置
修改代码如下：
# ===== ultralytics/yolo/engine/model.py 363行=====

# ===== 原代码 =====
if not overrides.get('resume'):  # manually set model only if not resuming
    self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
    self.model = self.trainer.model
            
# ===== 修改后代码 =====
if not overrides.get('resume'):  # manually set model only if not resuming
    # self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
    # self.model = self.trainer.model
    self.trainer.model = self.model.train()
1
2
3
4
5
6
7
8
9
10
11
12
修改2：自己新增一个my_train.py的训练代码

在主目录下新建一个 my_train.py 文件用于训练，该文件内容非常简单，如下所示：

from ultralytics import YOLO


if __name__ == "__main__":
    yolo = YOLO("prune.pt") # 加载剪枝后的模型

    yolo.train(data="D:/YOLO/yolov8-prune/ultralytics/datasets/VOC.yaml", epochs=50, amp=False, workers=8) # 训练
1
2
3
4
5
6
7
其中有以下几点值得注意：

加载剪枝后的模型，请修改为你自己的剪枝模型名称
关于训练参数的指定
data 表示之前训练时的 VOC.yaml 文件的绝对路径
epochs 迭代次数，根据自己实际需求设置
amp 混合精度设置为False
workers 工作核心数，根据自己的硬件设置，值越大训练越快
其它超参数博主并未设置，有需求可自行修改
剪枝模型微调训练

完成上述两点修改后，点击 my_train.py 即可开始剪枝模型的微调训练，训练后的文件会保存在 runs/detect/train 中，在训练之前请确保已经将 trainer.py 为 BN 层添加的 L1 正则化的代码注释掉！！！
————————————————
版权声明：本文为CSDN博主「爱听歌的周童鞋」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_40672115/article/details/130175558