from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# 加入以下的代码是为了避免错误，做的事就是指定一张显卡
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 只可见第一个CUDA设备
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# CUDA内核启动将同步执行（即阻塞模式）
# 主机提交任务给设备后，主机将会阻塞，等待设备将所提交的任务完成，并将控制权交回主机，
# 然后主机继续执行主机的程序。
# 这种模式的作用是保证程序按照预期的顺序执行。在同步的情况下，当主机向设备提交任务后，
# 设备开始执行任务，并立刻将控制权交回主机，所以主机将不会阻塞，而是直接继续执行主机的程序。
# 这种模式使得主机的程序能够按照顺序执行，保证程序的正确性和稳定性。


# region 一.准备工作
# region 1.1参数设置

train_on_gpu = torch.cuda.is_available()
# 判断cuda是否存在，存在就用gpu训练
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
# 不存在就用cpu
# 线程数
# 线程数是指计算机程序中能够处理的线程数目。这个数目通常受到计算机硬件的限制，如CPU的核数或逻辑处理器的数量。
# 在多线程环境中，线程数可以影响程序的性能和响应速度
num_workers = 0
# 采样批次为5，即每次取五个图片去训练
batch_size = 5
# 验证集比例
valid_size = 0.2
# 测试集、训练集、验证集。
# test、train
# train包括训练集和验证集，验证集就是每轮训练完后用训练集当中一定比例的图片，验证这一轮次的效果
# 测试集就是将几轮训练完后，用损失最小的那一轮权重再做一次输出，计算精确率
# endregion

# 1.2数据集的处理
# 1.2.1数据增强
# 使用Compose函数，接受一个由各种转换组成的列表，并将它们按顺序串联在一起，创建一个转换管道。
transform = transforms.Compose([
    transforms.Resize([224, 224]),  # 用resize函数统一图片大小
    transforms.RandomVerticalFlip(p=0.2),
    # 依据概率p对图片进行垂直翻转（增加数据多样性）
    # 这样帮助模型更好地理解图像中的内容，提高模型的识别准确率。
    transforms.ToTensor(),  # 将输入图像转换为pytorch的张量
    # 张量是一个可以包含任意数量维度的数组，通常将图像看作是一个三维张量，
    # 三个维度分别是高度、宽度和颜色通道
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # 使用指定的均值和标准差，对图像的每个颜色通道进行标准化
    # 标准化的过程就是将每个通道的值减去对应的均值和标准差
])

# 1.2.2选择数据
train_data = datasets.ImageFolder(r'D:\resnet18\cat_dog data\train',
                                  transform=transform)
test_data = datasets.ImageFolder(r'D:\resnet18\cat_dog data\test',
                                 transform=transform)
# 指定数据预处理的方式

# 1.2.3确定验证集随机采样
num_train = len(train_data) # 计算训练数据集的大小
indices = list(range(num_train))  # 创建一个列表，放训练集图片的索引，也就确定验证集图片个数
np.random.shuffle(indices)  # 将索引列表随机打乱顺序
split = int(np.floor(valid_size * num_train))  # 计算了验证集的大小=验证集比例*训练集数量
train_idx, valid_idx = indices[split:], indices[:split]
# 这样就把数据集划分成了两个部分，前split个用于验证，剩下用于训练
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
# 这两步创建了两个采样器，训练时可以随机选择一批样本

# 1.2.4确定数据集
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,
                                           num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler,
                                           num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
# 创建了数据加载器，从已经下载好的数据集中，每次随机采样，取五张，加载数据的线程数量为0


# 1.2.5确定输出分类
classes = ['cat', 'dog']


# region 二.确定网络
class BasicBlock(nn.Module):
    # 定义了一个名为BasicBlock的类，它继承自PyTorch的nn.Module类，表示它是一个可训练的神经网络模块
    def __init__(self, in_channels, out_channels, stride=[1, 1], padding=1) -> None:
        # 定义初始化函数，接收输入通道数in_channels，输出通道数out_channels，步长stride和填充padding作为参数
        super(BasicBlock, self).__init__()  # 调用nn.Module
        # 残差部分
        # 一。layer部分不改变特征图尺寸
        # 定义nn.Sequential模块，包括两个卷积层和两个归一层
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=padding, bias=False),
            # 2D卷积层，输入通道数为in_channels，输出通道数为out_channels，卷积核大小为3，步长为stride[0]，
            # 填充为padding，不使用偏置项（bias）
            nn.BatchNorm2d(out_channels),
            # 这是一个2D批量归一化层，输出通道数为out_channels。它可以在每个通道上计算均值和方差，对数据进行归一化
            nn.ReLU(inplace=True),  # 这是一个ReLU激活函数层，inplace=True表示原地操作，节省内存开销
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding, bias=False),  # 尺寸不变
            # 同上，卷积层参数与前一个卷积层相同，但输出的通道数与输入相同。
            nn.BatchNorm2d(out_channels)
            # 与前一个批量归一化层相同
        )


        # 二。 shortcut 部分
        '''
        如果输入输出通道不同，或者卷积步长做了改变，那就用if中的模型，否则不变。这里肯定是输入输出通道要改变的
        这里的shortcut模块就是将原来的特征图通过1*1卷积提高通道数
        '''
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        # 定义了一个名为 shortcut 的变量，它是一个空的神经网络序列（nn.Sequential）。
        # 通俗来说就是把nn.Sequential 当成容器，可以容纳多个神经网络模块，并按顺序连接它们。
        if stride[0] != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 设置卷积核为1 进行升降维
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
                # 一个卷积一个归一
            )

    def forward(self, x):
        # 定义了模型的前向传播函数。当你在训练过程中输入一个样本时，这个函数就会被调用。x是输入的特征图
        out = self.layer(x)
        # 将输入特征图x通过一个叫做self.layer的函数进行运算，然后将结果赋值给变量out。
        out += self.shortcut(x)
        # 将输入特征图x通过另一个叫做self.shortcut的函数进行运算，然后将结果加到out上。
        out = F.relu(out)
        # 使用了PyTorch的函数库中的ReLU函数对out进行非线性激活。
        # ReLU是一种常用的非线性激活函数，对于负数输出为0，对于正数输出为该数值本身。
        return out  # 处理后的特征图out返回


class ResNet18(nn.Module):
    def __init__(self, BasicBlock, num_classes=1) -> None:
        # 定义模型的构造函数。它接受两个参数，一个是BasicBlock类（表示残差块），另一个是num_classes（表示分类类别数，默认值为1）
        # 这里注意num_class是1，因为最后我们要输出一个0-1之间的数，小于0.5的作为一类，大于0.5的又作为一类
        super(ResNet18, self).__init__()  # 调用父类（nn.Module）的构造函数
        self.in_channels = 64  # 定义输入通道数为64

        # 第一层作为单独的 因为没有残差块。224*224*3
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            # 定义第一层卷积操作。使用7x7的卷积核，步长为2，填充为3，输入通道数为3，输出通道数为64。
            # 224*224*3【（224-7+2*3）/2+1=112.5】-112*112*3这里为啥是112了？
            nn.BatchNorm2d(64),  # 对应上一层的所有特征图做归一化
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # 最大池操作
            # 112*112*64-56*56*64  （112-3）/2 +1 =55.5=56
        )
        # 以下都是使用_make_layer方法构建由残差块组成的层，每一层由多个残差块组成的。每一层的输出大小都会被调整到上一层的输出大小的一半
        # （步长为2的卷积层之后），但最后一层例外，其输出大小为512
        # conv2_x
        self.conv2 = self._make_layer(BasicBlock, 64, [[1, 1], [1, 1]])  # 56*56*64-56*56*64
        # 输入通道数为64，输出通道数为64。

        # conv3_x
        self.conv3 = self._make_layer(BasicBlock, 128, [[2, 1], [1, 1]])  # 56*56*64-56*56*128
        # 输入通道数为64，输出通道数为128

        # conv4_x
        self.conv4 = self._make_layer(BasicBlock, 256, [[2, 1], [1, 1]])  # 56*56*128-56*56*256
        # 输入通道数为128，输出通道数为256

        # conv5_x
        self.conv5 = self._make_layer(BasicBlock, 512, [[2, 1], [1, 1]])  # 56*56*256-56*56*512
        # 输入通道数为256，输出通道数为512

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 使用自适应平均池化层对特征图进行全局平均池化，输出大小为1x1
        self.fc = nn.Linear(512, num_classes)  # 定义一个全连接层（线性层），输入大小为512，输出大小为num_classes
        # 最终输出的大小为num_classes，表示分类类别数。
    '''通过此函数方便的构造残差模块，
          bock就是BlasicBlock。
          输入通道是递接上一层通道的，
          输出通道自行确定。
          strides中写入几个stride的列表此layer就有几个残差模块，另外stride的值都选[1,1]，比较方便,尺寸就不会变了
    '''

    # 使用 nn.Sequential 函数将 layers 列表中的所有元素（即所有的残差块）连接起来，并返回这个顺序
    def _make_layer(self, block, out_channels, strides):
        # 定义一个名为 _make_layer 的方法，它接受三个参数：block（一个残差块），
        # out_channels（输出通道数）和strides（一个包含两个元素的元组，表示卷积层的步长）
        layers = []
        # 初始化一个空列表 layers，该列表将用于存储构建的层。
        for stride in strides:
            # 遍历 strides 中的每一个元素（步长）
            layers.append(block(self.in_channels, out_channels, stride))
            # 使用当前的 in_channels、out_channels 和 stride 参数调用 block（一个残差块）。并将这个残差块添加到 layers 列表中
            self.in_channels = out_channels  # 更新 self.in_channels 的值为 out_channels，为后续的层设置输入通道数
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)  # 调用模型中的conv1卷积层对输入x进行操作，结果保存到out中
        out = self.conv2(out)  # 接着调用模型中的conv2卷积层对上一步的结果进行操作，并保存到out中
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        # out = F.avg_pool2d(out,7)
        out = self.avgpool(out)  # 对上述卷积操作的结果使用平均池化操作，这里使用的是2x2的平均池化。
        out = out.reshape(x.shape[0], -1)
        # 将池化后的结果重塑为二维数组，其中第一维是batch size，第二维是所有特征值的集合。-1表示自动计算该维度的大小。
        out = torch.sigmoid(self.fc(out))  # 这里加入sigmoid以保证在0-1之间。而且最好调用torch.sigmod而非F.sigmod
        # 首先通过一个全连接层（self.fc）将特征映射到一维，然后使用Sigmoid函数将输出映射到0-1之间
        return out
# 这段代码中没有使用激活函数（如ReLU），这可能会导致在训练过程中出现梯度消失或爆炸的问题。
# 在实际使用中，通常会在卷积操作后添加激活函数以增加模型的非线性表达能力。

model = ResNet18(BasicBlock=BasicBlock)
print(model)
# endregion

# region 三. 训练-train/val

# 3.1设置训练参数
model = model.cuda()  # 模型还在cpu要放到gpu里，加速
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# 创建一个优化器。优化器是用于调整模型参数的算法，以使模型的性能在训练过程中不断改善。
# 这里使用的是随机梯度下降（SGD）作为优化器，并设置学习率为0.01
n_epochs = 8  # 训练轮次
valid_loss_min = np.Inf  # np.inf表示正无穷大，-np.inf表示负无穷大
# 设定一个变量valid_loss_min，并将其初始值设为无穷大（np.Inf）。这个变量通常用来跟踪验证集上的最小损失。
# 在每次验证之后，如果发现新的更小的验证损失，就更新这个变量

# 3.2开始循环训练
for epoch in range(1, n_epochs + 1):  # epoch从1到9重复执行下列代码8次

    # 设置损失，初始化训练和验证的损失为0.0
    train_loss = 0.0
    valid_loss = 0.0

    ###################
    #      训练模块     #
    ###################
    model.train()
    for data, target in train_loader:
        # 模型被设置为训练模式，然后对训练数据集进行迭代

        # 在gpu上去训练
        if train_on_gpu:  # 在gpu上运行
            data, target = data.cuda(), target.cuda().float().unsqueeze(1)

        optimizer.zero_grad()  # 将梯度清零 优化器清零梯度：
# 在进行训练之前，优化器（optimizer）会清零已经存在的梯度，这样在进行新的训练步骤时，不会受到之前梯度的影响
        output = model(data)  # 得到输出
        # 如果，模型中用的sigmoid，那输出必须都在0-1，否则就有问题
        loss = F.binary_cross_entropy(output,
                                      target)
        loss.backward()  # 求方向传播得梯度
        optimizer.step()  # 反向传播
        # 这两行代码用于调整模型的参数以最小化损失。backward函数计算损失对每个参数的梯度，step函数使用这些梯度来更新参数
        train_loss += loss.item() * data.size(0)
        # 这行代码将当前epoch的损失累加到train_loss变量中。loss.item()获取损失的值（排除梯度），data.size(0)得到批量大小
        # .item()是为了获得loss中不需要反向传播得内容，
        # data.size(0)为20因为data的shape是[20,3,32,32]
    ######################
    #       验证模块       #
    ######################
    model.eval()  # 下面的不做反向传播，即为验证部分
    for data, target in valid_loader:
        # 在gpu上计算
        if train_on_gpu:
            data, target = data.cuda(), target.cuda().float().unsqueeze(1)

        output = model(data)
        # 这行代码使用输入的数据（data）来运行模型，并将输出存储在变量output中。因为模型已经被设置为评估模式，所以这不会导致梯度的计算
        loss = F.binary_cross_entropy(output, target)  # 计算loss
        valid_loss += loss.item() * data.size(0)  # 计算验证损失
        # 这两行代码计算模型预测的二进制交叉熵损失（binary_cross_entropy）。这是用于二元分类问题的常见损失函数。
        # 然后，将损失累加到valid_loss变量中，以计算整个验证集的损失。loss.item()返回损失的值（不包括梯度），而data.size(0)得到批量大小


    train_loss = train_loss / len(train_loader.dataset)  # 计算平均损失
    valid_loss = valid_loss / len(valid_loader.dataset)
    # 这两行代码计算训练集和验证集的平均损失。这是通过将各自的总损失除以数据集中的样本数量来实现的。
    # 这样做的目的是为了消除数据大小差异的影响，使得不同大小的数据集之间的比较更为公平

    # 输出tain_loss和valid_loss
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

    # 保存模型权重
    if valid_loss <= valid_loss_min:
        # valid_loss_min = np.Inf，第一轮验证一下是否是无穷不是无穷则保存此valid_loss
        # 第二轮，验证一下此轮损失是否比上一轮小，小的话则保存新的这个损失并且保存新的权重
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        torch.save(model.state_dict(), 'res18.pt')
        # 这行代码将模型的当前状态（state_dict()）保存到名为res18.pt的文件中。这是PyTorch库的一种常见用法，用于保存模型权重和参数
        valid_loss_min = valid_loss
        # 最后，将当前验证损失赋值给valid_loss_min，以便在下一次迭代中使用。这是为了确保下一次迭代时，新的最小损失已经更新
        # 总的来说，这段代码的目的是在训练过程中定期检查验证集上的损失，并在损失下降时保存模型
# endregion

# region 四.测试
# 4.1设置参数
test_loss = 0.0
# 设置好test_loss并认定为一个精度值
class_correct = list(0. for i in range(2))  # [0.0, 0.0]
class_total = list(0. for i in range(2))  # [0.0, 0.0]
# 分别创建了两个列表，每个列表都包含两个0.0。

# 4.2开始测试
model.eval()  # model不反向传播
# 将模型设置为评估模式（model.eval()）。在评估模式下，模型不会进行反向传播（即不会更新权重），这是为了确保测试的准确性
state_dict = torch.load('res18.pt')
model.load_state_dict(state_dict)
# 加载模型权重，从名为'res18.pt'的文件中加载状态字典，并将其加载到模型中。这是使用PyTorch的一种常见方法来加载预训练的模型权重

for data, target in test_loader:  # 按照test_loader设置好的batch_size进行采样
    # 开始一个循环，遍历测试数据集的每个批次。test_loader是一个数据加载器，用于按批次提供测试数据

    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    output = model(data)
    target1 = target.float().unsqueeze(1)
    # 将数据和目标变量转移到GPU上（如果训练时使用GPU的话），然后使用模型对数据进行预测，
    # 并将目标变量转换为浮点数并增加一个额外的维度
    loss = F.binary_cross_entropy(output, target1)
    # 计算模型预测和目标之间的二元交叉熵损失。这是用于二元分类问题的常见损失函数
    test_loss += loss.item() * data.size(0)
    # 将当前批次的损失累加到test_loss变量中，乘以批次的大小（即样本数）。这是为了计算整个测试集的总损失

    pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in output]).cuda()
    # 通过此行代码求得索引值，这里预测得到的小于0.5的shiwei0，大于0.5的视为1。以此分类
    # 对模型的输出进行阈值操作（将值大于或等于0.5的预测为1，否则为0），并将结果存储在变量pred中。
    # 这个变量可能被用来跟踪模型对每个类别的预测

    # 获得预测结果的正确与否
    correct_tensor = pred.eq(target.data.view_as(pred))
    # 这行代码比较模型的预测结果pred和真实的标签target，返回一个布尔型的Tensor，其中每个元素表示对应的预测是否正确
    # x.eq(y)判断x和y的值是否相等，作比较以后输出新的tensor 这里的view_as和view一样都是调整格式只不过view是将target调整成和pred一样的格式，这里其实没变他俩格式都是Tensor（20，）
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(
        correct_tensor.cpu().numpy())
    # 将比较结果从Tensor转化为Numpy数组，同时删除可能存在的额外维度（np.squeeze的作用）。
    # 如果模型在GPU上训练，则首先将Tensor移动到CPU上（.cpu()），然后再转化成Numpy数组
    # 改成np格式[ True  True  True  True  True]
    # 计算准确值
    for i in range(batch_size):  # 这里再按照batch_size的值遍历，遍历每个批次中的样本
        label = target.data[i]   # 获取第i个样本的真实类别
        # 第一轮就是第一个图片的标签是3 tensor(3, device='cuda:0')
        # label=tensor([0.], device='cuda:0')
        class_correct[label] += correct[
            i].item()  # 增加第i个样本预测正确的次数。注意这里用label作为索引，因此label必须是整数或者只包含一个整数的张量
        # 【40，46】#注意这里头的label应该是：1.1个整数2.是张量的话只能是包含一个数的张量。
        # 所以这里用label做索引，labela选的target的数据就不能去增加维数，也不能变为精度型
        class_total[label] += 1  # 【50，50】
        # 增加第i个样本的总数

test_loss = test_loss / len(test_loader.dataset)  # 计算平均损失
print('Test Loss: {:.6f}\n'.format(test_loss))
# endregion

# region 五.获取精确度
# 通过得到的class_correct[i]和class_total[i]计算精度
for i in range(2):  # 遍历所有两类
    if class_total[i] > 0:  # 如果某类的总数大于0，则计算并打印该类的准确率
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[
                                                 i])))  # Test Accuracy of airplane: 46% (469/1000)这里np.sum这个sum无所谓，对于此项目每个类别的预测结果就是个数不是列表，这个sum应该是为了和其他模型的结果契合
    # 如果某类的总数大于0，则计算并打印该类的准确率
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))  # 算个总的预测精度
# endregion
