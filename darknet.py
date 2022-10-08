import paddle
import paddle.nn.functional as F
import numpy as np

# YOLO-V3骨干网络结构Darknet53的实现代码

class ConvBNLayer(paddle.nn.Layer):
    def __init__(self, ch_in, ch_out, 
                 kernel_size=3, stride=1, groups=1,
                 padding=0, act="leaky"):
        super(ConvBNLayer, self).__init__()
    
        self.conv = paddle.nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(0., 0.02)),
            bias_attr=False)
    
        self.batch_norm = paddle.nn.BatchNorm2D(
            num_features=ch_out,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(0., 0.02),
                regularizer=paddle.regularizer.L2Decay(0.)),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0),
                regularizer=paddle.regularizer.L2Decay(0.)))
        self.act = act

        
    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == 'leaky':
            out = F.leaky_relu(x=out, negative_slope=0.1)
        return out
    
class DownSample(paddle.nn.Layer):
    # 下采样，图片尺寸减半，具体实现方式是使用stirde=2的卷积
    def __init__(self,
                 ch_in,
                 ch_out,
                 kernel_size=3,
                 stride=2,
                 padding=1):

        super(DownSample, self).__init__()

        self.conv_bn_layer = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.ch_out = ch_out
    def forward(self, inputs):
        out = self.conv_bn_layer(inputs)
        return out

class BasicBlock(paddle.nn.Layer):
    """
    基本残差块的定义，输入x经过两层卷积，然后接第二层卷积的输出和输入x相加
    """
    def __init__(self, ch_in, ch_out):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            kernel_size=1,
            stride=1,
            padding=0
            )
        self.conv2 = ConvBNLayer(
            ch_in=ch_out,
            ch_out=ch_out*2,
            kernel_size=3,
            stride=1,
            padding=1
            )
    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        out = paddle.add(x=inputs, y=conv2)
        return out

     
class LayerWarp(paddle.nn.Layer):
    """
    添加多层残差块，组成Darknet53网络的一个层级
    """
    def __init__(self, ch_in, ch_out, count, is_test=True):
        super(LayerWarp,self).__init__()

        self.basicblock0 = BasicBlock(ch_in,
            ch_out)
        self.res_out_list = []
        for i in range(1, count):
            res_out = self.add_sublayer("basic_block_%d" % (i), # 使用add_sublayer添加子层
                BasicBlock(ch_out*2,
                    ch_out))
            self.res_out_list.append(res_out)

    def forward(self,inputs):
        y = self.basicblock0(inputs)
        for basic_block_i in self.res_out_list:
            y = basic_block_i(y)
        return y

# DarkNet 每组残差块的个数，来自DarkNet的网络结构图
DarkNet_cfg = {53: ([1, 2, 8, 8, 4])}

class DarkNet53_conv_body(paddle.nn.Layer):
    def __init__(self):
        super(DarkNet53_conv_body, self).__init__()
        self.stages = DarkNet_cfg[53]
        self.stages = self.stages[0:5]

        # 第一层卷积
        self.conv0 = ConvBNLayer(
            ch_in=3,
            ch_out=32,
            kernel_size=3,
            stride=1,
            padding=1)

        # 下采样，使用stride=2的卷积来实现
        self.downsample0 = DownSample(
            ch_in=32,
            ch_out=32 * 2)

        # 添加各个层级的实现
        self.darknet53_conv_block_list = []
        self.downsample_list = []
        for i, stage in enumerate(self.stages):
            conv_block = self.add_sublayer(
                "stage_%d" % (i),
                LayerWarp(32*(2**(i+1)),
                32*(2**i),
                stage))
            self.darknet53_conv_block_list.append(conv_block)
        # 两个层级之间使用DownSample将尺寸减半
        for i in range(len(self.stages) - 1):
            downsample = self.add_sublayer(
                "stage_%d_downsample" % i,
                DownSample(ch_in=32*(2**(i+1)),
                    ch_out=32*(2**(i+2))))
            self.downsample_list.append(downsample)

    def forward(self,inputs):
        out = self.conv0(inputs)
        #print("conv1:",out.numpy())
        out = self.downsample0(out)
        #print("dy:",out.numpy())
        blocks = []
        for i, conv_block_i in enumerate(self.darknet53_conv_block_list): #依次将各个层级作用在输入上面
            out = conv_block_i(out)
            blocks.append(out)
            if i < len(self.stages) - 1:
                out = self.downsample_list[i](out)
        return blocks[-1:-4:-1] # 将C0, C1, C2作为返回值


