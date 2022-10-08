# -*- coding: utf-8 -*-

import numpy as np
import time

import paddle

from darknet import ConvBNLayer, DarkNet53_conv_body

# 定义生成YOLO-V3预测输出的模块
# 也就是教程中图x所示的由ci生成ri和ti的过程
# 从骨干网络输出特征图C0得到跟预测相关的特征图P0
class YoloDetectionBlock(paddle.nn.Layer):
    # define YOLOv3 detection head
    # 使用多层卷积和BN提取特征
    def __init__(self,ch_in,ch_out,is_test=True):
        super(YoloDetectionBlock, self).__init__()

        assert ch_out % 2 == 0, \
            "channel {} cannot be divided by 2".format(ch_out)

        self.conv0 = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv1 = ConvBNLayer(
            ch_in=ch_out,
            ch_out=ch_out*2,
            kernel_size=3,
            stride=1,
            padding=1)
        self.conv2 = ConvBNLayer(
            ch_in=ch_out*2,
            ch_out=ch_out,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv3 = ConvBNLayer(
            ch_in=ch_out,
            ch_out=ch_out*2,
            kernel_size=3,
            stride=1,
            padding=1)
        self.route = ConvBNLayer(
            ch_in=ch_out*2,
            ch_out=ch_out,
            kernel_size=1,
            stride=1,
            padding=0)
        self.tip = ConvBNLayer(
            ch_in=ch_out,
            ch_out=ch_out*2,
            kernel_size=3,
            stride=1,
            padding=1)
    def forward(self, inputs):
        out = self.conv0(inputs)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        route = self.route(out)
        tip = self.tip(route)
        return route, tip

# 定义上采样模块
class Upsample(paddle.nn.Layer):
    def __init__(self, scale=2):
        super(Upsample,self).__init__()
        self.scale = scale

    def forward(self, inputs):
        # get dynamic upsample output shape
        shape_nchw = paddle.shape(inputs)
        shape_hw = paddle.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = paddle.cast(shape_hw, dtype='int32')
        out_shape = in_shape * self.scale
        out_shape.stop_gradient = True

        # reisze by actual_shape
        out = paddle.nn.functional.interpolate(
            x=inputs, scale_factor=self.scale, mode="NEAREST")
        return out

# 定义YOLOv3模型
class YOLOv3(paddle.nn.Layer):
    def __init__(self, num_classes=7):
        super(YOLOv3,self).__init__()

        self.num_classes = num_classes
        # 提取图像特征的骨干代码
        self.block = DarkNet53_conv_body()
        self.block_outputs = []
        self.yolo_blocks = []
        self.route_blocks_2 = []
        # 生成3个层级的特征图P0, P1, P2
        for i in range(3):
            # 添加从ci生成ri和ti的模块
            yolo_block = self.add_sublayer(
                "yolo_detecton_block_%d" % (i),
                YoloDetectionBlock(
                                   ch_in=512//(2**i)*2 if i==0 else 512//(2**i)*2 + 512//(2**i),
                                   ch_out = 512//(2**i)))
            self.yolo_blocks.append(yolo_block)

            num_filters = 3 * (self.num_classes + 5)

            # 添加从ti生成pi的模块，这是一个Conv2D操作，输出通道数为3 * (num_classes + 5)
            block_out = self.add_sublayer(
                "block_out_%d" % (i),
                paddle.nn.Conv2D(in_channels=512//(2**i)*2,
                       out_channels=num_filters,
                       kernel_size=1,
                       stride=1,
                       padding=0,
                       weight_attr=paddle.ParamAttr(
                           initializer=paddle.nn.initializer.Normal(0., 0.02)),
                       bias_attr=paddle.ParamAttr(
                           initializer=paddle.nn.initializer.Constant(0.0),
                           regularizer=paddle.regularizer.L2Decay(0.))))
            self.block_outputs.append(block_out)
            if i < 2:
                # 对ri进行卷积
                route = self.add_sublayer("route2_%d"%i,
                                          ConvBNLayer(ch_in=512//(2**i),
                                                      ch_out=256//(2**i),
                                                      kernel_size=1,
                                                      stride=1,
                                                      padding=0))
                self.route_blocks_2.append(route)
            # 将ri放大以便跟c_{i+1}保持同样的尺寸
            self.upsample = Upsample()
    def forward(self, inputs):
        outputs = []
        blocks = self.block(inputs)
        for i, block in enumerate(blocks):
            if i > 0:
                # 将r_{i-1}经过卷积和上采样之后得到特征图，与这一级的ci进行拼接
                block = paddle.concat([route, block], axis=1)
            # 从ci生成ti和ri
            route, tip = self.yolo_blocks[i](block)
            # 从ti生成pi
            block_out = self.block_outputs[i](tip)
            # 将pi放入列表
            outputs.append(block_out)

            if i < 2:
                # 对ri进行卷积调整通道数
                route = self.route_blocks_2[i](route)
                # 对ri进行放大，使其尺寸和c_{i+1}保持一致
                route = self.upsample(route)

        return outputs

    def get_loss(self, outputs, gtbox, gtlabel, gtscore=None,
                 anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
                 anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 ignore_thresh=0.7,
                 use_label_smooth=False):
        """
        使用paddle.vision.ops.yolo_loss，直接计算损失函数，过程更简洁，速度也更快
        """
        self.losses = []
        downsample = 32
        for i, out in enumerate(outputs): # 对三个层级分别求损失函数
            anchor_mask_i = anchor_masks[i]
            loss = paddle.vision.ops.yolo_loss(
                    x=out,  # out是P0, P1, P2中的一个
                    gt_box=gtbox,  # 真实框坐标
                    gt_label=gtlabel,  # 真实框类别
                    gt_score=gtscore,  # 真实框得分，使用mixup训练技巧时需要，不使用该技巧时直接设置为1，形状与gtlabel相同
                    anchors=anchors,   # 锚框尺寸，包含[w0, h0, w1, h1, ..., w8, h8]共9个锚框的尺寸
                    anchor_mask=anchor_mask_i, # 筛选锚框的mask，例如anchor_mask_i=[3, 4, 5]，将anchors中第3、4、5个锚框挑选出来给该层级使用
                    class_num=self.num_classes, # 分类类别数
                    ignore_thresh=ignore_thresh, # 当预测框与真实框IoU > ignore_thresh，标注objectness = -1
                    downsample_ratio=downsample, # 特征图相对于原图缩小的倍数，例如P0是32， P1是16，P2是8
                    use_label_smooth=False)      # 使用label_smooth训练技巧时会用到，这里没用此技巧，直接设置为False
            self.losses.append(paddle.mean(loss))  #mean对每张图片求和
            downsample = downsample // 2 # 下一级特征图的缩放倍数会减半
        return sum(self.losses) # 对每个层级求和

    def get_pred(self,
                 outputs,
                 im_shape=None,
                 anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
                 anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 valid_thresh = 0.01):
        downsample = 32
        total_boxes = []
        total_scores = []
        for i, out in enumerate(outputs):
            anchor_mask = anchor_masks[i]
            anchors_this_level = []
            for m in anchor_mask:
                anchors_this_level.append(anchors[2 * m])
                anchors_this_level.append(anchors[2 * m + 1])

            boxes, scores = paddle.vision.ops.yolo_box(
                   x=out,
                   img_size=im_shape,
                   anchors=anchors_this_level,
                   class_num=self.num_classes,
                   conf_thresh=valid_thresh,
                   downsample_ratio=downsample,
                   name="yolo_box" + str(i))
            total_boxes.append(boxes)
            total_scores.append(
                        paddle.transpose(
                        scores, perm=[0, 2, 1]))
            downsample = downsample // 2

        yolo_boxes = paddle.concat(total_boxes, axis=1)
        yolo_scores = paddle.concat(total_scores, axis=2)
        return yolo_boxes, yolo_scores

