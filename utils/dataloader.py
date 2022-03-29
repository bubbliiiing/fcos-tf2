import math
from random import shuffle

import cv2
import numpy as np
from PIL import Image
from tensorflow import keras

from utils.utils import cvtColor, preprocess_input


class FcosDatasets(keras.utils.Sequence):
    def __init__(self, annotation_lines, input_shape, batch_size, num_classes, strides=[8,16,32,64,128], limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]], sample_radiu_ratio=1.5, train=True):
        self.annotation_lines   = annotation_lines
        self.length             = len(self.annotation_lines)
        
        self.input_shape        = input_shape
        self.batch_size         = batch_size
        self.num_classes        = num_classes

        self.strides            = strides
        self.limit_range        = limit_range
        self.sample_radiu_ratio = sample_radiu_ratio
        self.train              = train

    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))
        
    def __getitem__(self, index):
        image_data      = []
        box_data        = []
        classes_data    = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
            i           = i % self.length
            #---------------------------------------------------#
            #   训练时进行数据的随机增强
            #   验证时不进行数据的随机增强
            #---------------------------------------------------#
            image, box  = self.get_random_data(self.annotation_lines[i], self.input_shape, random = self.train)
            
            image_data.append(preprocess_input(np.array(image, np.float32)))
            box_data.append(np.array(box[:, :4], dtype=np.float32))
            classes_data.append(np.array(box[:, 4], dtype=np.float32))

        cls_targets, cnt_targets, reg_targets = self.preprocess_true_boxes(np.array(box_data), np.array(classes_data))
        mask_pos    = np.array(cnt_targets > -1, np.float32)
        
        image_data  = np.array(image_data)
        cls_targets = np.concatenate([cls_targets, mask_pos], axis=-1)
        cnt_targets = np.concatenate([cnt_targets, mask_pos], axis=-1)
        reg_targets = np.concatenate([reg_targets, mask_pos], axis=-1)

        # print(cls_targets[0][cnt_targets[0][:,-1]!=1][0:2])
        # print(cnt_targets[0][cnt_targets[0][:,-1]!=1])
        # print(reg_targets[0][cnt_targets[0][:,-1]!=1])
        # for i in range(self.batch_size):
        #     temp_targets = reg_targets[i][reg_targets[i][:,-1]!=1]
        #     temp_cls = np.argmax(cls_targets[i][cnt_targets[i][:,-1]!=1][:,:-1], -1)
        #     print((temp_cls!=0).any())

        #     for target,cls in zip(temp_targets, temp_cls):
        #         draw_1=cv2.rectangle(image_data[i], (target[0],target[1]), (target[2],target[3]), (0,255,0), 2)
        #         draw_1=cv2.putText(draw_1, str(cls), (target[0],target[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        #     cv2.imshow("123", draw_1)
        #     cv2.waitKey(0)

        return image_data, {'classification': cls_targets,
                            'centerness'    : cnt_targets,
                            'regression'    : reg_targets,}

    def generate(self):
        i = 0
        while True:
            image_data      = []
            box_data        = []
            classes_data    = []
            for b in range(self.batch_size):
                if i==0:
                    np.random.shuffle(self.annotation_lines)
                #---------------------------------------------------#
                #   训练时进行数据的随机增强
                #   验证时不进行数据的随机增强
                #---------------------------------------------------#
                image, box  = self.get_random_data(self.annotation_lines[i], self.input_shape, random = self.train)
                
                i           = (i+1) % self.length
                image_data.append(preprocess_input(np.array(image, np.float32)))
                box_data.append(np.array(box[:, :4], dtype=np.float32))
                classes_data.append(np.array(box[:, 4], dtype=np.float32))
                
            cls_targets, cnt_targets, reg_targets = self.preprocess_true_boxes(np.array(box_data), np.array(classes_data))
            mask_pos    = np.array(cnt_targets > -1, np.float32)
            
            image_data  = np.array(image_data)
            cls_targets = np.concatenate([cls_targets, mask_pos], axis=-1)
            cnt_targets = np.concatenate([cnt_targets, mask_pos], axis=-1)
            reg_targets = np.concatenate([reg_targets, mask_pos], axis=-1)
            yield image_data, cls_targets, cnt_targets, reg_targets
            
    def on_epoch_end(self):
        shuffle(self.annotation_lines)

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, max_boxes=100, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line    = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = Image.open(line[0])
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            box_data = np.zeros((max_boxes,5))
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0]  = 0
                box[:, 2][box[:, 2]>w]      = w
                box[:, 3][box[:, 3]>h]      = h
                box_w   = box[:, 2] - box[:, 0]
                box_h   = box[:, 3] - box[:, 1]
                box     = box[np.logical_and(box_w>1, box_h>1)]
                if len(box)>max_boxes: box = box[:max_boxes]
                box_data[:len(box)] = box

            return image_data, box_data
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            if len(box)>max_boxes: box = box[:max_boxes]
            box_data[:len(box)] = box
        
        return image_data, box_data
    
    def get_img_output_length(self, height, width):
        def get_output_length(input_length):
            # input_length += 6
            filter_sizes = [3, 3, 3, 3, 3, 3, 3]
            padding = [1, 1, 1, 1, 1, 1, 1]
            stride = 2
            output_length = []
            for i in range(len(filter_sizes)):
                # input_length = (input_length - filter_size + stride) // stride
                input_length = (input_length + 2*padding[i] - filter_sizes[i]) // stride + 1
                if i>=2:
                    output_length.append(input_length)
            return output_length
        return get_output_length(height), get_output_length(width)

    def _get_grids(self, h, w, stride):
        shifts_x = np.arange(0, w * stride, stride, dtype=np.float32)
        shifts_y = np.arange(0, h * stride, stride, dtype=np.float32)

        shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)

        shift_x = np.reshape(shift_x, [-1])
        shift_y = np.reshape(shift_y, [-1])
        grid    = np.stack([shift_x, shift_y], -1) + stride // 2

        return grid

    def preprocess_true_boxes(self, gt_boxes, classes):
        cls_targets_all_level = []
        cnt_targets_all_level = []
        reg_targets_all_level = []
        output_length = self.get_img_output_length(self.input_shape[0], self.input_shape[1])

        for level in range(len(output_length[0])):
            stride      = self.strides[level]
            limit_range = self.limit_range[level]
            grids       = self._get_grids(output_length[0][level], output_length[1][level], stride)
            
            h_mul_w     = output_length[0][level] * output_length[1][level]

            x           = grids[:, 0]
            y           = grids[:, 1]

            #----------------------------------------------------------------#
            #   左上点、右下点不可以差距很大
            #----------------------------------------------------------------#
            # 求真实框的左上角和右下角相比于特征点的偏移情况
            # [1, h*w, 1] - [batch_size, 1, m] --> [batch_size, h*w, m]
            left_off    = x[None, :, None] - gt_boxes[..., 0][:, None, :]
            top_off     = y[None, :, None] - gt_boxes[..., 1][:, None, :]
            right_off   = gt_boxes[...,2][:, None, :] - x[None, :, None]
            bottom_off  = gt_boxes[...,3][:, None, :] - y[None, :, None]
            # [batch_size, h*w, m, 4]
            ltrb_off    = np.stack([left_off, top_off, right_off, bottom_off],-1)
            
            # 求每个框的面积
            # [batch_size, h*w, m]
            areas       = (ltrb_off[...,0] + ltrb_off[...,2]) * (ltrb_off[...,1] + ltrb_off[...,3])
            
            # [batch_size, h*w, m]
            off_min     = np.min(ltrb_off, -1)
            off_max     = np.max(ltrb_off, -1)

            # 将特征点不落在真实框内的特征点剔除。
            mask_in_gtboxes = off_min > 0
            # 前层特征适合小目标检测，深层特征适合大目标检测。
            mask_in_level   = (off_max > limit_range[0]) & (off_max <= limit_range[1])
            
            #----------------------------------------------------------------#
            #   中心点不可以差距很大
            #----------------------------------------------------------------#
            # 求真实框中心相比于特征点的偏移情况
            radiu       = stride * self.sample_radiu_ratio
            # 计算真实框中心的x轴坐标
            gt_center_x = (gt_boxes[...,0] + gt_boxes[...,2])/2
            # 计算真实框中心的y轴坐标
            gt_center_y = (gt_boxes[...,1] + gt_boxes[...,3])/2
            # [1,h*w,1] - [batch_size,1,m] --> [batch_size,h*w,m]
            c_left_off  = x[None, :, None] - gt_center_x[:, None, :]
            c_top_off   = y[None, :, None] - gt_center_y[:, None, :]
            c_right_off = gt_center_x[:, None, :] - x[None, :, None]
            c_bottom_off= gt_center_y[:, None, :] - y[None, :, None]

            # [batch_size, h*w, m, 4]
            c_ltrb_off  = np.stack([c_left_off, c_top_off, c_right_off, c_bottom_off],-1)
            c_off_max   = np.max(c_ltrb_off,-1)
            mask_center = c_off_max < radiu

            # 为正样本的特征点
            # [batch_size, h*w, m]
            mask_pos    = mask_in_gtboxes & mask_in_level & mask_center

            # 将所有不是正样本的特征点，面积设成max
            # [batch_size, h*w, m]
            areas[~mask_pos] = 99999999
            # 选取该特征点对应面积最小的框
            # [batch_size, h*w]
            areas_min_ind = np.argmin(areas,-1)
            # [batch_size*h*w, 4]
            reg_targets = ltrb_off[np.reshape(np.tile(np.expand_dims(np.arange(self.batch_size),-1), h_mul_w), [-1]),
                                   np.reshape(np.tile(np.arange(h_mul_w), self.batch_size), [-1]),
                                   np.reshape(areas_min_ind, [-1]), :]
            # [batch_size,h*w, 4]
            reg_targets = np.reshape(reg_targets,(self.batch_size,-1,4))
            
            # [batch_size,h*w, m]
            _classes    = np.broadcast_to(classes[:,None,:], np.shape(areas))
            cls_targets = _classes[np.reshape(np.tile(np.expand_dims(np.arange(self.batch_size),-1), h_mul_w), [-1]),
                                   np.reshape(np.tile(np.arange(h_mul_w), self.batch_size), [-1]),
                                   np.reshape(areas_min_ind, [-1])]
            # [batch_size,h*w, 1]
            cls_targets = np.reshape(cls_targets,(self.batch_size,-1,1))
            
            # [batch_size,h*w]
            left_right_min  = np.minimum(reg_targets[..., 0], reg_targets[..., 2])
            left_right_max  = np.maximum(reg_targets[..., 0], reg_targets[..., 2])
            top_bottom_min  = np.minimum(reg_targets[..., 1], reg_targets[..., 3])
            top_bottom_max  = np.maximum(reg_targets[..., 1], reg_targets[..., 3])

            # [batch_size,h*w,1]
            cnt_targets = np.expand_dims(np.sqrt(np.maximum((left_right_min*top_bottom_min)/(left_right_max*top_bottom_max+1e-10),0)), -1)

            # reg_targets[:, :, 0:1] = x[None, :, None] - reg_targets[:, :, 0:1]
            # reg_targets[:, :, 1:2] = y[None, :, None] - reg_targets[:, :, 1:2]
            # reg_targets[:, :, 2:3] = x[None, :, None] + reg_targets[:, :, 2:3]
            # reg_targets[:, :, 3:4] = y[None, :, None] + reg_targets[:, :, 3:4]

            assert np.shape(reg_targets)==(self.batch_size,h_mul_w,4)
            assert np.shape(cls_targets)==(self.batch_size,h_mul_w,1)
            assert np.shape(cnt_targets)==(self.batch_size,h_mul_w,1)

            # process neg grids
            mask_pos_2 = np.sum(mask_pos, -1) >= 1
            assert mask_pos_2.shape  == (self.batch_size,h_mul_w)

            cls_targets[~mask_pos_2] = -1
            cnt_targets[~mask_pos_2] = -1
            reg_targets[~mask_pos_2] = -1
    
            cls_targets= np.float32(np.arange(0, self.num_classes)[None,:] == cls_targets)
            cls_targets_all_level.append(cls_targets)
            cnt_targets_all_level.append(cnt_targets)
            reg_targets_all_level.append(reg_targets)
        
        return np.concatenate(cls_targets_all_level, 1), np.concatenate(cnt_targets_all_level, 1), np.concatenate(reg_targets_all_level, 1)
