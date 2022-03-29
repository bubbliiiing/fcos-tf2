import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

class BBoxUtility(object):
    def __init__(self, num_classes, nms_thresh=0.45, top_k=300):
        self.num_classes    = num_classes
        self._nms_thresh    = nms_thresh
        self._top_k         = top_k

    def bbox_iou(self, b1, b2):
        b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        
        inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                    np.maximum(inter_rect_y2 - inter_rect_y1, 0)
        
        area_b1 = (b1_x2-b1_x1)*(b1_y2-b1_y1)
        area_b2 = (b2_x2-b2_x1)*(b2_y2-b2_y1)
        
        iou = inter_area/np.maximum((area_b1+area_b2-inter_area),1e-6)
        return iou

    def efficientdet_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            #-----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            #-----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset  = (input_shape - new_shape)/2./input_shape
            scale   = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes
    
    def decode_box(self, predictions, image_shape, input_shape, letterbox_image, confidence=0.5):
        #---------------------------------------#
        #   centerness置信度
        #---------------------------------------#
        classifications = predictions[0]
        #---------------------------------------#
        #   centerness置信度
        #---------------------------------------#
        centerness      = predictions[1]
        #---------------------------------------#
        #   网络预测的结果
        #---------------------------------------#
        boxes           = predictions[2]

        results     = [None for _ in range(len(boxes))]
        #----------------------------------------------------------------------------------------------------------------#
        #   对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次
        #----------------------------------------------------------------------------------------------------------------#
        for i in range(len(boxes)):
            #--------------------------------#
            #   利用回归结果对先验框进行解码
            #--------------------------------#
            decode_bbox = boxes[i]

            class_conf  = np.expand_dims(np.max(classifications[i], 1), -1)
            class_pred  = np.expand_dims(np.argmax(classifications[i], 1), -1)
            class_conf  = np.sqrt(class_conf * centerness[i])
            #--------------------------------#
            #   判断置信度是否大于门限要求
            #--------------------------------#
            conf_mask       = (class_conf >= confidence)[:, 0]

            #--------------------------------#
            #   将预测结果进行堆叠
            #--------------------------------#
            detections      = np.concatenate((decode_bbox[conf_mask], class_conf[conf_mask], class_pred[conf_mask]), 1)
            unique_labels   = np.unique(detections[:,-1])

            #-------------------------------------------------------------------#
            #   对种类进行循环，
            #   非极大抑制的作用是筛选出一定区域内属于同一种类得分最大的框，
            #   对种类进行循环可以帮助我们对每一个类分别进行非极大抑制。
            #-------------------------------------------------------------------#
            for c in unique_labels:
                #------------------------------------------#
                #   获得某一类得分筛选后全部的预测结果
                #------------------------------------------#
                detections_class = detections[detections[:, -1] == c]
                #------------------------------------------#
                #   使用官方自带的非极大抑制会速度更快一些！
                #------------------------------------------#
                idx             = tf.image.non_max_suppression(detections_class[:, :4], detections_class[:, 4], self._top_k, iou_threshold=self._nms_thresh).numpy()
                max_detections  = detections_class[idx]
                # #------------------------------------------#
                # #   非官方的实现部分
                # #   获得某一类得分筛选后全部的预测结果
                # #------------------------------------------#
                # detections_class    = detections[detections[:, -1] == c]
                # scores              = detections_class[:, 4]
                # #------------------------------------------#
                # #   根据得分对该种类进行从大到小排序。
                # #------------------------------------------#
                # arg_sort            = np.argsort(scores)[::-1]
                # detections_class    = detections_class[arg_sort]
                # max_detections = []
                # while np.shape(detections_class)[0]>0:
                #     #-------------------------------------------------------------------------------------#
                #     #   每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
                #     #-------------------------------------------------------------------------------------#
                #     max_detections.append(detections_class[0])
                #     if len(detections_class) == 1:
                #         break
                #     ious             = self.bbox_iou(max_detections[-1], detections_class[1:])
                #     detections_class = detections_class[1:][ious < self._nms_thresh]
                results[i] = max_detections if results[i] is None else np.concatenate((results[i], max_detections), axis = 0)

            if results[i] is not None:
                results[i][:, [0, 2]] = results[i][:, [0, 2]] / input_shape[1]
                results[i][:, [1, 3]] = results[i][:, [1, 3]] / input_shape[0]
                results[i] = np.array(results[i])
                box_xy, box_wh = (results[i][:, 0:2] + results[i][:, 2:4])/2, results[i][:, 2:4] - results[i][:, 0:2]
                results[i][:, :4] = self.efficientdet_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

        return results

if __name__ == "__main__":
    def generate_meshgrid(inputs, strides):
        feature_shapes          = [np.shape(feature)[1:3] for feature in inputs]
        locations_per_feature   = []
        #--------------------------------------#
        #   对网格进行循环
        #--------------------------------------#
        for feature_shape, stride in zip(feature_shapes, strides):
            h = feature_shape[0]
            w = feature_shape[1]

            shifts_x = np.arange(0, w * stride, step=stride, dtype=np.float32)
            shifts_y = np.arange(0, h * stride, step=stride, dtype=np.float32)

            #--------------------------------------#
            #   创建网格
            #--------------------------------------#
            shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
            shift_x     = np.reshape(shift_x, (-1,))
            shift_y     = np.reshape(shift_y, (-1,))
            locations   = np.stack((shift_x, shift_y), axis=1) + stride // 2
            locations_per_feature.append(locations)
            
        #--------------------------------------#
        #   网格堆叠
        #--------------------------------------#
        locations = np.concatenate(locations_per_feature, axis=0)
        locations = np.tile(np.expand_dims(locations, axis=0), (np.shape(inputs[0])[0], 1, 1))
        return locations
    
    def decode_box(inputs):
        locations, regression = inputs
        x1 = locations[:, :, 0] - regression[:, :, 0]
        y1 = locations[:, :, 1] - regression[:, :, 1]
        x2 = locations[:, :, 0] + regression[:, :, 2]
        y2 = locations[:, :, 1] + regression[:, :, 3]
        # (batch_size, num_locations, 4)
        bboxes = np.stack([x1, y1, x2, y2], axis=-1)
        return bboxes
    
    input_shape = [640, 640]
    strides     = [8, 16, 32, 64, 128]
    
    features    = []
    regression  = []
    for i in range(len(strides)):
        features.append(np.random.randn(1, int(input_shape[0] / strides[i]), int(input_shape[1] / strides[i]), 256))
        regression.append(np.random.uniform(1, 1.25, [1, int(input_shape[0] / strides[i]) * int(input_shape[1] / strides[i]), 4]))
        
    locations   = generate_meshgrid(features, strides)
    regression  = np.exp(np.concatenate(regression, 1) * 4)
    bboxes      = decode_box([locations, regression])
        
    import matplotlib.pyplot as plt
    
    last_all = int(input_shape[0] / strides[-1] * input_shape[1] / strides[-1])
    
    fig = plt.figure()
    ax = fig.add_subplot(121)
    plt.ylim(-300,900)
    plt.xlim(-300,900)
    plt.scatter(locations[0][-last_all:, 0], locations[0][-last_all:, 1], color='black')
    plt.scatter(locations[0][-last_all:-last_all+3, 0], locations[0][-last_all:-last_all+3, 1], color='r')
    plt.gca().invert_yaxis()
    
    ax = fig.add_subplot(122)
    plt.ylim(-300,900)
    plt.xlim(-300,900)
    plt.scatter(locations[0][-last_all:, 0], locations[0][-last_all:, 1], color='black')
    
    plot_bboxes = np.array(bboxes[0, -last_all:, :], np.int32)
    box_widths  = plot_bboxes[:, 2] - plot_bboxes[:, 0]
    box_heights = plot_bboxes[:, 3] - plot_bboxes[:, 1]
    for i in range(3):
        rect = plt.Rectangle([plot_bboxes[i, 0], plot_bboxes[i, 1]], box_widths[i], box_heights[i], color="r", fill=False)
        ax.add_patch(rect)
    plt.gca().invert_yaxis()
    plt.show()