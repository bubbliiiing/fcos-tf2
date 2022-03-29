import math
from functools import partial

import tensorflow as tf
from tensorflow.keras import backend as K


def focal(alpha=0.25, gamma=2.0):
    def _focal(y_true, y_pred):
        location_state  = y_true[:, :,  -1]
        labels          = y_true[:, :, :-1]
        
        alpha_factor    = K.ones_like(labels) * alpha
        alpha_factor    = tf.where(K.equal(labels, 1), alpha_factor, 1 - alpha_factor)

        focal_weight    = tf.where(K.equal(labels, 1), 1 - y_pred, y_pred)
        focal_weight    = alpha_factor * focal_weight ** gamma
        cls_loss        = focal_weight * K.binary_crossentropy(labels, y_pred)

        normalizer      = tf.where(K.equal(location_state, 1))
        normalizer      = K.cast(K.shape(normalizer)[0], K.floatx())
        normalizer      = K.maximum(K.cast_to_floatx(1.0), normalizer)
        return K.sum(cls_loss)/normalizer
    return _focal

def iou():
    def _iou(y_true, y_pred):
        location_state  = y_true[:, :, -1]
        indices         = tf.where(K.equal(location_state, 1))

        y_regr_pred         = tf.gather_nd(y_pred, indices)
        y_regr_true         = tf.gather_nd(y_true, indices)
        y_regr_true         = y_regr_true[:, :4]
        # y_regr_true         = tf.Print(y_regr_true, [y_regr_true, y_regr_pred],summarize=10)

        # (num_pos, )
        pred_left           = y_regr_pred[:, 0]
        pred_top            = y_regr_pred[:, 1]
        pred_right          = y_regr_pred[:, 2]
        pred_bottom         = y_regr_pred[:, 3]

        # (num_pos, )
        target_left         = y_regr_true[:, 0]
        target_top          = y_regr_true[:, 1]
        target_right        = y_regr_true[:, 2]
        target_bottom       = y_regr_true[:, 3]

        # 求真实框和预测框所有的iou
        target_area         = (target_left + target_right) * (target_top + target_bottom)
        pred_area           = (pred_left + pred_right) * (pred_top + pred_bottom)
        w_intersect         = tf.minimum(pred_left, target_left) + tf.minimum(pred_right, target_right)
        h_intersect         = tf.minimum(pred_bottom, target_bottom) + tf.minimum(pred_top, target_top)

        area_intersect      = w_intersect * h_intersect
        area_union          = target_area + pred_area - area_intersect
        iou                 = area_intersect / tf.maximum(area_union, 1.0)

        w_enclose           = tf.maximum(pred_left, target_left) + tf.maximum(pred_right, target_right)
        h_enclose           = tf.maximum(pred_bottom, target_bottom) + tf.maximum(pred_top, target_top)
        enclose_area        = w_enclose*h_enclose

        losses              = 1 - iou + (enclose_area - area_union) / tf.maximum(enclose_area, 1)
        # losses                = 1 - iou
        # losses                = tf.Print(losses, [iou, losses],summarize=10)

        normalizer          = tf.where(K.equal(location_state, 1))
        normalizer          = K.cast(K.shape(normalizer)[0], K.floatx())
        normalizer          = K.maximum(K.cast_to_floatx(1.0), normalizer)

        return tf.reduce_sum(losses)/normalizer

    return _iou

def bce():
    def _bce(y_true, y_pred):
        location_state      = y_true[:, :, -1]
        indices             = tf.where(K.equal(location_state, 1))

        y_centerness_pred   = tf.gather_nd(y_pred, indices)
        y_true              = tf.gather_nd(y_true, indices)
        y_centerness_true   = y_true[:, 0:1]
        
        losses              = K.binary_crossentropy(target=y_centerness_true, output=y_centerness_pred),

        normalizer          = tf.where(K.equal(location_state, 1))
        normalizer          = K.cast(K.shape(normalizer)[0], K.floatx())
        normalizer          = K.maximum(K.cast_to_floatx(1.0), normalizer)
        return  tf.reduce_sum(losses)/normalizer

    return _bce

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

