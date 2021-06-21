import tensorflow as tf
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

import utils.tf_util as tf_util


def placeholder_inputs(batch_size, v_size):
    pointclouds_pl = tf.placeholder(dtype=tf.float32, shape=(batch_size, v_size, v_size, v_size, 1))
    labels_pl = tf.placeholder(dtype=tf.int64, shape=(batch_size, v_size, v_size, v_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """3D UNET for point clouds.
      Input:
          point_cloud: TF tensor BxNx3
          is_training: boolean
          bn_decay: float between 0 and 1
      Output:
          net: TF tensor BxNx3, reconstructed point clouds
          end_points: dict
      """
    batch_size = point_cloud.get_shape()[0].value
    depth = point_cloud.get_shape()[1].value
    height = point_cloud.get_shape()[2].value
    width = point_cloud.get_shape()[3].value
    pad = 'SAME'
    drop_r = 0.5
    base_fil = 32
    input_image = point_cloud

    # Encoder
    net = tf_util.conv3d(input_image, base_fil, [3, 3, 3],
                         padding=pad, stride=[1, 1, 1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net_f_1 = tf_util.conv3d(net, 2*base_fil, [3, 3, 3],
                             padding=pad, stride=[1, 1, 1],
                             bn=True, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)
    net = tf_util.max_pool3d(net_f_1, [2, 2, 2], stride=[2, 2, 2], scope="maxpool1", padding=pad)
    net = tf_util.conv3d(net, 2*base_fil, [3, 3, 3],
                         padding=pad, stride=[1, 1, 1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net_f_2 = tf_util.conv3d(net, 4*base_fil, [3, 3, 3],
                             padding=pad, stride=[1, 1, 1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
    net_f_2 = tf.layers.dropout(net_f_2, rate=drop_r, training=is_training)
    net = tf_util.max_pool3d(net_f_2, [2, 2, 2],stride=[2, 2, 2], scope="maxpool2",padding=pad)
    net = tf_util.conv3d(net, 4*base_fil, [3, 3, 3],padding=pad, stride=[1, 1, 1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    net_f_3 = tf_util.conv3d(net, 8*base_fil, [3, 3, 3],
                             padding=pad, stride=[1, 1, 1],
                             bn=True, is_training=is_training,
                             scope='conv6', bn_decay=bn_decay)
    net_f_3 = tf.layers.dropout(net_f_3, rate=drop_r, training=is_training)
    net = tf_util.max_pool3d(net_f_3, [2, 2, 2],stride=[2, 2, 2], scope="maxpool3",padding=pad)
    net = tf_util.conv3d(net, 8*base_fil, [3, 3, 3],
                         padding=pad, stride=[1, 1, 1],
                         bn=True, is_training=is_training,
                         scope='conv7', bn_decay=bn_decay)
    # Decoder
    net = tf_util.conv3d_transpose(net, 16*base_fil, [2, 2, 2],padding=pad, stride=[2, 2, 2], bn=True,
                                   is_training=is_training,scope='deconv1', bn_decay=bn_decay)
    net_concat_3 = tf.concat([net_f_3, net], axis=-1)
    net = tf_util.conv3d(net_concat_3, 8*base_fil, [3, 3, 3],
                         padding=pad, stride=[1, 1, 1],
                         bn=True, is_training=is_training,
                         scope='conv8', bn_decay=bn_decay)
    net = tf_util.conv3d(net, 8*base_fil, [3, 3, 3],
                         padding=pad, stride=[1, 1, 1],
                         bn=True, is_training=is_training,
                         scope='conv9', bn_decay=bn_decay)
    net = tf.layers.dropout(net, rate=drop_r, training=is_training)
    net = tf_util.conv3d_transpose(net, 8*base_fil, [2, 2, 2],
                         padding=pad, stride=[2, 2, 2],
                         bn=True, is_training=is_training,scope='deconv2', bn_decay=bn_decay)
    net_concat_2 = tf.concat([net_f_2, net], axis=-1)
    net = tf_util.conv3d(net_concat_2, 4*base_fil, [3, 3, 3],
                         padding=pad, stride=[1, 1, 1],
                         bn=True, is_training=is_training,
                         scope='conv10', bn_decay=bn_decay)
    net = tf_util.conv3d(net, 4*base_fil, [3, 3, 3],
                         padding=pad, stride=[1, 1, 1],
                         bn=True, is_training=is_training,
                         scope='conv11', bn_decay=bn_decay)
    net = tf.layers.dropout(net, rate=drop_r, training=is_training)
    net = tf_util.conv3d_transpose(net, 4*base_fil, [2, 2, 2],
                         padding=pad, stride=[2, 2, 2],
                         bn=True, is_training=is_training,scope='deconv3', bn_decay=bn_decay)
    net_concat_1 = tf.concat([net_f_1, net], axis=-1)
    net = tf_util.conv3d(net_concat_1, 2*base_fil, [3, 3, 3],padding=pad, stride=[1, 1, 1],
                         bn=True, is_training=is_training,
                         scope='conv12', bn_decay=bn_decay)
    net = tf_util.conv3d(net, 2*base_fil, [3, 3, 3],
                         padding=pad, stride=[1, 1, 1],
                         bn=True, is_training=is_training,
                         scope='conv13', bn_decay=bn_decay)
    net = tf.layers.dropout(net, rate=drop_r, training=is_training)
    net = tf_util.conv3d(net, 7, [1, 1, 1],padding=pad, stride=[1, 1, 1],bn=False, is_training=is_training,
                         scope='conv14', bn_decay=bn_decay, activation_fn=None)

    return net


def get_loss(label, logit, do_weights):
        """
        pred: BxDxHxH,
        label: BxDxHxW,
        """
        #batch_size, depth,height, width = pred.shape
        label = tf.expand_dims(label, -1)
        label = tf.squeeze(tf.one_hot(label, 7, axis=-1), axis=-2)
        loss_weights = [0.0, 0.1, 0.9]
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit)
        if do_weights == True:
            weighted_loss = tf.reshape(tf.constant(loss_weights), [1, 1, 1, 1, 3])  # Format to the right size
            weighted_one_hot = tf.reduce_sum(weighted_loss * label, axis=-1)
            loss = loss * weighted_one_hot
        return tf.reduce_mean(loss)


def dice_loss(label, logits):
    label = tf.expand_dims(label, -1)
    onehots_true = tf.squeeze(tf.one_hot(label, 3, axis=-1), axis=-2)
    probabilities = tf.nn.softmax(logits)
    numerator = tf.reduce_sum(onehots_true * probabilities, axis=0)

    denominator = tf.reduce_sum(onehots_true + probabilities, axis=0)
    loss = 1.0 - (2.0 * numerator + 1) / (denominator + 1)
    return tf.reduce_mean(loss)




def simple_loss(label, logits):
    label = tf.expand_dims(label, -1)
    label = tf.squeeze(tf.one_hot(label, 3, axis=-1), axis=-2)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits)
    return tf.reduce_mean(loss)


def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta = 0.5

    ones = tf.ones(tf.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = tf.reduce_sum(p0 * g0, axis=[0, 1, 2, 3])
    den = num + alpha * tf.reduce_sum(p0 * g1, axis=[0, 1, 2, 3]) + beta * tf.reduce_sum(p1 * g0, axis=[0, 1, 2, 3])

    T = tf.reduce_sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = tf.cast(tf.shape(y_true)[-1], 'float32')
    return Ncl - T




def dice_coef_2cat(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 10 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = tf.reshape(tf.one_hot(tf.cast(y_true, 'int32'), 3)[...,1:], -1)
    y_pred_f = tf.reshape(y_pred[...,1:], -1)
    intersect = tf.reduce_sum(y_true_f * y_pred_f, axis=-1)
    denom = tf.reduce_sum(y_true_f + y_pred_f, axis=-1)
    return tf.reduce_mean((2. * intersect / (denom + smooth)))

def dice_coef_2cat_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef_2cat(y_true, y_pred)

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((64, 128, 128, 384, 5))
        prob, pred,net_f1, net_f2, net_f3 = get_model(inputs, tf.constant(True))
        print(prob)
        labels_pl=tf.zeros((64, 128, 128, 384), dtype='int64')
        los = get_loss(labels_pl, prob, True)
        print(los)

