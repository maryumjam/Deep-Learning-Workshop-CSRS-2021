import argparse
from datetime import datetime
import tensorflow as tf
import socket
import importlib
from utils.provider import *
import models.Unet3D_network as ATD
from scipy.io import loadmat
import numpy as np
import os
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import *
from scipy.ndimage.interpolation import affine_transform

BASE_DIR = os.path.dirname(os.path.abspath(''))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)  # model
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='models', help='Model name [default: model]')
parser.add_argument('--category', default=None, help='Which single class to train on [default: None]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--XY', type=int, default=32, help='Point Number [default: 2048]')
parser.add_argument('--Z', type=int, default=32, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=10, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=100000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.8, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--no_rotation', action='store_true', help='Disable random rotation during training.')
parser.add_argument('--model_path', default='log/check_point/best_model_epoch_003.ckpt',
                    help='model checkpoint file path [default: log/model.ckpt]')

FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
VOXEL_SIZE = FLAGS.XY
VOXEL_Z = FLAGS.Z
MODELRESTORE = FLAGS.model_path
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model)  # import network leia
# module
MODEL_FILE = os.path.join(BASE_DIR, 'model', FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp train.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_VAL = open(os.path.join(LOG_DIR, 'log_val.txt'), 'w')
LOG_TVAL = open(os.path.join(LOG_DIR, 'log_tval.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99
K = 3
HOSTNAME = socket.gethostname()
print("Start Time", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
# readFiles('voxels', "train_files_pw.txt", True)
ALL_FILES = getDataFiles('train_files_pw.txt')

# Load ALL data
data_batch_list_train = []
label_batch_list_train = []
data_batch_list_test = []
label_batch_list_test = []
data_batch_list_train_T = []
label_batch_list_train_T = []
data_batch_list_test_T = []
label_batch_list_test_T = []


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def log_string_val(out_str):
    LOG_VAL.write(out_str + '\n')
    LOG_VAL.flush()
    print(out_str)


def log_string_tval(out_str):
    LOG_TVAL.write(out_str + '\n')
    LOG_TVAL.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.000001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    From tensorflow tutorial: cifar10/cifar10_multi_gpu_train.py
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        # for g, _ in grad_and_vars:
        for g, v in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads



def create_batches(occ_g, lab_g):
    lst_voxel = []
    lst_voxel_lbl = []

    number_cells_x = np.ceil(occ_g.shape[0] / VOXEL_SIZE)
    number_cells_y = np.ceil(occ_g.shape[1] / VOXEL_SIZE)
    number_cells_z = np.ceil(occ_g.shape[2] / VOXEL_Z)
    print(occ_g.shape)

    # print(pred2D.shape)
    sum = 0
    # print("Count before adding to list", occ_g[occ_g > 0].shape)
    for i in range(int(number_cells_x)):
        for j in range(int(number_cells_y)):
            for k in range(int(number_cells_z)):
                start_idx = i * VOXEL_SIZE
                end_inx = (i + 1) * VOXEL_SIZE
                start_idy = j * VOXEL_SIZE
                end_iny = (j + 1) * VOXEL_SIZE
                start_idz = k * VOXEL_Z
                end_inz = (k + 1) * VOXEL_Z


                occ_t_z = np.zeros(shape=(VOXEL_SIZE, VOXEL_SIZE, VOXEL_Z), dtype=occ_g.dtype)
                occ_t_z[0:int(end_inx - start_idx), 0:int(end_iny - start_idy), 0:int(occ_g.shape[2])] = occ_g[
                                                                                                         start_idx:end_inx,
                                                                                                         start_idy:end_iny,
                                                                                                         start_idz:end_inz]
                label_t = np.zeros(shape=(VOXEL_SIZE, VOXEL_SIZE, VOXEL_Z), dtype=lab_g.dtype)
                label_t[0:int(end_inx - start_idx), 0:int(end_iny - start_idy), 0:int(lab_g.shape[2])] = lab_g[
                                                                                                         start_idx:end_inx,
                                                                                                         start_idy:end_iny,
                                                                                                         start_idz:end_inz]

                lst_voxel.append(occ_t_z)
                lst_voxel_lbl.append(label_t)

    if np.mod(len(lst_voxel), BATCH_SIZE) > 0:
        buffer_float = np.zeros(shape=(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE), dtype=occ_g.dtype)
        buffer_int = np.zeros(shape=(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE), dtype=lab_g.dtype)

        remaining = np.mod(len(lst_voxel), BATCH_SIZE)
        nos = BATCH_SIZE + remaining
        for l in range(int(nos)):
            lst_voxel.append(buffer_float)
            lst_voxel_lbl.append(buffer_int)

    return lst_voxel, lst_voxel_lbl



def inference(sess, ops, images, lbal, batch_size):
    ''' pc: BxNx3 array, return BxN pred '''
    num_batches = images.shape[0] / batch_size
    batch_loss_sum = 0
    for i in range(int(num_batches)):
        start_idx = i * BATCH_SIZE
        end_idx = (i + 1) * BATCH_SIZE
        val_data = images
        val_label = lbal
        feed_dict = {ops['pointclouds_pl']: val_data,
                     ops['labels_pl']: val_label,
                     ops['is_training_pl']: False}
        batch_logits, batch_loss, batch_pred, fnet = sess.run([ops['prob'], ops['loss'], ops['pred'], ops['fnet']],
                                                              feed_dict=feed_dict)
        batch_logits = np.argmax(batch_logits, -1)
        batch_loss_sum += batch_loss
    return batch_logits, (batch_loss_sum / float(num_batches))


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl = ATD.placeholder_inputs(BATCH_SIZE, VOXEL_SIZE)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.get_variable('batch', [],
                                    initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Get model and loss
            prob = ATD.get_model(pointclouds_pl, is_training=is_training_pl, bn_decay=bn_decay)
            total_loss = ATD.get_loss(labels_pl, prob,False )
            # tf.summary.scalar('loss', loss)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            train_op = optimizer.minimize(total_loss, global_step=batch)

            correct = tf.equal(tf.argmax(prob, 4), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

        # Add ops to save and restore all the variables.

        saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        # print(MODELRESTORE)
        #saver.restore(sess, MODELRESTORE)
        # Add summary writers
        print("Model restored.")
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': prob,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        best_loss = 1e20
        lst_train = []
        lst_test = []

        kf = KFold(n_splits=K)
        average_loss_train = 0
        average_accuracy_train = 0
        average_loss_test = 0
        average_accuracy_test = 0

        for train_set, test_set in kf.split(ALL_FILES):
            print("%s %s" % (train_set, test_set))
            lst_train.append(train_set)
            lst_test.append(test_set)
            lst_fold_loss_train = 0
            lst_fold_accuracy_train = 0
            lst_fold_accuracy_test = 0
            lst_fold_loss_test = 0
            log_string('**** K FOLD ****')
            for epoch in range(MAX_EPOCH):
                log_string('**** EPOCH %03d ****' % (epoch))
                sys.stdout.flush()
                count = 0
                loss = 0
                accuracy = 0
                loss_e = 0
                accuracy_e = 0
                count_epoch = 0
                for train_file in train_set:
                    print(ALL_FILES[train_file])

                    occ_grid, label_grid, mean_z, mean_z_absolute, lst_pi, num_returns = load_h5_data(
                        os.path.join('preprocessing', 'data', 'voxels', ALL_FILES[train_file]))
                    lst_occ, lst_lbl = create_batches(occ_grid, label_grid)

                    loss_f, acc_f = train_one_epoch(sess, ops, train_writer, lst_occ,lst_lbl)
                    loss += loss_f
                    accuracy += acc_f

                log_string('mean training loss for complete Epoch: %f' % (loss / float(len(train_set))))
                log_string('mean training accuracy for complete Epoch: %f' % (accuracy / float(len(train_set))))

                log_string_val('%f' % (loss / float(len(train_set))))
                log_string_val('%f' % (accuracy / float(len(train_set))))

                lst_fold_loss_train += (loss / float(len(train_set)))
                lst_fold_accuracy_train += (accuracy / float(len(train_set)))
                log_string('---- EVALUATION ----')

                for test_file in test_set:
                    print(ALL_FILES[test_file])

                    t_occ_grid, t_label_grid, t_mean_z, t_mean_z_absolute, t_point_index, t_num_returns = load_h5_data(
                        os.path.join('preprocessing', 'data', 'voxels', ALL_FILES[test_file]))
                    lst_occ_t, lst_lbl_t= create_batches(t_occ_grid, t_label_grid)
                    loss_f_e, acc_f_e = eval_one_epoch(sess, ops, test_writer, lst_occ_t, lst_lbl_t)
                    loss_e += loss_f_e
                    accuracy_e += acc_f_e

                log_string('mean validation loss for complete Epoch: %f' % (loss_e / float(len(test_set))))
                log_string('mean validation accuracy for complete Epoch: %f' % (accuracy_e / float(len(test_set))))
                log_string_tval('%f' % (loss_e / float(len(test_set))))
                log_string_tval('%f' % (accuracy_e / float(len(test_set))))
                lst_fold_loss_test += (loss_e / float(len(test_set)))
                lst_fold_accuracy_test += (accuracy_e / float(len(test_set)))
                if loss_e < best_loss:
                    best_loss = loss_e
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt" % (epoch)))
                    log_string("Model saved in file: %s" % save_path)
                else:
                    count_epoch += 1
                    if count_epoch > 10:
                        save_path = saver.save(sess, os.path.join(LOG_DIR, "last_model_epoch_%03d.ckpt" % (epoch)))
                        log_string("Loss wasn't decreasing. Model saved in file: %s" % save_path)
                        # return
                        # Save the variables to disk.
                if epoch % 3 == 0:
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                    log_string("Model saved in file: %s" % save_path)

            log_string('mean training loss for complete Fold: %f' % (lst_fold_loss_train / float(MAX_EPOCH)))
            log_string('mean training accuracy for complete Fold: %f' % (lst_fold_accuracy_train / float(MAX_EPOCH)))
            average_loss_train += (lst_fold_loss_train / float(MAX_EPOCH))
            average_accuracy_train += (lst_fold_accuracy_train / float(MAX_EPOCH))
            log_string('mean validation loss for complete Fold: %f' % (lst_fold_loss_test / float(MAX_EPOCH)))
            log_string('mean validation accuracy for complete Fold: %f' % (lst_fold_accuracy_test / float(MAX_EPOCH)))
            average_loss_test += (lst_fold_loss_test / float(MAX_EPOCH))
            average_accuracy_test += (lst_fold_accuracy_test / float(MAX_EPOCH))
        log_string('mean training loss for complete Model: %f' % (average_loss_train / float(K)))
        log_string('mean training accuracy for complete Model: %f' % (average_accuracy_train / float(K)))
        log_string('mean validation loss for complete Model: %f' % (average_loss_test / float(K)))
        log_string('mean validation accuracy for complete Model: %f' % (average_accuracy_test / float(K)))


def train_one_epoch(sess, ops, train_writer, train_data, label_target):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    print(train_data[0].shape)
    total_cells = len(train_data)
    log_string(str(datetime.now()))
    num_batches = total_cells // BATCH_SIZE
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    for batch_idx in range(int(num_batches)):
        if batch_idx % 100 == 0:
            print('Current batch/total batch num: of a file %d/%d' % (batch_idx, num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        batch_data = np.expand_dims(train_data[start_idx:end_idx], -1)


        batch_data = np.transpose(batch_data, (0, 3, 1, 2, 4))

        batch_label_t = np.asarray(label_target[start_idx:end_idx])
        batch_label_t = np.transpose(batch_label_t, (0, 3, 1, 2))
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label_t,
                     ops['is_training_pl']: is_training, }

        summary, step, _, loss_val, prob_val = sess.run(
            [ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(prob_val, -1)
        correct = np.sum(pred_val[np.equal(pred_val, batch_label_t)] > 0)

        total_correct += correct
        total_seen += np.prod(batch_label_t[batch_label_t > 0].shape)
        train_writer.add_summary(summary, step)
        loss_sum += loss_val
    print(loss_sum / float(num_batches))
    return (loss_sum / float(num_batches)), (total_correct / float(total_seen))


def eval_one_epoch(sess, ops, test_writer, test_data,  target_data):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    total_cells = len(test_data)
    target_data = np.asarray(target_data)

    log_string(str(datetime.now()))
    num_batches = total_cells // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    log_string(str(datetime.now()))
    for batch_idx in range(int(num_batches)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        batch_data = np.expand_dims(test_data[start_idx:end_idx], -1)
        batch_data = np.transpose(batch_data, (0, 3, 1, 2, 4))
        batch_data_t = np.asarray(target_data[start_idx:end_idx])

        batch_data_t = np.transpose(batch_data_t, (0, 3, 1, 2))
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_data_t,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, prob_val = sess.run([ops['merged'], ops['step'], ops['loss'],
                                                      ops['pred']], feed_dict=feed_dict)

        test_writer.add_summary(summary, step)
        pred_val = np.argmax(prob_val, -1)
        correct = np.sum(pred_val[np.equal(pred_val, batch_data_t)] > 0)
        total_correct += correct
        total_seen += np.prod(batch_data_t[batch_data_t > 0].shape)

        loss_sum += loss_val
    log_string('Cumlative loss: %f' % loss_sum)
    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string("accuracy. %f" % (total_correct / float(total_seen)))

    EPOCH_CNT += 1
    return (loss_sum / float(num_batches)), (total_correct / float(total_seen))


def weird_division(n, d):
    return n / d if d else 0


if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    train()
    LOG_FOUT.close()