import argparse
import tensorflow as tf
from utils.provider import *
from scipy.io import *
import numpy as np
import os
import sys
from sklearn.metrics import *
import csv

BASE_DIR = os.path.dirname(os.path.abspath(''))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)  # model
sys.path.append(ROOT_DIR)  # provider
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import models.Unet3D_network as ATD

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--XY', type=int, default=32, help='Point Number [default: 2048]')
parser.add_argument('--Z', type=int, default=32, help='Point Number [default: 2048]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--model', default='model', help='Model name [default: model]')
parser.add_argument('--model_path', default='log/best_model_epoch_002.ckpt',
                    help='model checkpoint file path [default: log/model.ckpt]')

FLAGS = parser.parse_args()

LOG_DIR = FLAGS.log_dir
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
VOXEL_SIZE = FLAGS.XY
VOXEL_Z = FLAGS.Z
BATCH_SIZE = FLAGS.batch_size
# MODEL = importlib.import_module(FLAGS.model) # import network module
# readFiles('voxels','test_files.txt', False)
ALL_FILES = getDataFiles('test_files.txt')
data_batch_list_test = []
label_batch_list_test = []

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')



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




def get_model_val(batch_size, v_size):
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl = ATD.placeholder_inputs(BATCH_SIZE, VOXEL_SIZE)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            # prob, pred, net_prob,n_f1, n_f2, n_f3,net_concat_1, net_concat_2, net_concat_3 = get_model(pointclouds_pl,is_training_pl)
            prob = ATD.get_model(pointclouds_pl, is_training_pl)
            total_loss = ATD.get_loss(labels_pl, prob, False)
            # losses = tf.get_collection('losses', scope)
            # total_loss = tf.add_n(losses, name='total_loss')
            saver = tf.train.Saver()
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': prob,
               'loss': total_loss,
               }
        return sess, ops


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def undobatching(acc_grdi, pred_grid, prob_grid, occ_grid, acc_labl):
    lab_pred_grid = np.transpose(pred_grid, (0, 2, 3, 1))
    lab_conf_grid = np.transpose(prob_grid, (0, 2, 3, 1, 4))
    # occ_grid =  np.transpose(occ_grid, (0, 2, 3, 1))
    # lab_pred_grid=pred_grid

    bcount = 0
    no_cllx = np.ceil(acc_grdi.shape[0] / VOXEL_SIZE)
    no_clly = np.ceil(acc_grdi.shape[1] / VOXEL_SIZE)
    no_cllz = np.ceil(acc_grdi.shape[2] / VOXEL_Z)
    pred_label = np.zeros((int(no_cllx * VOXEL_SIZE), int(no_clly * VOXEL_SIZE), int(no_cllz * VOXEL_Z)),
                          dtype='double')
    pred_conf = np.zeros((int(no_cllx * VOXEL_SIZE), int(no_clly * VOXEL_SIZE), int(no_cllz * VOXEL_SIZE), 7),
                         dtype='double')
    locc_grid = np.zeros((int(no_cllx * VOXEL_SIZE), int(no_clly * VOXEL_SIZE), int(no_cllz * VOXEL_Z)), dtype='double')
    gt_label = np.zeros((int(no_cllx * VOXEL_SIZE), int(no_clly * VOXEL_SIZE), int(no_cllz * VOXEL_Z)), dtype='int64')
    for i in range(int(no_cllx)):
        for j in range(int(no_clly)):
            for k in range(int(no_cllz)):
                start_idx = i * VOXEL_SIZE
                end_inx = (i + 1) * VOXEL_SIZE
                start_idy = j * VOXEL_SIZE
                end_iny = (j + 1) * VOXEL_SIZE
                start_idz = k * VOXEL_Z
                end_inz = (k + 1) * VOXEL_Z
                pred_label[start_idx:end_inx, start_idy:end_iny, start_idz:end_inz] = lab_pred_grid[bcount,
                                                                                      0:VOXEL_SIZE, 0:VOXEL_SIZE,
                                                                                      0:VOXEL_Z]
                pred_conf[start_idx:end_inx, start_idy:end_iny, start_idz:end_inz, :] = lab_conf_grid[bcount,
                                                                                        0:VOXEL_SIZE, 0:VOXEL_SIZE,
                                                                                        0:VOXEL_SIZE, :]
                locc_grid[start_idx:end_inx, start_idy:end_iny, start_idz:end_inz] = occ_grid[bcount][0:VOXEL_SIZE,
                                                                                     0:VOXEL_SIZE, 0:VOXEL_Z]
                bcount = bcount + 1

    gt_label[0: acc_grdi.shape[0], 0:acc_grdi.shape[1], 0:acc_grdi.shape[2]] = acc_labl[0: acc_grdi.shape[0],
                                                                               0:acc_grdi.shape[1], 0:acc_grdi.shape[2]]
    return pred_label, pred_conf, locc_grid, gt_label


def inference(sess, ops, train_data, label_target, batch_size):
    ''' pc: BxNx3 array, return BxN pred '''

    total_cells = len(train_data)
    num_batches = total_cells // batch_size
    batch_loss_sum = 0
    logits = np.zeros((len(train_data), train_data[0].shape[2], train_data[0].shape[1], train_data[0].shape[0]))
    logits_conf = np.zeros((len(train_data), train_data[0].shape[2], train_data[0].shape[1], train_data[0].shape[0], 7))

    for i in range(int(num_batches)):
        start_idx = i * BATCH_SIZE
        end_idx = (i + 1) * BATCH_SIZE
        batch_data = np.expand_dims(train_data[start_idx:end_idx], -1)
        batch_data = np.transpose(batch_data, (0, 3, 1, 2, 4))

        batch_label_t = np.asarray(label_target[start_idx:end_idx])
        batch_label_t = np.transpose(batch_label_t, (0, 3, 1, 2))
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label_t,
                     ops['is_training_pl']: False}
        batch_logits, batch_loss = sess.run([ops['pred'], ops['loss']], feed_dict=feed_dict)

        # from scipy.special import softmax
        batch_prob = batch_logits
        batch_logits = np.argmax(batch_logits, -1)
        logits[i * batch_size:(i + 1) * batch_size, ...] = batch_logits  # batch_prob[:, :, :, :, 1]
        logits_conf[i * batch_size:(i + 1) * batch_size, ...] = batch_prob
        # logits[i*batch_size:(i+1)*batch_size, ...] = n_f1_val [:,:,:,:, 0]#np.argmax(n_f1_final, -1)
        batch_loss_sum += batch_loss
    return logits, logits_conf, (batch_loss_sum / float(num_batches))  # , logits_pred




if __name__ == '__main__':

    loss_sum = 0
    sess, ops = get_model_val(batch_size=BATCH_SIZE, v_size=VOXEL_SIZE)
    for test_file in ALL_FILES:
        print(test_file)

        occ_grid, label_grid, mean_z, mean_z_absolute, lst_pi, num_returns = load_h5_data(
            os.path.join('preprocessing', 'data', 'voxels', test_file))
        # label_grid[label_grid==3]=2
        print("Before ", occ_grid[occ_grid > 0].shape)
        lst_occ, lst_lbl= create_batches(occ_grid, label_grid)
        log_string('**** Testing *****')
        log_string('**** Filename ***** %s' % test_file)
        log_string("No of Tower voxels in Target data: %f" % len(label_grid[label_grid == 1]))
        log_string("No of Ground voxels in target data: %f" % len(label_grid[label_grid == 2]))
        log_string("No of Low Veg voxels in Target data: %f" % len(label_grid[label_grid == 3]))
        log_string("No of High Veg voxels in target data: %f" % len(label_grid[label_grid == 4]))
        log_string("No of Building voxels in Target data: %f" % len(label_grid[label_grid == 5]))
        log_string("No of Powerline voxels in target data: %f" % len(label_grid[label_grid == 6]))

        pred, prob, loss = inference(sess, ops, lst_occ,  lst_lbl,
                                     BATCH_SIZE)
        predlabel, problabel, locc, gt = undobatching(occ_grid, pred, prob, lst_occ, label_grid)
        print("undobatching", locc[locc > 0].shape)
        from sklearn.metrics import confusion_matrix

        confusion = confusion_matrix(gt.flatten(), predlabel.flatten())
        print(confusion)
        from sklearn.metrics import classification_report, multilabel_confusion_matrix

        print('\nClassification Report\n')
        multilabel_confusion_matrix(gt.flatten().astype(int), predlabel.flatten().astype(int),
                                    labels=[0, 1, 2, 3, 4, 5, 6])
        # save_hdf5_Rvoxel_3D(locc, predlabel, mean_z,mean_z_absolute, lst_pi, test_file)
        save_hdf5_Rvoxel_3D_pred(locc, predlabel, problabel, mean_z, mean_z_absolute, lst_pi, test_file)