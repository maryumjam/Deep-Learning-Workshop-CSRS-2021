import os
import sys
import numpy as np
import h5py
import laspy as lp
BASE_DIR = os.path.dirname(os.path.abspath(os.path.join(' ')))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')



def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def save_hdf5_voxelStuff(occupancy_grid_subarea,filename):
    filname = os.path.splitext(os.path.basename(filename))[0]
    unique_filename = filname
    fpath = os.path.join(BASE_DIR,'Grid', unique_filename + ".h5")
    hf = h5py.File(fpath, 'w')


    hf.create_dataset('occupancy_grid', data=occupancy_grid_subarea, compression='gzip', compression_opts=4, dtype='float32')
    hf.close()
    return


def load_h5_data(h5_filename):
    f = h5py.File(h5_filename, 'r+')
    data = f['occupancy_grid'][:]
    label = f['label'][:]
    mean_z = f['mean_z'][:]
    mean_z_absolute = f['mean_z_absolute'][:]
    point_index = f['point_index'][:]
    num_returns=f['num_returns'][:]
    return (data, label,mean_z,mean_z_absolute, point_index, num_returns)

def load_h5_grid(h5_filename):
    f = h5py.File(h5_filename, 'r+')
    data = f['occupancy_grid'][:]
    label = f['label'][:]
    meanz = f['meanz'][:]
    return (data, label, meanz)


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


def getDataFiles(list_filename):
    print(list_filename)
    return [line.rstrip() for line in open(os.path.join(BASE_DIR, list_filename))]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)


def loadPointcloud(data_label_filename):
    if data_label_filename[-3:] == 'txt':
        data_label = np.loadtxt(data_label_filename)
    elif data_label_filename[-3:] == 'asc':
        data_label = np.loadtxt(data_label_filename)
    elif data_label_filename[-3:] == 'npz':
        data_label = np.load(data_label_filename)
    elif data_label_filename[-3:] == "las" or data_label_filename[-3:] == "laz":
        inFile = lp.file.File(data_label_filename, mode="r")
        data_label = np.zeros([len(inFile), 13], dtype=None)
        min_gps_time = 1e100
        max_gps_time = 0.0
        data_label[:, 0] = inFile.X * inFile.header.scale[0] + inFile.header.offset[0]
        data_label[:, 1] = inFile.Y * inFile.header.scale[1] + inFile.header.offset[1]
        data_label[:, 2] = inFile.Z * inFile.header.scale[2] + inFile.header.offset[2]
        data_label[:, 3] = inFile.intensity
        data_label[:, 4] = inFile.gps_time
        data_label[:, 5] = inFile.scan_angle_rank
        data_label[:, 6] = inFile.edge_flight_line
        data_label[:, 7] = inFile.pt_src_id
        data_label[:, 8] = inFile.user_data
        data_label[:, 9] = inFile.num_returns
        data_label[:, 10] = inFile.return_num
        data_label[:, 11] = inFile.scan_dir_flag
        data_label[:, 12] = inFile.classification
    else:
        print('Unknown file type! exiting.')
        exit()
    return data_label

def save_hdf5_Rvoxel(occupancy_grid_subarea, label_subarea, mean_x_subarea, mean_y_subarea, mean_z_subarea,
                    intensity_subarea, point_index, filename):
    filname = os.path.splitext(os.path.basename(filename))[0]
    unique_filename = filname
    fpath = os.path.join(BASE_DIR, 'preprocessing', 'data', 'voxels_results', unique_filename + "_.h5")
    hf = h5py.File(fpath, 'w')

    hf.create_dataset('occupancy_grid', data=occupancy_grid_subarea, compression='gzip', compression_opts=4, dtype='float32')
    hf.create_dataset('label', data=label_subarea,  compression='gzip', compression_opts=4, dtype='uint8')
    hf.create_dataset('mean_x', data=mean_x_subarea, compression='gzip', compression_opts=4, dtype='float32')
    hf.create_dataset('mean_y', data=mean_y_subarea, compression='gzip', compression_opts=4, dtype='float32')
    hf.create_dataset('mean_z', data=mean_z_subarea, compression='gzip', compression_opts=4, dtype='float32')
    hf.create_dataset('intensity', data=intensity_subarea, compression='gzip', compression_opts=4, dtype='float32')
    hf.create_dataset('point_index', data=point_index, compression='gzip', compression_opts=4, dtype='uint32')


    hf.close()

    return


def save_hdf5_Rvoxel_3D(occupancy_grid_subarea, label_subarea, mean_z_subarea, mean_z_abs_subarea, point_index, filename):
    filname = os.path.splitext(os.path.basename(filename))[0]
    unique_filename = filname
    fpath = os.path.join(BASE_DIR, 'preprocessing', 'data', 'voxels_results', unique_filename + ".h5")
    hf = h5py.File(fpath, 'w')

    hf.create_dataset('occupancy_grid', data=occupancy_grid_subarea, compression='gzip', compression_opts=4, dtype='float32')
    hf.create_dataset('label', data=label_subarea,  compression='gzip', compression_opts=4, dtype='uint8')
    hf.create_dataset('mean_z', data=mean_z_subarea, compression='gzip', compression_opts=4, dtype='float32')
    hf.create_dataset('mean_z_abs', data=mean_z_abs_subarea, compression='gzip', compression_opts=4, dtype='float32')
    hf.create_dataset('point_index', data=point_index, compression='gzip', compression_opts=4, dtype='uint32')
    hf.close()

    return

def save_hdf5_Rvoxel_3D_pred(occupancy_grid_subarea, label_subarea, prob_subarea, mean_z_subarea, mean_z_abs_subarea, point_index, filename):
    filname = os.path.splitext(os.path.basename(filename))[0]
    unique_filename = filname
    fpath = os.path.join(BASE_DIR, 'preprocessing', 'data', 'voxels_results', unique_filename + ".h5")
    hf = h5py.File(fpath, 'w')

    hf.create_dataset('occupancy_grid', data=occupancy_grid_subarea, compression='gzip', compression_opts=4, dtype='float32')
    hf.create_dataset('label', data=label_subarea,  compression='gzip', compression_opts=4, dtype='uint8')
    hf.create_dataset('prob', data=prob_subarea,  compression='gzip', compression_opts=4, dtype='float32')
    hf.create_dataset('mean_z', data=mean_z_subarea, compression='gzip', compression_opts=4, dtype='float32')
    hf.create_dataset('mean_z_abs', data=mean_z_abs_subarea, compression='gzip', compression_opts=4, dtype='float32')
    hf.create_dataset('point_index', data=point_index, compression='gzip', compression_opts=4, dtype='uint32')
    hf.close()

    return
def save_hdf5_voxel(occupancy_grid_subarea,occupancy_grid_un_subarea, label_subarea,mean_z_subarea,            mean_z_avox_subarea,point_index,num_returns_subarea,totalpoints_batch,novox_X_batch,novox_Y_batch,novox_Z_batch,num_block_x,num_block_y,l,filename):
    filname = os.path.splitext(os.path.basename(filename))[0]
    unique_filename = filname
    fpath = os.path.join(BASE_DIR,'voxels_KP', unique_filename + ".h5")
    hf = h5py.File(fpath, 'w')

    hf.create_dataset('occupancy_grid', data=occupancy_grid_subarea, compression='gzip', compression_opts=4, dtype='float32')
    hf.create_dataset('un_occupancy_grid', data=occupancy_grid_un_subarea, compression='gzip', compression_opts=4,dtype='float32')
    hf.create_dataset('label', data=label_subarea,  compression='gzip', compression_opts=4, dtype='uint8')
    hf.create_dataset('mean_z', data=mean_z_subarea, compression='gzip', compression_opts=4, dtype='float32')
    hf.create_dataset('mean_z_absolute',data=mean_z_avox_subarea, compression='gzip', compression_opts=4, dtype='float32')
    hf.create_dataset('point_index', data=point_index, compression='gzip', compression_opts=4, dtype='float32')
    hf.create_dataset('num_returns', data=num_returns_subarea, compression='gzip', compression_opts=4, dtype='float32')
    hf.create_dataset('total_points', data=totalpoints_batch, compression='gzip', compression_opts=4, dtype='float32')
    hf.create_dataset('novox_X_batch', data=novox_X_batch, compression='gzip', compression_opts=4, dtype='float32')
    hf.create_dataset('novox_Y_batch', data=novox_Y_batch, compression='gzip', compression_opts=4, dtype='float32')
    hf.create_dataset('novox_Z_batch', data=novox_Z_batch, compression='gzip', compression_opts=4, dtype='float32')
    hf.create_dataset('num_block_x', data=num_block_x, compression='gzip', compression_opts=4, dtype='float32')
    hf.create_dataset('num_block_y', data=num_block_y, compression='gzip', compression_opts=4, dtype='float32')
    hf.create_dataset('non_empty_blocks', data=l, compression='gzip', compression_opts=4, dtype='float32')
    hf.close()

    return





def load_h5_Com(h5_filename, dataset, directory):
    f = h5py.File(os.path.join(BASE_DIR, directory, h5_filename))
    data = f[dataset][:]

    return data


def save_txtfile(point_cloud, area, filename):
    filname = os.path.splitext(os.path.basename(filename))[0]
    print("Saving a  .txt file.")
    unique_filename = str(area) + filname
    fpath = os.path.join(BASE_DIR, 'preprocessing', 'data', 'subscenestxt', unique_filename + ".txt")
    np.savetxt(fpath, point_cloud)


def save_lasUp(point_cloud, filename, output_dir,minx, miny, minz):
    filname=os.path.splitext(os.path.basename(filename))[0]
    print("Saving a  .las file.")
    header = lp.header.Header(point_format=1)
    unique_filename = filname
    fpath=os.path.join(BASE_DIR, 'preprocessing', 'data', output_dir, unique_filename+".las")
    outfile = lp.file.File(fpath, mode="w", header=header)
    '''
    xmin = np.floor(np.min(point_cloud[:, 0]))
    ymin = np.floor(np.min(point_cloud[:, 1]))
    zmin = np.floor(np.min(point_cloud[:, 2]))
    '''

    outfile.header.offset = [minx, miny, minz]
    outfile.header.scale = [1, 1, 1]

    outfile.X = point_cloud[:, 0]
    outfile.Y = point_cloud[:, 1]
    outfile.Z = point_cloud[:, 2]

    outfile.intensity = point_cloud[:, 3]
    outfile.gps_time = point_cloud[:, 4]
    outfile.scan_angle_rank = point_cloud[:, 5]
    outfile.edge_flight_line = (point_cloud[:, 6]).astype(int)
    outfile.pt_src_id = point_cloud [:, 7]
    outfile.user_data = point_cloud[:, 8]
    outfile.num_returns = (point_cloud[:, 9]).astype(int)
    outfile.return_num = (point_cloud[:, 10]).astype(int)
    outfile.scan_dir_flag = (point_cloud[:, 11]).astype(int)

    outfile.classification = (point_cloud[:, 12]).astype(int)
    outfile.close()
    print("Total points"+str(point_cloud.shape[0]))
    print("Subarea file saved "+unique_filename+"_.las at ", os.path.dirname(os.path.abspath(outfile.filename)))


def save_lasfie(point_cloud, area, filename, minx, miny, minz):
    filname=os.path.splitext(os.path.basename(filename))[0]
    print("Saving a  .las file.")
    header = lp.header.Header(point_format=1)
    unique_filename = str(area)+filname
    fpath=os.path.join(BASE_DIR, 'subscenes', unique_filename+".las")
    outfile = lp.file.File(fpath, mode="w", header=header)
    '''
    xmin = np.floor(np.min(point_cloud[:, 0]))
    ymin = np.floor(np.min(point_cloud[:, 1]))
    zmin = np.floor(np.min(point_cloud[:, 2]))
    '''

    outfile.header.offset = [minx, miny, minz]
    outfile.header.scale = [1, 1, 1]

    outfile.X = point_cloud[:, 0]
    outfile.Y = point_cloud[:, 1]
    outfile.Z = point_cloud[:, 2]

    outfile.intensity = point_cloud[:, 3]
    outfile.gps_time = point_cloud[:, 4]
    outfile.scan_angle_rank = point_cloud[:, 5]
    outfile.edge_flight_line = (point_cloud[:, 6]).astype(int)
    outfile.pt_src_id = point_cloud [:, 7]
    outfile.user_data = point_cloud[:, 8]
    outfile.num_returns = (point_cloud[:, 9]).astype(int)
    outfile.return_num = (point_cloud[:, 10]).astype(int)
    outfile.scan_dir_flag = (point_cloud[:, 11]).astype(int)

    outfile.classification = (point_cloud[:, 12]).astype(int)
    outfile.close()
    print("Total points"+str(point_cloud.shape[0]))
    print("Subarea file saved "+unique_filename+"_.las at ", os.path.dirname(os.path.abspath(outfile.filename)))




def save_result_txt(point_cloud, fpath):
    print("Saving a  .txt file.")
    np.savetxt(fpath, point_cloud)
    print("Total points" + str(point_cloud.shape[0]))
    print("file saved .txt at ", fpath)




def save_cresult_las(point_cloud, filename,xmin,ymin,zmin):
    print("Saving a  .las file.")
    header = lp.header.Header(point_format=1)
    fpath =filename
    outfile = lp.file.File(fpath, mode="w", header=header)


    outfile.header.offset = [xmin, ymin, zmin]
    outfile.header.scale = [1, 1, 1]

    outfile.X = point_cloud[:, 0]
    outfile.Y = point_cloud[:, 1]
    outfile.Z = point_cloud[:, 2]

    outfile.intensity = point_cloud[:, 3]
    outfile.gps_time = point_cloud[:, 4]
    outfile.scan_angle_rank = point_cloud[:, 5]
    outfile.edge_flight_line = (point_cloud[:, 6]).astype(int)
    outfile.pt_src_id = point_cloud[:, 7]
    outfile.user_data = point_cloud[:, 8]
    outfile.num_returns = (point_cloud[:, 9]).astype(int)
    outfile.return_num = (point_cloud[:, 10]).astype(int)
    outfile.scan_dir_flag = (point_cloud[:, 11]).astype(int)

    outfile.classification = (point_cloud[:, -1]).astype(int)
    outfile.close()
    print("Total points" + str(point_cloud.shape[0]))
    print("file saved " + outfile.filename + ".las at ", os.path.dirname(os.path.abspath(outfile.filename)))


def save_cresult_txtlas(point_cloud, filename,xmin,ymin,zmin):
    print("Saving a  .las file.")
    header = lp.header.Header(point_format=1)
    fpath =filename
    outfile = lp.file.File(fpath, mode="w", header=header)


    outfile.header.offset = [xmin, ymin, zmin]
    outfile.header.scale = [1, 1, 1]

    outfile.X = point_cloud[:, 0]
    outfile.Y = point_cloud[:, 1]
    outfile.Z = point_cloud[:, 2]

    outfile.intensity = point_cloud[:, 3]
    outfile.gps_time = point_cloud[:, 4]
    outfile.scan_angle_rank = point_cloud[:, 5]
    outfile.edge_flight_line = (point_cloud[:, 6]).astype(int)
    outfile.pt_src_id = point_cloud[:, 7]
    outfile.user_data = point_cloud[:, 8]
    outfile.num_returns = (point_cloud[:, 9]).astype(int)
    outfile.return_num = (point_cloud[:, 10]).astype(int)
    outfile.scan_dir_flag = (point_cloud[:, 11]).astype(int)

    outfile.classification = (point_cloud[:, -1]).astype(int)
    outfile.close()
    print("Total points" + str(point_cloud.shape[0]))
    print("file saved " + outfile.filename + ".las at ", os.path.dirname(os.path.abspath(outfile.filename)))


def savecolored_cloud(point_cloud, filename, xmin, ymin, zmin):
    print("Saving a  .las file.")
    header = lp.header.Header(point_format=1)
    fpath = filename
    outfile = lp.file.File(fpath, mode="w", header=header)

    outfile.header.offset = [xmin, ymin, zmin]
    outfile.header.scale = [1, 1, 1]

    outfile.X = point_cloud[:, 0]
    outfile.Y = point_cloud[:, 1]
    outfile.Z = point_cloud[:, 2]

    outfile.intensity = point_cloud[:, 3]
    outfile.gps_time = point_cloud[:, 4]
    outfile.scan_angle_rank = point_cloud[:, 5]
    outfile.edge_flight_line = (point_cloud[:, 6]).astype(int)
    outfile.pt_src_id = point_cloud[:, 7]
    outfile.user_data = point_cloud[:, 8]
    outfile.num_returns = (point_cloud[:, 9]).astype(int)
    outfile.return_num = (point_cloud[:, 10]).astype(int)
    outfile.scan_dir_flag = (point_cloud[:, 11]).astype(int)


    outfile.classification = (point_cloud[:, -1]).astype(int)
    outfile.close()
    print("Total points" + str(point_cloud.shape[0]))
    print("file saved " + outfile.filename + ".las at ", os.path.dirname(os.path.abspath(outfile.filename)))