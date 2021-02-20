import numpy as np
from octree_partition import partition_octree
import time
from glob import glob
import tensorflow as tf
import multiprocessing
from tqdm import tqdm
from pyntcloud import PyntCloud
import pandas as pd


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2 - time1) * 1000.0))
        return ret

    return wrap



def voxel_block_2_octree(box,oct_seq):
    box_size=box.shape[0]
    child_bbox=int(box_size/2)
    if(box_size>2):
        for d in range(2):
            for h in range(2):
                for w in range(2):
                    child_box=box[d * child_bbox:(d + 1) * child_bbox, h * child_bbox:(h + 1) * child_bbox, w * child_bbox:(w + 1) * child_bbox]
                    if(np.sum(child_box)!=0):
                        oct_seq.append(1)
                        voxel_block_2_octree(child_box,oct_seq)
                    else:
                        oct_seq.append(0)
    else:
        curr_octant=[int(x) for x in box.flatten()]
        oct_seq+=curr_octant
    return oct_seq


# generate bin stream from ply file
def get_bin_stream_blocks(path_to_ply, pc_level, departition_level):
    # co 10 level --> binstr of 10 level, blocks size =1
    level = int(departition_level)
    pc = PyntCloud.from_file(path_to_ply)
    points = pc.points.values
    no_oc_voxels = len(points)
    box = int(2 ** pc_level)
    blocks2, binstr2 = timing(partition_octree)(points, [0, 0, 0], [box, box, box], level)
    return no_oc_voxels, blocks2, binstr2


# Main launcher
def input_fn(points, batch_size, dense_tensor_shape, data_format, repeat=True, shuffle=True, prefetch_size=1):
    print('point shape: ', points.shape)
    # Create input data pipeline.

    dataset = tf.data.Dataset.from_generator(lambda: iter(points), tf.int64, tf.TensorShape([None, 3]))
    if shuffle:
        dataset = dataset.shuffle(len(points))
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.map(lambda x: pc_to_tf(x, dense_tensor_shape, data_format))
    dataset = dataset.map(lambda x: process_x(x, dense_tensor_shape))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_size)

    return dataset


def df_to_pc(df):
    points = df[['x', 'y', 'z']].values
    return points


def pa_to_df(points):
    cols = ['x', 'y', 'z', 'red', 'green', 'blue']
    types = (['float32'] * 3) + (['uint8'] * 3)
    d = {}
    assert 3 <= points.shape[1] <= 6
    for i in range(points.shape[1]):
        col = cols[i]
        dtype = types[i]
        d[col] = points[:, i].astype(dtype)
    df = pd.DataFrame(data=d)
    return df


def pc_to_df(pc):
    points = pc.points
    return pa_to_df(points)


def pc_to_tf(points, dense_tensor_shape, data_format):
    x = points
    assert data_format in ['channels_last', 'channels_first']
    # Add one channel (channels_last convention)
    if data_format == 'channels_last':
        x = tf.pad(x, [[0, 0], [0, 1]])
    else:
        x = tf.pad(x, [[0, 0], [1, 0]])
    st = tf.sparse.SparseTensor(x, tf.ones_like(x[:, 0]), dense_tensor_shape)
    # print('st in pc to tf: ',st)
    return st


def process_x(x, dense_tensor_shape):
    x = tf.sparse.to_dense(x, default_value=0, validate_indices=False)
    x.set_shape(dense_tensor_shape)
    x = tf.cast(x, tf.float32)
    # print('x in process x: ',x)
    return x


def get_shape_data(resolution, data_format):
    assert data_format in ['channels_last', 'channels_first']
    bbox_min = 0
    bbox_max = resolution
    p_max = np.array([bbox_max, bbox_max, bbox_max])
    p_min = np.array([bbox_min, bbox_min, bbox_min])
    if data_format == 'channels_last':
        dense_tensor_shape = np.concatenate([p_max, [1]]).astype('int64')
    else:
        dense_tensor_shape = np.concatenate([[1], p_max]).astype('int64')

    return p_min, p_max, dense_tensor_shape


def get_files(input_glob):
    return np.array(glob(input_glob, recursive=True))


def load_pc(path):
    try:
        pc = PyntCloud.from_file(path)
        points=pc.points
        ret = df_to_pc(points)
        return ret
    except:
        return


def load_points(files, batch_size=32):
    files_len = len(files)

    with multiprocessing.Pool() as p:
        # logger.info('Loading PCs into memory (parallel reading)')
        points = np.array(list(tqdm(p.imap(load_pc, files, batch_size), total=files_len)))

    return points


