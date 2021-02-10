# VoxelDNN
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar
import pickle
from voxelDNN import VoxelDNN
import argparse

def generating_box(model_path):
    # Building model
    depth = 64
    height = 64
    width = 64
    n_channel = 1
    output_channel = 2
    voxelDNN = VoxelDNN()
    voxelDNN=voxelDNN.restore_voxelDNN(model_path)
    samples = np.zeros((1, depth, height, width, n_channel), dtype='float32')
    progbar = Progbar(64 * 64 * 64)
    for d in range(depth):
        for h in range(height):
            for w in range(width):
                logits = voxelDNN(samples)
                next_sample = tf.random.categorical(logits[:, d, h, w, :], 1)  # sample from distribution
                samples[:, d, h, w, 0] = np.round((next_sample.numpy() / (output_channel - 1))[:, 0])
                progbar.add(1)
    with open('./Model/voxelDNN/box646464.pkl', 'wb') as f:
        pickle.dump(samples, f)
    print('Finish generating pc')

def causality_checking(model_path):
    # Building model
    depth = 64
    height = 64
    width = 64
    n_channel = 1
    output_channel = 2
    box=np.random.randint(0,2,(1,depth,height,width,n_channel))
    box=box.astype('float32')
    voxelDNN = VoxelDNN()
    voxel_DNN=voxelDNN.restore_voxelDNN(model_path)
    predicted_box1=voxel_DNN(box)
    predicted_box1=np.asarray(predicted_box1,dtype='float32')
    i=0
    predicted_box2 = np.zeros((1, depth, height, width, output_channel), dtype='float32')
    progbar = Progbar(64 * 64 * 64)
    for d in range(depth):
        for h in range(height):
            for w in range(width):
                #if i>100:
                #   break
                tmp_box=np.random.randint(0,2,(1,depth,height,width,n_channel))#np.zeros((1, depth, height, width, n_channel), dtype='float32')
                tmp_box=tmp_box.astype(dtype='float32')
                tmp_box[:,:d,:,:,:]=box[:,:d,:,:,:]
                tmp_box[:,d,:h,:,:]=box[:,d,:h,:,:]
                tmp_box[:,d,h,:w,:]=box[:,d,h,:w,:]
                predicted=voxel_DNN(tmp_box)
                predicted_box2[:,d,h,w,:]=predicted[:,d,h,w,:]
                i+=1
           
                progbar.add(1)
    predicted_box2=np.asarray(predicted_box2,dtype='float32')
    compare=predicted_box2==predicted_box1
    print('Check 4: ',np.count_nonzero(compare), compare.all())




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encoding octree')
    parser.add_argument("-level", '--octreedepth', type=int,
                        default=10,
                        help='depth of input octree to pass for encoder')

    parser.add_argument("-ply", '--plypath', type=str, help='path to input ply file')
    parser.add_argument("-model", '--model_path', type=str, help='path to input saved model file')
    args = parser.parse_args()
    causality_checking(args.model_path)
    #generating_box(args.model_path)
    # departition_level = args.octreedepth-6
    # occupancy_map_explore(args.plypath, args.octreedepth,departition_level )
