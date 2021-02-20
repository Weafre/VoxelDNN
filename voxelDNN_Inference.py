# VoxelDNN
import numpy as np
from tensorflow.keras.utils import Progbar
from voxelDNN import VoxelDNN
from supporting_fcs import get_bin_stream_blocks
import argparse


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


# blocks to occupancy maps
def pc_2_block_oc3(blocks, bbox_max=512):
    no_blocks = len(blocks)
    blocks_oc = np.zeros((no_blocks, bbox_max, bbox_max, bbox_max, 1), dtype=np.float32)
    for i, block in enumerate(blocks):
        block = block[:, 0:3]
        block = block.astype(np.uint32)
        blocks_oc[i, block[:, 0], block[:, 1], block[:, 2], 0] = 1.0
    return blocks_oc


def occupancy_map_explore(ply_path, pc_level, departition_level):
    no_oc_voxels, blocks, binstr = get_bin_stream_blocks(ply_path, pc_level, departition_level)
    print('Finished loading model and ply to oc')
    boxes = pc_2_block_oc3(blocks, bbox_max=64)
    print('Boxes shape:', boxes.shape)
    #print('Number of empty boxes 32: ', count)
    return boxes, binstr, no_oc_voxels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encoding octree')
    parser.add_argument("-level", '--octreedepth', type=int,
                        default=10,
                        help='depth of input octree to pass for encoder')

    parser.add_argument("-ply", '--plypath', type=str, help='path to input ply file')
    parser.add_argument("-model", '--model_path', type=str, help='path to input saved model file')
    args = parser.parse_args()
    causality_checking(args.model_path)

