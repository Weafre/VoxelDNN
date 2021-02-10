# inputs: path to saved model, path to point clouds;
# output: bit per occupied voxel

import arithmetic_coding
import numpy as np
import argparse
from supporting_fcs import  occupancy_map_explore
from voxelDNN_meta_endec import load_compressed_file
import gzip
import pickle
from voxelDNN import  VoxelDNN
import tensorflow as tf



# encoding from breadth first sequence for parallel computing
def voxelDNN_decoding(args):
    ply_path, model_path, outputfile, metadata = args
    #reading metadata
    with gzip.open(metadata, "rb") as f:
        pc_level, departition_level, decoded_binstr=load_compressed_file(f)
    #getting encoding input data
    boxes,binstr,no_oc_voxels=occupancy_map_explore(ply_path,pc_level,departition_level)

    #restore voxelDNN
    voxelDNN = VoxelDNN()
    voxel_DNN = voxelDNN.restore_voxelDNN(model_path)
    with open(metadata+'.pkl','rb') as f:
        flags=pickle.load(f)
    with open(outputfile, "rb") as inp:
        bitin = arithmetic_coding.BitInputStream(inp)
        decoded_boxes=decompress_from_adaptive_freqs(boxes,flags, voxel_DNN, bitin)
    decoded_boxes=decoded_boxes.astype(int)
    boxes=boxes.astype(int)
    compare=decoded_boxes[:2] == boxes[:2]
    print('Check 1: decoded pc level: ',pc_level)
    print('Check 2: decoded block level',  departition_level)
    print('Check 3: decoded binstr ', binstr == decoded_binstr)
    print('Check 4: decoded boxes' ,np.count_nonzero(compare),compare.all())


def decompress_from_adaptive_freqs(boxes,flags, voxelDNN, bitin):
    dec = arithmetic_coding.ArithmeticDecoder(32, bitin)
    no_box=len(boxes)
    bbox_max=boxes[0].shape[0]
    #progbar = Progbar(1 * bbox_max ** 3)#chang 1 to no_box if want to decode the all boxes
    decoded_boxes=np.zeros((no_box,bbox_max,bbox_max,bbox_max,1))
    fl_idx=0
    #for i in range(no_box):
    for i in range(2):##chang 1 to no_box if want to decode the all boxes
        box = []
        box.append(boxes[i, :, :, :, :])
        box = np.asarray(box)
        print('number of non empty voxels: ', np.sum(box))

        if flags[fl_idx]==0:
            print('decoding as a whole box')
            probs = tf.nn.softmax(voxelDNN(box)[0, :, :, :, :], axis=-1)
            probs = np.asarray(probs, dtype='float32')
            for d in range(bbox_max):
                for h in range(bbox_max):
                    for w in range(bbox_max):

                        fre = [probs[d, h, w, 0], probs[d, h, w, 1], 0.]
                        fre = np.asarray(fre)
                        fre = (2 ** 10 * fre)
                        fre = fre.astype(int)
                        fre += 1
                        freq = arithmetic_coding.NeuralFrequencyTable(fre)
                        symbol = dec.read(freq)
                        decoded_boxes[i,d,h,w,0]=symbol
                        #progbar.add(1)
            symbol=dec.read(freq)
            fl_idx+=1
        else:
            fl_idx+=1
            print('decoding using 32 block size')
            child_bbox_max = 32
            print(flags[0:9])
            for d in range(2):
                for h in range(2):
                    for w in range(2):
                        child_box = box[0, d * child_bbox_max:(d + 1) * child_bbox_max,
                                    h * child_bbox_max:(h + 1) * child_bbox_max,
                                    w * child_bbox_max:(w + 1) * child_bbox_max, :]
                        if(flags[fl_idx]==0):
                            print('zero block 32')
                            decoded_boxes[i, d * child_bbox_max:(d + 1) * child_bbox_max,
                                  h * child_bbox_max:(h + 1) * child_bbox_max,
                                  w * child_bbox_max:(w + 1) * child_bbox_max, :]=0
                            #progbar.add(32*32*32)
                        else:
                            print('occupied block 32')
                            fake_box = np.zeros_like(box)
                            fake_box[0, d * child_bbox_max:(d + 1) * child_bbox_max,
                            h * child_bbox_max:(h + 1) * child_bbox_max,
                            w * child_bbox_max:(w + 1) * child_bbox_max, :] = child_box
                            probs = tf.nn.softmax(voxelDNN(fake_box)[0, :, :, :, :], axis=-1)
                            probs = np.asarray(probs, dtype='float32')
                            probs = probs[d * child_bbox_max:(d + 1) * child_bbox_max,
                                    h * child_bbox_max:(h + 1) * child_bbox_max,
                                    w * child_bbox_max:(w + 1) * child_bbox_max, :]
                            decoded_child_box=np.zeros_like(child_box)
                            for c_d in range(child_bbox_max):
                                for c_h in range(child_bbox_max):
                                    for c_w in range(child_bbox_max):
                                        #probs = np.asarray(probs, dtype='float32')
                                        fre = [probs[c_d, c_h, c_w, 0], probs[c_d, c_h, c_w, 1], 0.]
                                        fre = np.asarray(fre)
                                        fre = (2 ** 10 * fre)
                                        fre = fre.astype(int)
                                        fre += 1
                                        freq = arithmetic_coding.NeuralFrequencyTable(fre)
                                        symbol = dec.read(freq)
                                        decoded_child_box[ c_d, c_h, c_w, 0] = symbol
                                        #progbar.add(1)
                            symbol = dec.read(freq)
                            print('\n ending symbol:', symbol)
                            decoded_boxes[i, d * child_bbox_max:(d + 1) * child_bbox_max,
                            h * child_bbox_max:(h + 1) * child_bbox_max,
                            w * child_bbox_max:(w + 1) * child_bbox_max, :] = decoded_child_box
                            compare=decoded_child_box==child_box
                            print('decoded nonempty boxes:', np.count_nonzero(compare))
                        fl_idx+=1
    return decoded_boxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encoding octree')
    parser.add_argument("-level", '--octreedepth', type=int,
                        default=10,
                        help='depth of input octree to pass for encoder')

    parser.add_argument("-ply", '--plypath', type=str, help='path to input ply file')
    parser.add_argument("-pkl", '--pklpath', type=str, help='path to input pkl file')
    parser.add_argument("-model", '--modelpath', type=str, help='path to input model .h5 file')

    parser.add_argument("-output", '--outputfile', type=str,
                        default=7,
                        help='name of output file')
    parser.add_argument("-metadata", '--output_metadata', type=str,
                        default=7,
                        help='name of output file')
    parser.add_argument("-heatmap", '--output_heatmap', type=str,
                        default=7,
                        help='name of output heatmap pkl')

    args = parser.parse_args()
    voxelDNN_decoding([args.plypath, args.modelpath, args.outputfile,args.output_metadata])
