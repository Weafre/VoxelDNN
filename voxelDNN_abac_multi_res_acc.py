# inputs: path to saved model, path to point clouds;
# output: bit per occupied voxel
import contextlib
import arithmetic_coding
import numpy as np
import os
import argparse
import time
from voxelDNN_Inference import  occupancy_map_explore
from voxelDNN_meta_endec import save_compressed_file
import gzip
import pickle
from voxelDNN import VoxelDNN
import tensorflow as tf


# encode using 1, 2 3,3 level
# statistic for individual block
# encoding from breadth first sequence for parallel computing
def VoxelDNN_encoding(args):
    pc_level, ply_path, model_path ,bl_par_depth= args
    departition_level = pc_level - 6
    sequence_name = os.path.split(ply_path)[1]
    sequence=os.path.splitext(sequence_name)[0]
    output_path = model_path+str(sequence)
    os.makedirs(output_path,exist_ok=True)
    output_path=output_path+'/'+str(bl_par_depth)+'levels'
    outputfile = output_path+'.blocks.bin'
    metadata_file = output_path + '.metadata.bin'
    heatmap_file = output_path +'.static.pkl'

    start = time.time()

    boxes,binstr,no_oc_voxels=occupancy_map_explore(ply_path,pc_level,departition_level)

    voxelDNN = VoxelDNN(residual_blocks=2)
    voxel_DNN = voxelDNN.restore_voxelDNN(model_path)

    #encoding blocks
    flags=[]
    with contextlib.closing(arithmetic_coding.BitOutputStream(open(outputfile, "wb"))) as bitout, contextlib.closing(arithmetic_coding.BitOutputStream(open(outputfile+'test', "wb"))) as bitest:
        heatmap,flags,no_oc_voxels=voxelDNN_encoding_slave(boxes, voxel_DNN, bitout,bitest,flags,bl_par_depth)
    with open(heatmap_file,'wb') as f:
        pickle.dump(heatmap,f)
    with gzip.open(metadata_file, "wb") as f:
        ret = save_compressed_file(binstr, pc_level, departition_level)
        f.write(ret)
    file_size = int(os.stat(outputfile).st_size) * 8
    metadata_size = int(os.stat(metadata_file).st_size) * 8 + len(flags)*2
    avg_bpov = (file_size + metadata_size ) / no_oc_voxels
    print('\n \nEncoded file: ', ply_path)
    end = time.time()
    print('Encoding time: ', end - start)
    print('Model: ', model_path)
    print('Occupied Voxels: %04d' % no_oc_voxels)
    print('Blocks bitstream: ', outputfile)
    print('Metadata bitstream', metadata_file )
    print('Encoding statics: ', heatmap_file)
    print('Metadata and file size(in bits): ', metadata_size, file_size)
    print('Average bits per occupied voxels: %.04f' % avg_bpov)


def voxelDNN_encoding_slave(oc, voxelDNN, bitout,bitest,flags,par_bl_level):
    static=[]
    enc = arithmetic_coding.ArithmeticEncoder(32, bitout)
    test_enc = arithmetic_coding.ArithmeticEncoder(32, bitest)
    no_ocv=0
    for i in range(oc.shape[0]):
        curr_bits_cnt=[[],[],[],[]]
        print('Encoding block ', i , ' over ', oc.shape[0], 'blocks', end='\r')
        box = []
        box.append(oc[i, :, :, :, :])
        ocv=np.sum(oc[i])
        box = np.asarray(box)

        curr_level=1
        max_level=par_bl_level
        op3,flag3=encode_child_box_test(box,voxelDNN,test_enc,bitest,curr_level, max_level)
        idx=0
        _,curr_bits_cnt=encode_child_box_worker(box, voxelDNN, enc,bitout, flag3, idx,curr_bits_cnt, curr_level,max_level)
        for fl in flag3:
            flags.append(fl)
        static.append([ocv,flag3,op3,curr_bits_cnt])
        no_ocv+=ocv
    enc.finish()  # Flush remaining code bits
    return static,flags,no_ocv

def encode_single_box(box,voxelDNN,enc,bitstream):
    #box 1x64x64x64x1 --> encode ans a box using voxel DNN
    # encoding block as one
    first_bit = bitstream.countingByte * 8
    bbox_max=64
    fake_box = np.zeros((1, bbox_max, bbox_max, bbox_max, 1))
    box_size=box.shape[1]
    fake_box[:,0:box_size,0:box_size,0:box_size,:]=box
    probs = tf.nn.softmax(voxelDNN(fake_box)[0, :, :, :, :], axis=-1)
    probs = probs[0:box_size, 0:box_size, 0:box_size, :]
    probs = np.asarray(probs, dtype='float32')
    for d in range(box_size):
        for h in range(box_size):
            for w in range(box_size):
                symbol = int(box[0,d, h, w, 0])
                fre = [probs[d, h, w, 0], probs[d, h, w, 1], 0.]
                fre = np.asarray(fre)
                fre = (2 ** 10 * fre)
                fre = fre.astype(int)
                fre += 1
                freq = arithmetic_coding.NeuralFrequencyTable(fre)
                enc.write(freq, symbol)
    enc.write(freq, 2)  # EOF
    last_bit = bitstream.countingByte * 8
    return last_bit-first_bit
def encode_child_box_test(box,voxelDNN,test_enc,bitest,curr_level, max_level):
    # box 1xdxdxdx1 --> decide to partition or not
    # flag = 0 if child box is empty;
    # flag = 1 if non empty and encode using voxelDNN as single block
    # flag = 2 if it it will be further partitioned
    child_bbox_max = int(box.shape[1]/2)
    no_bits=0
    flag=[]
    flag.append(2)
    no_bits+=2
    for d in range(2):
        for h in range(2):
            for w in range(2):
                child_box = box[:, d * child_bbox_max:(d + 1) * child_bbox_max,
                            h * child_bbox_max:(h + 1) * child_bbox_max,
                            w * child_bbox_max:(w + 1) * child_bbox_max, :]
                child_flags=[]
                if np.sum(child_box) == 0.:
                    child_flags.append(0)
                    child_no_bits=2
                else:
                    # means the current block is not empty
                    if(curr_level==max_level):
                        child_no_bits=encode_single_box(child_box,voxelDNN,test_enc,bitest)
                        child_flags.append(1)
                        child_no_bits = child_no_bits+2

                    else:
                        #encoding as one
                        op1=encode_single_box(child_box, voxelDNN, test_enc,bitest)
                        op1 = op1+2

                        #encoding using 8 sub child blocks
                        op2,rec_child_flag=encode_child_box_test(child_box, voxelDNN,test_enc,bitest,curr_level+1,max_level)
                        if op2 > op1:
                            child_no_bits = op1
                            child_flags.append(1)
                        else:
                            child_no_bits = op2
                            child_flags=rec_child_flag
                for fl in child_flags:
                    flag.append(fl)
                no_bits+=child_no_bits
    no_bits1 = encode_single_box(box, voxelDNN, test_enc, bitest) + 2
    flag1 = [1]
    if (no_bits>no_bits1):
        return no_bits1,flag1
    else:
        return no_bits,flag



def encode_child_box_worker(box, voxelDNN, enc, bitout,flag,idx,bit_cnt,curr_level, max_level):
    # box 1x64x64x64x1 --> flag 0 if child box 32 is empty;
    # flag 1 if non empty and encode using voxelDNN
    if flag[idx]==2:
        idx+=1
        child_bbox_max = int(box.shape[1] / 2)
        for d in range(2):
            for h in range(2):
                for w in range(2):
                    child_box = box[:, d * child_bbox_max:(d + 1) * child_bbox_max,
                                h * child_bbox_max:(h + 1) * child_bbox_max,
                                w * child_bbox_max:(w + 1) * child_bbox_max, :]
                    ocv=np.sum(child_box)
                    if ocv == 0.:
                        if(flag[idx]!=0):
                            print('************** checking condition 1: ',flag[idx],idx)
                        idx+=1
                    else:
                        if(curr_level==max_level):
                            bit_cnt[curr_level].append([encode_single_box(child_box, voxelDNN, enc,bitout),ocv])

                            if (flag[idx] != 1):
                                print('************** checking condition 2: ', flag[idx], idx)
                            idx+=1
                        else:
                            if (flag[idx] == 1):
                                bit_cnt[curr_level].append([encode_single_box(child_box, voxelDNN, enc,bitout),ocv])
                                idx+=1
                            elif(flag[idx]==2):
                            #idx+=1
                                idx,bit_cnt=encode_child_box_worker(child_box,voxelDNN,enc,bitout,flag,idx,bit_cnt,curr_level+1,max_level)
    elif flag[idx] == 1:
        ocv = np.sum(box)
        bit_cnt[curr_level - 1].append([ocv, encode_single_box(box, voxelDNN, enc, bitout)])
        idx += 1
    return idx,bit_cnt





# Main launcher
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encoding octree')
    parser.add_argument("-level", '--octreedepth', type=int,
                        default=10,
                        help='depth of input octree to pass for encoder')
    parser.add_argument("-depth", '--partitioningdepth', type=int,
                        default=3,
                        help='max depth to partition block')

    parser.add_argument("-ply", '--plypath', type=str, help='path to input ply file')
    parser.add_argument("-pkl", '--pklpath', type=str, help='path to input pkl file')
    parser.add_argument("-model", '--modelpath', type=str, help='path to input model .h5 file')

    args = parser.parse_args()
    VoxelDNN_encoding([args.octreedepth, args.plypath, args.modelpath,args.partitioningdepth])
