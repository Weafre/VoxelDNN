# LEARNING-BASED LOSSLESS COMPRESSION OF 3D POINT CLOUD GEOMETRY
* **Authors**:
[Dat T. Nguyen](https://scholar.google.com/citations?hl=en&user=uqqqlGgAAAAJ),
[Maurice Quach](https://scholar.google.com/citations?user=atvnc2MAAAAJ),
[Giuseppe Valenzise](https://scholar.google.com/citations?user=7ftDv4gAAAAJ) and
[Pierre Duhamel](https://scholar.google.com/citations?user=gWj_W9YAAAAJ&hl=en&oi=ao)  
* **Affiliation**: Université Paris-Saclay, CNRS, CentraleSupélec, Laboratoire des signaux et systèmes, 91190 Gif-sur-Yvette, France
* **Links**: [[Paper]](https://arxiv.org/abs/2011.14700)
## Prerequisites
* Python 3.8
* Tensorflow 2.3.1 with CUDA 10.1.243 and cuDNN 7.6.5

Run command below to install all prerequired packages:
    
    pip3 install -r requirements.txt



## Prepare datasets
The training data are the .ply files containing x,y,z coordinates of points within a 64x64x64 patch divided from Point Clouds. Our Point Clouds download from [ModelNet40](http://modelnet.cs.princeton.edu),[MPEG 8i](http://plenodb.jpeg.org/pc/8ilabs) and [Microsoft](http://plenodb.jpeg.org/pc/microsoft). The ModelNet40 dataset provides train and test folder separately. For MPEG and Microsoft dataset, you must manually select PCs into train and test. Training data generation is similar to this [repo](https://github.com/mauriceqch/pcc_geo_cnn_v2) for each dataset. The commands below first select 200 densiest Point Clouds (PC) from ModelNet40, convert it from mesh to PC and then divide each PC into occupied blocks of size 64x64x64 voxels (saved in .ply format)

        python ds_select_largest.py datasets/ModelNet40 datasets/ModelNet40_200 200
        python ds_mesh_to_pc.py datasets/ModelNet40_200 datasets/ModelNet40_200_pc512 --vg_size 512
        python ds_pc_octree_blocks.py datasets/ModelNet40_200_pc512 datasets/ModelNet40_200_pc512_oct3 --vg_size 512 --level 3 
     
      
You only need to run the last command for MPEG and Microsoft after selecting PCs into `train/` and `test/` folder. Note that MPEG 8i and Microsoft are 10-bits point clouds thus, you must change --vg_size to 1024 and --level to 4:

    python ds_pc_octree_blocks.py datasets/MPEG/10bitdepth/ datasets/MPEG/10bitdepth_2_oct4/ --vg_size 1024 --level 4

The `datsets/` folder of MPEG and Microsoft should look like this:

    dataset/
    └── MPEG/
        └── 10bitdepth/              downloaded PCs from MPEG
            ├── train/               contains .ply PCs for training 
            └── test/                contains .ply PCs for validation         
        └── 10bitdepth_2_oct4/
            ├── train/               contains .ply files of 64x64x64 blocks for training 
            └── test/                contains .ply files of 64x64x64 blocks for validation
    └── Microsoft/
        └── 10bitdepth/              downloaded PCs from Microsoft
            ├── train/               contains .ply PCs for training 
            └── test/                contains .ply PCs for validation         
        └── 10bitdepth_2_oct4/
            ├── train/               contains .ply files of 64x64x64 blocks for training 
            └── test/                contains .ply files of 64x64x64 blocks for validation


## Training
Run the following command:
    
    python3 voxelDNN.py -blocksize 64 -nfilters 64 -inputmodel Model/voxelDNN/ -outputmodel Model/voxelDNN/ -dataset datasets/ModelNet40_200_pc512_oct3/ -dataset datasets/Microsoft/10bitdepth_2_oct4/ -dataset datasets/MPEG/10bitdepth_2_oct4/  -batch 8 -epochs 50
    
## Encoder
Encoding command: 

    python3 voxelCNN_abac_multi_res_acc.py -level 10  -ply TestPC/Microsoft/10bits/phil10/ply/frame0010.ply -model Model/voxelDNN/ -depth 3
Run `python3 voxelCNN_abac_multi_res_acc.py -h` for more details about the arguments. The encoder outputs look like this:

    Encoded file:  TestPC/Microsoft/10bits/phil10/ply/frame0010.ply
    Encoding time:  7440.565563201904
    Models:  Model/voxelDNN/
    Occupied Voxels: 1559008
    Blocks bitstream:  Model/voxelDNN/frame0010/3levels.blocks.bin
    Metadata bitstream Model/voxelDNN/frame0010/3levels.metadata.bin
    Encoding statics:  Model/voxelDNN/frame0010/3levels.static.pkl
    Metadata and file size(in bits):  11780 1283288
    Average bits per occupied voxels: 0.8307
This `Model/voxelDNN/frame0010/3levels.static.pkl` file contains encoding statics for all blocks 64 including partitioning flags, number of occupied voxels and number of bits spent for each child blocks.
## Decoder
The purpose of our decoder is to make sure the bitstream is decodable and the decoded point cloud is identical to the input point cloud. In order to reduce complexity, the decoder use the input PC to predict the probability distributions. We will obtain the exactly the same distributions as sequential prediction from decoded voxels because of the causality contraints is enforced with masked filters. The distributions are then passed to an arithmetic decoder:

    python3 voxelDNN_abac_multi_res_dec.py -ply TestPC/Microsoft/10bits/phil10/ply/frame0010.ply -model Model/voxelDNN/ -output Model/voxelDNN/frame0010/3levels.blocks.bin -metadata Model/voxelDNN/frame0010/3levels.metadata.bin -heatmap Model/voxelDNN/frame0010/3levels.static.pkl
## Citation

    @misc{nguyen2020learningbased,
          title={Learning-based lossless compression of 3D point cloud geometry}, 
          author={Dat Thanh Nguyen and Maurice Quach and Giuseppe Valenzise and Pierre Duhamel},
          year={2020},
          eprint={2011.14700},
          archivePrefix={arXiv},
          primaryClass={eess.IV}
    }
