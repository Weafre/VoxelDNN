# VoxelDNN
import random as rn
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import nn
from tensorflow.keras import initializers
from tensorflow.keras.utils import Progbar
from supporting_fcs import input_fn, get_shape_data, get_files, load_points
import os
import sys
import argparse
import tensorflow.keras.backend as K
import datetime
random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
rn.seed(random_seed)


# Defining main block
class MaskedConv3D(keras.layers.Layer):

    def __init__(self,
                 mask_type,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'):
        super(MaskedConv3D, self).__init__()

        assert mask_type in {'A', 'B'}
        self.mask_type = mask_type

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel',
                                      shape=(self.kernel_size,
                                             self.kernel_size,
                                             self.kernel_size,
                                             int(input_shape[-1]),
                                             self.filters),
                                      initializer=self.kernel_initializer,
                                      trainable=True)

        self.bias = self.add_weight('bias',
                                    shape=(self.filters,),
                                    initializer=self.bias_initializer,
                                    trainable=True)

        center = self.kernel_size // 2

        mask = np.ones(self.kernel.shape, dtype=np.float32)
        mask[center, center, center + (self.mask_type == 'B'):, :, :] = 0.  # centre depth layer, center row
        mask[center, center + 1:, :, :, :] = 0.  # center depth layer, lower row
        mask[center + 1:, :, :, :, :] = 0.  # behind layers,all row, columns

        self.mask = tf.constant(mask, dtype=tf.float32, name='mask')

    def call(self, input):
        masked_kernel = tf.math.multiply(self.mask, self.kernel)
        x = nn.conv3d(input,
                      masked_kernel,
                      strides=[1, self.strides, self.strides, self.strides, 1],
                      padding=self.padding)
        x = nn.bias_add(x, self.bias)
        return x


class ResidualBlock(keras.Model):

    def __init__(self, h):
        super(ResidualBlock, self).__init__(name='')

        self.conv2a = keras.layers.Conv3D(filters=h, kernel_size=1, strides=1)
        self.conv2b = MaskedConv3D(mask_type='B', filters=h, kernel_size=5, strides=1)
        self.conv2c = keras.layers.Conv3D(filters=2 * h, kernel_size=1, strides=1)

    def call(self, input_tensor):
        x = nn.relu(input_tensor)
        x = self.conv2a(x)

        x = nn.relu(x)
        x = self.conv2b(x)

        x = nn.relu(x)
        x = self.conv2c(x)

        x += input_tensor
        return x


def compute_acc(y_true, y_predict,loss,fl,writer,step):
    if not fl:
        y_true = tf.argmax(y_true, axis=4)
        y_predict = tf.argmax(y_predict, axis=4)
    else:
        y_predict =tf.math.round(y_predict)
    tp = tf.math.count_nonzero(y_predict * y_true, dtype=tf.float32)
    tn = tf.math.count_nonzero((y_predict - 1) * (y_true - 1), dtype=tf.float32)
    fp = tf.math.count_nonzero(y_predict * (y_true - 1), dtype=tf.float32)
    fn = tf.math.count_nonzero((y_predict - 1) * y_true, dtype=tf.float32)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp)
    f1_score = (2 * precision * recall) / (precision + recall)
    with writer.as_default():
        tf.summary.scalar("bc/loss", loss, step)
        tf.summary.scalar("bc/precision", precision,step)
        tf.summary.scalar("bc/recall", recall,step)
        tf.summary.scalar("bc/accuracy", accuracy,step)
        tf.summary.scalar("bc/specificity", specificity,step)
        tf.summary.scalar("bc/f1_score", f1_score,step)
    a = [tp, tn, fp, fn, precision, recall, accuracy, specificity, f1_score]
    return a


class VoxelDNN():
    def __init__(self, depth=64, height=64, width=64, n_channel=1, output_channel=2,residual_blocks=2):
        self.depth = depth
        self.height = height
        self.width = width
        self.n_channel = n_channel
        self.output_channel = output_channel
        self.residual_blocks=residual_blocks
        self.init__ = super(VoxelDNN, self).__init__()

    def build_voxelDNN_model(self):
        # Creating model
        inputs = keras.layers.Input(shape=(self.depth, self.height, self.width, self.n_channel))
        x = MaskedConv3D(mask_type='A', filters=64, kernel_size=7, strides=1)(inputs)
        for i in range(self.residual_blocks):
            x = ResidualBlock(h=32)(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = MaskedConv3D(mask_type='B', filters=64, kernel_size=1, strides=1)(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = MaskedConv3D(mask_type='B', filters=self.output_channel, kernel_size=1, strides=1)(x)
        #x = nn.softmax(x, axis=-1)#add or remove softmax here
        voxelDNN = keras.Model(inputs=inputs, outputs=x)

        voxelDNN.summary()
        return voxelDNN

    def build_voxelDNN_model2(self):
        # Creating model
        inputs = keras.layers.Input(shape=(self.depth, self.height, self.width, self.n_channel))
        x = MaskedConv3D(mask_type='A', filters=128, kernel_size=7, strides=1)(inputs)
        #x = MaskedConv3D(mask_type='B', filters=32, kernel_size=7, strides=1)(x)
        for i in range(self.residual_blocks):
            x = ResidualBlock(h=64)(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = MaskedConv3D(mask_type='B', filters=128, kernel_size=1, strides=1)(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = MaskedConv3D(mask_type='B', filters=self.output_channel, kernel_size=1, strides=1)(x)
        #x = nn.softmax(x, axis=-1)#add or remove softmax here
        voxelDNN = keras.Model(inputs=inputs, outputs=x)

        voxelDNN.summary()
        return voxelDNN

    def calling_dataset(self,training_dirs, batch_size=2):
        # loading data and # Creating input stream using tf.data API
        # training_dir="../../database/ModelNet40_200_pc512_oct3_4k/**/*.ply"
        # batch_size=2
        files=[]
        for training_dir in training_dirs:
            training_dir=training_dir+'**/*.ply'
            p_min, p_max, dense_tensor_shape = get_shape_data(self.depth, 'channels_last')
            paths = get_files(training_dir)
            files=np.concatenate((files,paths), axis=0)
        assert len(files) > 0
        rn.shuffle(files)  # shuffle file
        print('Total blocks for training: ', len(files))
        points = load_points(files)

        files_cat = np.array([os.path.split(os.path.split(x)[0])[1] for x in files])
        points_train = points[files_cat == 'train']
        points_val = points[files_cat == 'test']
        number_training_data = len(points_train)
        train_dataset = input_fn(points_train, batch_size, dense_tensor_shape, 'channels_last', repeat=False,
                                 shuffle=True)
        # train_dataset=train_dataset.batch(batch_size)
        test_dataset = input_fn(points_val, batch_size, dense_tensor_shape, 'channels_last', repeat=False, shuffle=True)
        # train_dataset = input_fn_gaussain(points_train, batch_size, dense_tensor_shape, 'channels_last',gaussain_power=1, repeat=False,
        #                          shuffle=True)
        # # train_dataset=train_dataset.batch(batch_size)
        # test_dataset = input_fn_gaussain(points_val, batch_size, dense_tensor_shape, 'channels_last',gaussain_power=0, repeat=False, shuffle=True)
        return train_dataset, test_dataset,number_training_data

    # optimization
    # Prepare optimizer and loss function
    def focal_loss(self,gamma=2., alpha=0.5):

        # gamma = float(gamma)
        # alpha = float(alpha)
        # gamma = 2.
        # alpha = 0.9

        def focal_loss_fixed(y_true, y_pred):
            """Focal loss for multi-classification
            FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
            Notice: y_pred is probability after softmax
            gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
            d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
            Focal Loss for Dense Object Detection
            https://arxiv.org/abs/1708.02002

            Arguments:
                y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
                y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

            Keyword Arguments:
                gamma {float} -- (default: {2.0})
                alpha {float} -- (default: {4.0})

            Returns:
                [tensor] -- loss.
            """
            epsilon = 1.e-9
            y_true = tf.convert_to_tensor(y_true, tf.float32)
            y_pred = tf.convert_to_tensor(y_pred, tf.float32)

            model_out = tf.add(y_pred, epsilon)
            ce = tf.multiply(y_true, -tf.math.log(model_out))
            weight = tf.multiply(y_true, tf.math.pow(tf.subtract(1., model_out), gamma))
            fl = tf.math.multiply(alpha, tf.math.multiply(weight, ce))
            reduced_fl = tf.math.reduce_max(fl, axis=1)
            return tf.math.reduce_mean(reduced_fl)

        def focal_loss2(y_true, y_pred,alpha):
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

            pt_1 = K.clip(pt_1, 1e-3, .999)
            pt_0 = K.clip(pt_0, 1e-3, .999)
            loss=-K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
                (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
            # with writer.as_default():
            #     tf.summary.scalar("bc/loss", loss,step)
            return loss


        return focal_loss2




    def train_voxelDNN(self,batch,epochs,fl, model_path,saved_model,dataset):
        #log directory
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = model_path+'log' + current_time + '/train'
        test_log_dir = model_path + 'log' + current_time + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        voxelDNN = self.build_voxelDNN_model()
        [train_dataset, test_dataset,number_training_data] = self.calling_dataset(training_dirs=dataset, batch_size=batch)
        lr_decay = 0.999995
        learning_rate = 1e-3
        optimizer = tf.optimizers.Adam(lr=learning_rate)
        if not fl:
            compute_loss = keras.losses.CategoricalCrossentropy(from_logits=True, )
        else:
            #focal loss:
            compute_loss = self.focal_loss()
        # Training loop
        n_epochs = epochs
        n_iter = int(number_training_data / batch)
        #n_iter=int(2000/batch)
        vars_to_save = {"Weight_biases": voxelDNN.trainable_variables, "optimizer": optimizer}
        checkpoint = tf.train.Checkpoint(**vars_to_save)
        losses = [1.,1.]
        try: 
           vars_to_load = {"Weight_biases": voxelDNN.trainable_variables, "optimizer": optimizer}
           checkpoint = tf.train.Checkpoint(**vars_to_load)
           # Restore variables from latest checkpoint.
           #saved_model_path='Model/voxelDNN_scratch/'
           latest_ckpt = tf.train.latest_checkpoint(saved_model)
           checkpoint.restore(latest_ckpt)
           print('Loaded last checkpoint')
        except:
           print('training from scratch')
        for epoch in range(n_epochs):
            progbar = Progbar(n_iter)
            print('Epoch {:}/{:}'.format(epoch + 1, n_epochs))
            loss_per_epochs=[]
            #training dataset
            i=0
            # if(epoch==0):
            #     optimizer.lr = learning_rate
            for i_iter, batch_x in enumerate(train_dataset):
                batch_y = tf.cast(batch_x, tf.int32)
                #    print(batch_x.shape,batch_y.shape)
                # if(i_iter%10==0):
                #     optimizer.lr = optimizer.lr * lr_decay
                #training step
                with tf.GradientTape() as ae_tape:

                    logits = voxelDNN(batch_x, training=True)
                    if fl:
                        #print('training with focal loss')
                        batch_y = tf.cast(batch_y, tf.float32)
                        logits = tf.nn.softmax(logits, axis=-1)[:,:,:,:,1]
                        logits=tf.reshape(logits,
                                   (logits.shape[0], self.depth, self.height, self.width, 1))
                        y_true=tf.reshape(batch_y,
                                            (batch_y.shape[0], self.depth, self.height, self.width, 1))

                        #print(logits[0,1,1,0:10,:],y_true[0,1,1,0:10,:])
                    else:
                        y_true = tf.one_hot(batch_y, self.output_channel)
                        y_true = tf.reshape(y_true,
                                            (y_true.shape[0], self.depth, self.height, self.width, self.output_channel))
                    loss = compute_loss(y_true, logits)

                    metrics = compute_acc(y_true, logits,loss,fl,train_summary_writer,int(epoch*n_iter+i_iter))
                gradients = ae_tape.gradient(loss, voxelDNN.trainable_variables)
                gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                optimizer.apply_gradients(zip(gradients, voxelDNN.trainable_variables))
                loss_per_epochs.append(loss)
                progbar.add(1, values=[('loss', loss),('f1', metrics[8])])
                # i+=1
                # if(i>n_iter):
                #     break
            avg_train_loss=np.average(loss_per_epochs)
            losses.append(avg_train_loss)
            print("\nTraining result on epoch: %i, train loss: %f " % (epoch, avg_train_loss))
            # Testing dataset
            test_losses=[]
            test_metrics=[]
            #i=0
            for i_iter, batch_x in enumerate(test_dataset):
                batch_y = tf.cast(batch_x, tf.int32)
                logits = voxelDNN(batch_x, training=True)
                if fl:
                    batch_y = tf.cast(batch_y, tf.float32)
                    logits = tf.nn.softmax(logits, axis=-1)[:, :, :, :, 1]
                    logits = tf.reshape(logits,
                                        (logits.shape[0], self.depth, self.height, self.width, 1))
                    y_true = tf.reshape(batch_y,
                                        (batch_y.shape[0], self.depth, self.height, self.width, 1))

                else:
                    y_true = tf.one_hot(batch_y, self.output_channel)
                    y_true = tf.reshape(y_true,
                                        (y_true.shape[0], self.depth, self.height, self.width, self.output_channel))

                loss = compute_loss(y_true, logits)

                #loss = compute_loss(y_true, logits,test_summary_writer,i_iter)
                metrics = compute_acc(y_true, logits,loss,fl,test_summary_writer,i_iter)
                test_losses.append(loss)
                test_metrics.append(metrics)
                # i+=1
                # if(i>4):
                #     break

            test_metrics=np.asarray(test_metrics)
            avg_metrics=np.average(test_metrics,axis=0)
            avg_test_loss=np.average(test_losses)
            #print results

            print("Testing result on epoch: %i, test loss: %f " % (epoch, avg_test_loss))
            tf.print(' tp: ', avg_metrics[0], ' tn: ', avg_metrics[1], ' fp: ', avg_metrics[2], ' fn: ', avg_metrics[3],
                     ' precision: ', avg_metrics[4], ' recall: ', avg_metrics[5], ' accuracy: ', avg_metrics[6],
                     ' specificity ', avg_metrics[7], ' f1 ', avg_metrics[8], output_stream=sys.stdout)
            if(avg_train_loss<=losses[-2]):
                saver = tf.train.CheckpointManager(checkpoint, checkpoint_name='ckpt_train_loss_%f_test_loss_%f' % (avg_train_loss, avg_test_loss),
                                                   directory=model_path, max_to_keep=10)
                saver.save()
                print('Saved model')
            else:
                print('Not saving model')
            #break
        # plt.plot(losses)
        # plt.xlabel('Epochs')
        # plt.title('Loss')
        # plt.savefig(loss_img_path)

    def restore_voxelDNN(self,model_path):
        voxelDNN = self.build_voxelDNN_model()
        voxelDNN.summary()
        learning_rate = 1e-3
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        vars_to_load = {"Weight_biases": voxelDNN.trainable_variables, "optimizer": optimizer}
        checkpoint = tf.train.Checkpoint(**vars_to_load)
        # Restore variables from latest checkpoint.
        latest_ckpt = tf.train.latest_checkpoint(model_path)
        checkpoint.restore(latest_ckpt)
        print('Loaded last checkpoint')
        return voxelDNN
if __name__ == "__main__":
    # Command line main application function.
    parser = argparse.ArgumentParser(description='Encoding octree')
    parser.add_argument("-fl", '--focal_loss', type=int,
                        default=0,
                        help='focaloss')
    parser.add_argument("-blocksize", '--block_size', type=int,
                        default=64,
                        help='input size of block')
    parser.add_argument("-batch", '--batch_size', type=int,
                        default=2,
                        help='batch size')
    parser.add_argument("-epochs", '--epochs', type=int,
                        default=2,
                        help='number of training epochs')
    parser.add_argument("-inputmodel", '--savedmodel', type=str, help='path to saved model file')
    #parser.add_argument("-loss", '--loss_img_name', type=str, help='name of loss image')
    parser.add_argument("-outputmodel", '--saving_model_path', type=str, help='path to output model file')
    parser.add_argument("-dataset", '--dataset', action='append', type=str, help='path to dataset ')
    args = parser.parse_args()
    #batch, epochs, loss_img_path, model_path=[args.batch_size,args.epochs,args.loss_img_name,args.saving_model_path]
    voxelDNN=VoxelDNN(depth=64,height=64,width=64,n_channel=1,output_channel=2,residual_blocks=2)
    voxelDNN.train_voxelDNN(args.batch_size,args.epochs,args.focal_loss,args.saving_model_path,args.savedmodel, args.dataset)
