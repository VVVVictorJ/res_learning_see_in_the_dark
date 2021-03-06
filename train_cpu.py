import os,time,scipy.io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import rawpy
import glob
from network import *
from pack import pack_raw
import cv2
from PIL import Image



input_dir = 'H:/sony/Sony/Sony/short/'
gt_dir = 'H:/sony/Sony_gt_16bitPNG/gt/'
checkpoint_dir = './result/'
result_dir= './result/'


# get train IDs
train_fns = glob.glob(gt_dir + '0*.png')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]
print(train_ids)
ps = 512  # patch size for training
save_freq = 2

sess =tf.Session(config=tf.ConfigProto(
    device_count={"CPU":4},inter_op_parallelism_threads=0,
    intra_op_parallelism_threads=0,
))
in_image = tf.placeholder(tf.float32,[None,None,None,4])
gt_image = tf.placeholder(tf.float32,[None,None,None,3])
out_image = res_unet(in_image)

G_Loss = tf.reduce_mean(tf.abs(out_image-gt_image))

t_vars = tf.trainable_variables()

lr = tf.placeholder(tf.float32)

G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_Loss)

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

print("before ckpt")

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

if ckpt:
    print('load '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

gt_images = [None] * 6000
input_images = {}
input_images['300'] = [None] * len(train_ids)
input_images['250'] = [None] * len(train_ids)
input_images['100'] = [None] * len(train_ids)

g_loss = np.zeros((5000, 1))

allfolders = glob.glob(result_dir + '*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

learning_rate = 1e-4
for epoch in range(lastepoch, 4001):
    if os.path.isdir(result_dir + '%04d' % epoch):
        continue
    cnt = 0
    if epoch > 2000:
        learning_rate = 1e-5
    print("start epoch")

    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id

            train_id = train_ids[ind]
            in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
            in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
            in_fn = os.path.basename(in_path)

            gt_files = glob.glob(gt_dir + '%05d_00*.png' % train_id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])#groudtruth
            ratio = min(gt_exposure / in_exposure, 300)

            st = time.time()
            cnt += 1

            if input_images[str(ratio)[0:3]][ind] is None:
                raw = rawpy.imread(in_path)
                input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio
                im = Image.open(gt_path)
                im = np.asarray(im)
                gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

            # crop
            H = input_images[str(ratio)[0:3]][ind].shape[1]
            W = input_images[str(ratio)[0:3]][ind].shape[2]

            xx = np.random.randint(0, W - ps)
            yy = np.random.randint(0, H - ps)
            input_patch = input_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
            gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

            if np.random.randint(2, size=1)[0] == 1:  # random flip
                input_patch = np.flip(input_patch, axis=1)
                gt_patch = np.flip(gt_patch, axis=1)
            if np.random.randint(2, size=1)[0] == 1:
                input_patch = np.flip(input_patch, axis=2)
                gt_patch = np.flip(gt_patch, axis=2)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                input_patch = np.transpose(input_patch, (0, 2, 1, 3))
                gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

            input_patch = np.minimum(input_patch, 1.0)

            _, G_current, output = sess.run([G_opt, G_Loss, out_image],
                                            feed_dict={in_image: input_patch, gt_image: gt_patch, lr: learning_rate})
            output = np.minimum(np.maximum(output, 0), 1)
            g_loss[ind] = G_current

            print("%d %d Loss=%.3f Time=%.3f" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st))

            if epoch % save_freq == 0:
                if not os.path.isdir(result_dir + '%04d' % epoch):
                    os.makedirs(result_dir + '%04d' % epoch)

                temp = np.concatenate((gt_patch[0, :, :, :], output[0, :, :, :]), axis=1)
                scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                    result_dir + '%04d/%05d_00_train_%d.jpg' % (epoch, train_id, ratio))
                # Image.fromarray((output * 255).astype(np.uint8)).save(
                #     result_dir + '%04d/%05d_00_train_%d.jpg' % (epoch, train_id, ratio))

    saver.save(sess, checkpoint_dir + 'model.ckpt')
