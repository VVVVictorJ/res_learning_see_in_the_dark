import os,time,scipy.io
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
# get test IDs
test_fns = glob.glob(gt_dir + '/1*.png')
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

start = time.time()
sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 4])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
out_image = res_unet(in_image)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

if not os.path.isdir(result_dir + 'final/'):
    os.makedirs(result_dir + 'final/')

for test_id in test_ids:
    # test the first image in each sequence
    in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)
    for k in range(len(in_files)):
        if k<10:
            in_path = in_files[k]
            in_fn = os.path.basename(in_path)
            print(in_fn)
            gt_files = glob.glob(gt_dir + '%05d_00*.png' % test_id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)

            raw = rawpy.imread(in_path)
            input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio

            im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
            scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

            # gt_raw = rawpy.imread(gt_path)
            #im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            im = Image.open(gt_path)
            im = np.asarray(im)
            gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

            input_full = np.minimum(input_full, 1.0)

            output = sess.run(out_image, feed_dict={in_image: input_full})
            output = np.minimum(np.maximum(output, 0), 1)

            output = output[0, :, :, :]
            gt_full = gt_full[0, :, :, :]
            scale_full = scale_full[0, :, :, :]
            scale_full = scale_full * np.mean(gt_full) / np.mean(
                scale_full)  # scale the low-light image to the same mean of the groundtruth

            scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + 'final/%5d_00_%d_out.png' % (test_id, ratio))
            scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + 'final/%5d_00_%d_scale.png' % (test_id, ratio))
            scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + 'final/%5d_00_%d_gt.png' % (test_id, ratio))