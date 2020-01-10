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

in_path = 'D:/test/iphone/7p/frame_3/IMG_6395.DNG'

in_fn = os.path.basename(in_path)
ratio = 150

raw = rawpy.imread(in_path)
input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
input_full = np.minimum(input_full, 1.0)

output = sess.run(out_image, feed_dict={in_image: input_full})
output = np.minimum(np.maximum(output, 0), 1)

output = output[0, :, :, :]
Image.fromarray((output*255).astype(np.uint8)).save(result_dir + 'final/10_r_%d_out.png' %  ratio)
print(time.time()-start)