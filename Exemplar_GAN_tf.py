#!/usr/bin/env python3
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import BatchNormalization as batch_norm
import warnings
import os
import errno
import numpy as np
import json
import logging
import subprocess
import uuid
import sys
import imageio
from skimage.transform import resize
import logging
import logging.handlers as handlers
import sys
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

## Ops.py-------------------------------------------------------------------------------------------
def log_sum_exp(x, axis=1):
    m = tf.reduce_max(x, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x - m), axis=axis))

#the implements of leakyRelu
def lrelu(x, alpha= 0.2, name="LeakyReLU"):
    return tf.maximum(x , alpha*x)

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h= 2, d_w=2, stddev=0.02, spectural_normed=False,
           name="conv2d", padding='SAME'):
    with tf.variable_scope(name):

        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        if spectural_normed:
            conv = tf.nn.conv2d(input_, spectral_norm(w), strides=[1, d_h, d_w, 1], padding=padding)
        else:
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def instance_norm(input, scope="instance_norm"):

    with tf.variable_scope(scope):

        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)

        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv

        return scale * normalized + offset

def weight_normalization(weight, scope='weight_norm'):

  """based upon openai's https://github.com/openai/generating-reviews-discovering-sentiment/blob/master/encoder.py"""
  weight_shape_list = weight.get_shape().as_list()
  if len(weight.get_shape()) == 2: #I think you want to sum on axis [0,1,2]
    g_shape = [weight_shape_list[1]]
  else:
    raise ValueError('dimensions unacceptable for weight normalization')

  with tf.variable_scope(scope):

    g = tf.get_variable('g_scalar', shape=g_shape, initializer = tf.ones_initializer())
    weight = g * tf.nn.l2_normalize(weight, dim=0)

    return weight

def de_conv(input_, output_shape,
             k_h=5, k_w=5, d_h=2.0, d_w=2.0, stddev=0.02,
             name="deconv2d", with_w=False):

    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=stddev))
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1.0, d_h, d_w, 1.0])
        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], tf.float32, initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases

        else:
            return deconv

def avgpool2d(x, k=2):
    return tf.nn.avg_pool(x, ksize=[1, k, k ,1], strides=[1, k, k, 1], padding='SAME')

def upscale(x, scale):
    _, h, w, _ = get_conv_shape(x)
    return resize_nearest_neighbor(x, (h * scale, w * scale))

def get_conv_shape(tensor):
    shape = int_shape(tensor)
    return shape

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def resize_nearest_neighbor(x, new_size):
    x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def fully_connect(input_, output_size, scope=None, stddev=0.02, spectural_normed=True,
                  bias_start=0.0, with_w=False):

  shape = input_.get_shape().as_list()
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size], tf.float32,
      initializer=tf.constant_initializer(bias_start))

    if spectural_normed:
        mul = tf.matmul(input_, spectral_norm(matrix))
    else:
        mul = tf.matmul(input_, matrix)
    if with_w:
      return mul + bias, matrix, bias
    else:
      return mul + bias

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3 , [x , y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2] , y_shapes[3]])])

def batch_normal(input , scope="scope" , reuse=False):
    return batch_norm(input , epsilon=1e-5, decay=0.9 , scale=True, scope=scope , reuse=reuse, fused=True, updates_collections=None)

def Residual(x, output_dims=256, kernel=3, strides=1, residual_name='resi'):

    with tf.variable_scope('residual_{}'.format(residual_name)) as scope:

        conv1 = instance_norm(conv2d(x, output_dims, k_h=kernel, k_w=kernel, d_h=strides, d_w=strides, name="conv1"), scope='in1')
        conv2 = instance_norm(conv2d(tf.nn.relu(conv1), output_dims, k_h=kernel, k_w=kernel,
                                     d_h=strides, d_w=strides, name="conv2"), scope='in2')
        resi = x + conv2
        return tf.nn.relu(resi)

NO_OPS = 'NO_OPS'

def _l2normalize(v, eps=1e-12):
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_norm(w, iteration= 1):

    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    # w = tf.reshape(w, [1, w.shape.as_list()[0] * w.shape.as_list()[1]])

    u = tf.get_variable("u", [1, w.shape.as_list()[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None

    for i in range(iteration):

        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = _l2normalize(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = _l2normalize(u_)

    #real_sn = tf.svd(w, compute_uv=False)[...,0]
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma
    #Get the real spectral norm
    #real_sn_after = tf.svd(w_norm, compute_uv=False)[..., 0]

    #frobenius norm
    #f_norm = tf.norm(w, ord='fro', axis=[0, 1])

    #tf.summary.scalar("real_sn", real_sn)
    tf.summary.scalar("powder_sigma", tf.reduce_mean(sigma))
    #tf.summary.scalar("real_sn_afterln", real_sn_after)
    #tf.summary.scalar("f_norm", f_norm)

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm
# --------------------------------------------------------------------------
# Utils.py...
# --------------------------------------------------------------------------

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_image(image_path , image_size, is_crop= True, resize_w= 64, is_grayscale= False, is_test=False):
    return transform(imread(image_path , is_grayscale), image_size, is_crop , resize_w, is_test=is_test)

def transform(image, npx = 64 , is_crop=False, resize_w=64, is_test=False):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image , npx , resize_w=resize_w, is_test=is_test)
    else:
        cropped_image = image
        cropped_image = resize(cropped_image ,
                            [resize_w , resize_w])
    return np.array(cropped_image)/127.5 - 1

def center_crop(x, crop_h , crop_w=None, resize_w=64, is_test=False):

    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))

    if not is_test:
        rate = np.random.uniform(0, 1, size=1)
        if rate < 0.5:
            x = np.fliplr(x)
    # return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
    #                            [resize_w, resize_w])
    return resize(x[20:218 - 20, 0: 178], [resize_w, resize_w])

def save_images(images, size, image_path, is_ouput=False):
    return imsave(inverse_transform(images, is_ouput), size, image_path)

def imread(path, is_grayscale=False):
    if (is_grayscale):
        return imageio.imread(path, flatten=True).astype(np.float)
    else:
        return imageio.imread(path).astype(np.float)

def imsave(images, size, path):
    return imageio.imsave(path, merge(images, size))

def merge(images, size):

    if size[0] + size[1] == 2:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w: i * w + w, :] = image

    else:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w: i * w + w, :] = image

    return img

def inverse_transform(image, is_ouput=False):

    if is_ouput == True:
        print(image[0])
        logger.debug(image[0])
    result = ((image + 1) * 127.5).astype(np.uint8)
    if is_ouput == True:
        print(result)
        logger.debug(result)
    return result

log_interval = 1000

def read_image_list_for_Eyes(category):

    json_cat = category + "/data.json"
    with open(json_cat, 'r') as f:
        data = json.load(f)
    non_usable_images = 0
    total_images = 0
    all_iden_info = []
    all_ref_info = []

    test_all_iden_info = []
    test_all_ref_info = []

    #c: id
    #k: name of identity
    #v: details.
            
    for c, (k, v) in enumerate(data.items()):
        identity_info = []

        is_close = False
        is_close_id = 0

        if c % log_interval == 0:
            print('Processed {}/{}'.format(c, len(data)))
            logger.debug('Processed {}/{}'.format(c, len(data)))
        number_of_images = len(v)
        if number_of_images < 2:
            continue
        do_not_use = False
        files_to_skip = []
        for i in range(len(v)):
            total_images += 1
            if not os.path.exists(category + "/" +v[i]['filename']):
                number_of_images -= 1
                non_usable_images += 1
                files_to_skip.append(v[i]['filename'])
                if number_of_images < 2:
                    non_usable_images += 1
                    do_not_use = True
                    break

        if do_not_use:
            continue
        skipped = 0
        for i in range(len(v)):
            if v[i]['filename'] in files_to_skip:
                skipped += 1
                continue
            if is_close or v[i]['opened'] is None or v[i]['opened'] < 0.60:
                is_close = True
            if v[i]['opened'] is None or v[i]['opened'] < 0.60:
                is_close_id = i - skipped

            str_info = str(v[i]['filename']) + "_"

            if 'eye_left' in v[i] and v[i]['eye_left'] != None:
                str_info += str(v[i]['eye_left']['y']) + "_"
                str_info += str(v[i]['eye_left']['x']) + "_"
            else:
                str_info += str(0) + "_"
                str_info += str(0) + "_"

            if 'box_left' in v[i] and v[i]['box_left'] != None:
                str_info += str(v[i]['box_left']['h']) + "_"
                str_info += str(v[i]['box_left']['w']) + "_"
            else:
                str_info += str(0) + "_"
                str_info += str(0) + "_"

            if 'eye_right' in v[i] and v[i]['eye_right'] != None:
                str_info += str(v[i]['eye_right']['y']) + "_"
                str_info += str(v[i]['eye_right']['x']) + "_"
            else:
                str_info += str(0) + "_"
                str_info += str(0) + "_"

            if 'box_right' in v[i] and v[i]['box_right'] != None:
                str_info += str(v[i]['box_right']['h']) + "_"
                str_info += str(v[i]['box_right']['w'])
            else:
                str_info += str(0) + "_"
                str_info += str(0)

            identity_info.append(str_info)

        if is_close == False:
            # logger.info("close_id", is_close_id)
            # logger.info("v", len(v))
            # logger.info("FTS", len(files_to_skip))
            # logger.info("id", len(identity_info))
            for j in range(len(v)):
                if v[j]['filename'] in files_to_skip:
                    continue
                
                first_n = np.random.randint(0, len(v) - len(files_to_skip), size=1)[0]
                all_iden_info.append(identity_info[first_n])
                middle_value = identity_info[first_n]
                identity_info.remove(middle_value)

                second_n = np.random.randint(0, len(v) - len(files_to_skip) - 1, size=1)[0]
                all_ref_info.append(identity_info[second_n])

                identity_info.append(middle_value)

        else:

            #append twice with different reference result.
            # logger.info("close_id", is_close_id)
            # logger.info("v", len(v))
            # logger.info("FTS", len(files_to_skip))
            # logger.info("id", len(identity_info))
            middle_value = identity_info[is_close_id]
            test_all_iden_info.append(middle_value)
            identity_info.remove(middle_value)

            second_n = np.random.randint(0, len(v) - len(files_to_skip) - 1, size=1)[0]
            test_all_ref_info.append(identity_info[second_n])

            test_all_iden_info.append(middle_value)

            second_n = np.random.randint(0, len(v) - len(files_to_skip) - 1, size=1)[0]
            test_all_ref_info.append(identity_info[second_n])

    assert len(all_iden_info) == len(all_ref_info)
    assert len(test_all_iden_info) == len(test_all_ref_info)

    print("train_data: {}".format(len(all_iden_info)))
    logger.debug("train_data: {}".format(len(all_iden_info)))
    print("test_data: {}".format(len(test_all_iden_info)))
    logger.debug("test_data: {}".format(len(test_all_iden_info)))
    return all_iden_info, all_ref_info, test_all_iden_info, test_all_ref_info, total_images, non_usable_images

class Eyes(object):

    def __init__(self, image_path):
        self.dataname = "Eyes"
        self.image_size = 256
        self.channel = 3
        self.image_path = image_path
        self.dims = self.image_size*self.image_size
        self.shape = [self.image_size, self.image_size, self.channel]
        self.train_images_name, self.train_eye_pos_name, self.train_ref_images_name, self.train_ref_pos_name, \
            self.test_images_name, self.test_eye_pos_name, self.test_ref_images_name, self.test_ref_pos_name = self.load_Eyes(image_path)

    def load_Eyes(self, image_path):

        images_list, images_ref_list, test_images_list, test_images_ref_list, total_images, non_usable_images = read_image_list_for_Eyes(image_path)
        print("Read all images. Total images:", total_images, ' Usable images: ', total_images - non_usable_images )
        logger.debug("Read all images. Total images: {}. Usable images: {}".format(total_images, total_images - non_usable_images ))
        train_images_name = []
        train_eye_pos_name = []
        train_ref_images_name = []
        train_ref_pos_name = []

        test_images_name = []
        test_eye_pos_name = []
        test_ref_images_name = []
        test_ref_pos_name = []

        #train
        for images_info_str in images_list:

            eye_pos = []
            image_name, left_eye_x, left_eye_y, left_eye_h, left_eye_w, right_eye_x, right_eye_y,\
                right_eye_h, right_eye_w = images_info_str.split('_', 9)
            eye_pos.append((int(left_eye_x), int(left_eye_y), int(left_eye_h), int(left_eye_w), int(right_eye_x),
                            int(right_eye_y), int(right_eye_h), int(right_eye_w)))
            image_name = os.path.join(self.image_path, image_name)

            train_images_name.append(image_name)
            train_eye_pos_name.append(eye_pos)

        for images_info_str in images_ref_list:

            eye_pos = []
            image_name, left_eye_x, left_eye_y, left_eye_h, left_eye_w, right_eye_x, right_eye_y, \
                right_eye_h, right_eye_w = images_info_str.split('_', 9)

            eye_pos.append((int(left_eye_x), int(left_eye_y), int(left_eye_h), int(left_eye_w), int(right_eye_x),
                            int(right_eye_y), int(right_eye_h), int(right_eye_w)))

            image_name = os.path.join(self.image_path, image_name)
            train_ref_images_name.append(image_name)
            train_ref_pos_name.append(eye_pos)

        for images_info_str in test_images_list:

            eye_pos = []
            image_name, left_eye_x, left_eye_y, left_eye_h, left_eye_w, right_eye_x, right_eye_y, \
            right_eye_h, right_eye_w = images_info_str.split('_', 9)
            eye_pos.append(
                (int(left_eye_x), int(left_eye_y), int(left_eye_h), int(left_eye_w), int(right_eye_x),
                 int(right_eye_y), int(right_eye_h), int(right_eye_w)))
            image_name = os.path.join(self.image_path, image_name)

            test_images_name.append(image_name)
            test_eye_pos_name.append(eye_pos)

        for images_info_str in test_images_ref_list:

            eye_pos = []
            image_name, left_eye_x, left_eye_y, left_eye_h, left_eye_w, right_eye_x, right_eye_y, \
            right_eye_h, right_eye_w = images_info_str.split('_', 9)
            eye_pos.append(
                (int(left_eye_x), int(left_eye_y), int(left_eye_h), int(left_eye_w), int(right_eye_x),
                 int(right_eye_y), int(right_eye_h), int(right_eye_w)))
            image_name = os.path.join(self.image_path, image_name)
            test_ref_images_name.append(image_name)
            test_ref_pos_name.append(eye_pos)

        assert len(train_images_name) == len(train_eye_pos_name) == len(train_ref_images_name) == len(train_ref_pos_name)
        assert len(test_images_name) == len(test_eye_pos_name) == len(test_ref_images_name) == len(test_ref_pos_name)

        return train_images_name, train_eye_pos_name, train_ref_images_name, train_ref_pos_name, \
               test_images_name, test_eye_pos_name, test_ref_images_name, test_ref_pos_name

    def getShapeForData(self, filenames, is_test=False):

        array = [get_image(batch_file, 108, is_crop=False, resize_w=256,
                           is_grayscale=False, is_test=is_test) for batch_file in filenames]
        sample_images = np.array(array)
        return sample_images

    def getNextBatch(self, batch_num=0, batch_size=64, is_shuffle=True):

        ro_num = len(self.train_images_name) / batch_size
        if batch_num % ro_num == 0 and is_shuffle:

            length = len(self.train_images_name)
            perm = np.arange(length)
            np.random.shuffle(perm)

            self.train_images_name = np.array(self.train_images_name)
            self.train_images_name = self.train_images_name[perm]

            self.train_eye_pos_name = np.array(self.train_eye_pos_name)
            self.train_eye_pos_name = self.train_eye_pos_name[perm]

            self.train_ref_images_name = np.array(self.train_ref_images_name)
            self.train_ref_images_name = self.train_ref_images_name[perm]

            self.train_ref_pos_name = np.array(self.train_ref_pos_name)
            self.train_ref_pos_name = self.train_ref_pos_name[perm]

        return self.train_images_name[int((batch_num % ro_num) * batch_size): int((batch_num % ro_num + 1) * batch_size)], \
               self.train_eye_pos_name[int((batch_num % ro_num) * batch_size): int((batch_num % ro_num + 1) * batch_size)], \
               self.train_ref_images_name[int((batch_num % ro_num) * batch_size): int((batch_num % ro_num + 1) * batch_size)], \
                self.train_ref_pos_name[int((batch_num % ro_num) * batch_size): int((batch_num % ro_num + 1) * batch_size)]

    def getTestNextBatch(self, batch_num=0, batch_size=64, is_shuffle=True):

        ro_num = len(self.test_images_name) / batch_size
        if batch_num == 0 and is_shuffle:

            length = len(self.test_images_name)
            perm = np.arange(length)
            np.random.shuffle(perm)

            self.test_images_name = np.array(self.test_images_name)
            self.test_images_name = self.test_images_name[perm]

            self.test_eye_pos_name = np.array(self.test_eye_pos_name)
            self.test_eye_pos_name = self.test_eye_pos_name[perm]

            self.test_ref_images_name = np.array(self.test_ref_images_name)
            self.test_ref_images_name = self.test_ref_images_name[perm]

            self.test_ref_pos_name = np.array(self.test_ref_pos_name)
            self.test_ref_pos_name = self.test_ref_pos_name[perm]

        return self.test_images_name[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.test_eye_pos_name[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.test_ref_images_name[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.test_ref_pos_name[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size]
# -------------------------------------------------------------------------------
# Exemplar GAN.py...
# -------------------------------------------------------------------------------

class ExemplarGAN(object):

    # build model
    def __init__(self, batch_size, max_iters, model_path, data_ob, sample_path, log_dir, learning_rate, is_load, lam_recon,
                 lam_gp, use_sp, beta1, beta2, n_critic):

        self.batch_size = batch_size
        self.max_iters = max_iters
        self.model_path = model_path
        self.data_ob = data_ob
        self.sample_path = sample_path
        self.log_dir = log_dir
        self.learning_rate = learning_rate
        self.log_vars = []
        self.channel = data_ob.channel
        self.shape = data_ob.shape
        self.lam_recon = lam_recon
        self.lam_gp = lam_gp
        self.use_sp = use_sp
        self.is_load = is_load
        self.beta1 = beta1
        self.beta2 = beta2
        self.n_critic = n_critic
        self.output_size = data_ob.image_size
        self.input_img = tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        self.exemplar_images = tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        self.img_mask = tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        self.exemplar_mask =  tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        self.domain_label = tf.placeholder(tf.int32, [batch_size])
        self.lr_decay = tf.placeholder(tf.float32, None, name='lr_decay')

    def build_model_GAN(self):

        self.incomplete_img = self.input_img * (1 - self.img_mask)
        self.local_real_img = self.input_img * self.img_mask

        self.x_tilde = self.encode_decode(self.incomplete_img, self.exemplar_images, 1 - self.img_mask, self.exemplar_mask, reuse=False)
        self.local_fake_img = self.x_tilde * self.img_mask

        self.D_real_gan_logits = self.discriminate(self.input_img, self.exemplar_images, self.local_real_img, spectural_normed=self.use_sp, reuse=False)
        self.D_fake_gan_logits = self.discriminate(self.x_tilde, self.exemplar_images, self.local_fake_img, spectural_normed=self.use_sp, reuse=True)

        self.D_loss = self.loss_dis(self.D_real_gan_logits, self.D_fake_gan_logits)
        self.G_gan_loss = self.loss_gen(self.D_fake_gan_logits)

        self.recon_loss = tf.reduce_mean(
            tf.reduce_sum(tf.abs(self.x_tilde - self.input_img), axis=[1, 2, 3]) / (
            self.output_size * self.output_size * self.channel))

        self.G_loss = self.G_gan_loss + self.lam_recon * self.recon_loss

        self.log_vars.append(("D_loss", self.D_loss))
        self.log_vars.append(("G_loss", self.G_loss))

        self.t_vars = tf.trainable_variables()
        self.d_vars = [var for var in self.t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in self.t_vars if 'encode_decode' in var.name]

        print("d_vars", len(self.d_vars))
        logger.debug("d_vars: {}".format(len(self.d_vars)))
        print("e_vars", len(self.g_vars))
        logger.debug("e_vars: {}".format(len(self.g_vars)))

        self.saver = tf.train.Saver()
        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

    def build_test_model_GAN(self):

        self.incomplete_img = self.input_img * (1 - self.img_mask)
        self.x_tilde = self.encode_decode(self.incomplete_img, self.exemplar_images, 1 - self.img_mask, self.exemplar_mask, reuse=False)
        self.t_vars = tf.trainable_variables()
        self.g_vars = [var for var in self.t_vars if 'encode_decode' in var.name]
        self.saver = tf.train.Saver()

    def loss_dis(self, d_real_logits, d_fake_logits):

        l1 = tf.reduce_mean(tf.nn.softplus(-d_real_logits))
        l2 = tf.reduce_mean(tf.nn.softplus(d_fake_logits))

        return l1 + l2

    def loss_gen(self, d_fake_logits):
        return tf.reduce_mean(tf.nn.softplus(-d_fake_logits))

    def test(self, test_step):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            load_step = test_step
            self.saver.restore(sess, os.path.join(self.model_path, 'model_{:06d}.ckpt'.format(load_step)))
            batch_num = len(self.data_ob.test_images_name) / self.batch_size

            for j in range(batch_num):

                test_data_list, batch_eye_pos, test_ex_list, test_eye_pos = self.data_ob.getTestNextBatch(batch_num=j, batch_size=self.batch_size,
                                                                               is_shuffle=False)
                batch_images_array = self.data_ob.getShapeForData(test_data_list, is_test=True)
                batch_exem_array = self.data_ob.getShapeForData(test_ex_list, is_test=True)
                batch_eye_pos = np.squeeze(batch_eye_pos)
                test_eye_pos = np.squeeze(test_eye_pos)
                x_tilde, incomplete_img = sess.run(
                    [self.x_tilde, self.incomplete_img],
                    feed_dict={self.input_img: batch_images_array, self.exemplar_images: batch_exem_array, self.img_mask: self.get_Mask(batch_eye_pos),
                               self.exemplar_mask: self.get_Mask(test_eye_pos)})
                output_concat = np.concatenate(
                    [batch_images_array, batch_exem_array, incomplete_img, x_tilde], axis=0)
                print(output_concat.shape)
                logger.debug(output_concat.shape)
                save_images(output_concat, [output_concat.shape[0] / 4, 4],
                            '{}/{:02d}_output.jpg'.format(self.sample_path, j))

    # do train
    def train(self):

        d_trainer = tf.train.AdamOptimizer(self.learning_rate * self.lr_decay, beta1=self.beta1, beta2=self.beta2)
        d_gradients = d_trainer.compute_gradients(self.D_loss, var_list=self.d_vars)
        opti_D = d_trainer.apply_gradients(d_gradients)

        m_trainer = tf.train.AdamOptimizer(self.learning_rate * self.lr_decay, beta1=self.beta1, beta2=self.beta2)
        m_gradients = m_trainer.compute_gradients(self.G_loss, var_list=self.g_vars)
        opti_M = m_trainer.apply_gradients(m_gradients)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            step = 0
            step2 = 0
            lr_decay = 1

            if self.is_load:
                self.saver.restore(sess, os.path.join(self.model_path, 'model_{:06d}.ckpt'.format(step)))

            while step <= self.max_iters:

                if step > 20000 and lr_decay > 0.1:
                    lr_decay = (self.max_iters - step) / float(self.max_iters - 10000)

                for i in range(self.n_critic):

                    train_data_list, batch_eye_pos, batch_train_ex_list, batch_ex_eye_pos = self.data_ob.getNextBatch(step2, self.batch_size)
                    batch_images_array = self.data_ob.getShapeForData(train_data_list)
                    batch_exem_array = self.data_ob.getShapeForData(batch_train_ex_list)
                    batch_eye_pos = np.squeeze(batch_eye_pos)

                    batch_ex_eye_pos = np.squeeze(batch_ex_eye_pos)
                    f_d = {self.input_img: batch_images_array, self.exemplar_images: batch_exem_array,
                           self.img_mask: self.get_Mask(batch_eye_pos), self.exemplar_mask: self.get_Mask(batch_ex_eye_pos), self.lr_decay: lr_decay}

                    # optimize D
                    sess.run(opti_D, feed_dict=f_d)
                    step2 += 1

                # optimize M
                sess.run(opti_M, feed_dict=f_d)
                summary_str = sess.run(summary_op, feed_dict=f_d)
                summary_writer.add_summary(summary_str, step)

                if step % 50 == 0:
                    d_loss,  g_loss = sess.run([self.D_loss, self.G_loss],
                        feed_dict=f_d)
                    print("step %d / %d , d_loss = %.4f, g_loss=%.4f" % (step, max_iters, d_loss, g_loss))
                    logger.debug("step {} / {} , d_loss = {}, g_loss={}".format(step, max_iters, d_loss, g_loss))

                if np.mod(step, 400) == 0:

                    x_tilde, incomplete_img, local_real, local_fake = sess.run([self.x_tilde, self.incomplete_img, self.local_real_img, self.local_fake_img], feed_dict=f_d)
                    output_concat = np.concatenate([batch_images_array, batch_exem_array, incomplete_img, x_tilde, local_real, local_fake], axis=0)
                    save_images(output_concat, [output_concat.shape[0]/self.batch_size, self.batch_size],
                                            '{}/{:02d}_output.jpg'.format(self.sample_path, step))
                if np.mod(step, 2000) == 0:
                    self.saver.save(sess, os.path.join(self.model_path, 'model_{:06d}.ckpt'.format(step)))

                step += 1

            save_path = self.saver.save(sess, self.model_path)
            print("Model saved in file: %s" % save_path)
            logger.debug("Model saved in file: %s" % save_path)

    def discriminate(self, x_var, x_exemplar, local_x_var, spectural_normed=False, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            if reuse == True:
                scope.reuse_variables()

            conv = tf.concat([x_var, x_exemplar], axis=3)
            for i in range(5):
                output_dim = np.minimum(64 * np.power(2, i+1), 512)
                conv = lrelu(conv2d(conv, spectural_normed=spectural_normed, output_dim=output_dim, name='dis_conv_{}'.format(i)))

            conv = tf.reshape(conv, shape=[self.batch_size, conv.shape[1] * conv.shape[2] * conv.shape[3]])
            ful_global = fully_connect(conv, output_size=output_dim, spectural_normed=spectural_normed, scope='dis_fully1')

            conv = local_x_var
            for i in range(5):
                output_dim = np.minimum(64 * np.power(2, i+1), 512)
                conv = lrelu(conv2d(conv, spectural_normed=spectural_normed, output_dim=output_dim, name='dis_conv_2_{}'.format(i)))

            conv = tf.reshape(conv, shape=[self.batch_size, conv.shape[1] * conv.shape[2] * conv.shape[3]])
            ful_local = fully_connect(conv, output_size=output_dim, spectural_normed=spectural_normed, scope='dis_fully2')

            gan_logits = fully_connect(tf.concat([ful_global, ful_local], axis=1), output_size=1, spectural_normed=spectural_normed, scope='dis_fully3')

            return gan_logits

    def encode_decode(self, x_var, x_exemplar, img_mask, exemplar_mask, reuse=False):

        with tf.variable_scope("encode_decode") as scope:

            if reuse == True:
                scope.reuse_variables()

            x_var = tf.concat([x_var, img_mask, x_exemplar, exemplar_mask], axis=3)

            conv1 = tf.nn.relu(
                instance_norm(conv2d(x_var, output_dim=64, k_w=7, k_h=7, d_w=1, d_h=1, name='e_c1'), scope='e_in1'))
            conv2 = tf.nn.relu(
                instance_norm(conv2d(conv1, output_dim=128, k_w=4, k_h=4, d_w=2, d_h=2, name='e_c2'), scope='e_in2'))
            conv3 = tf.nn.relu(
                instance_norm(conv2d(conv2, output_dim=256, k_w=4, k_h=4, d_w=2, d_h=2, name='e_c3'), scope='e_in3'))

            r1 = Residual(conv3, residual_name='re_1')
            r2 = Residual(r1, residual_name='re_2')
            r3 = Residual(r2, residual_name='re_3')
            r4 = Residual(r3, residual_name='re_4')
            r5 = Residual(r4, residual_name='re_5')
            r6 = Residual(r5, residual_name='re_6')

            g_deconv1 = tf.nn.relu(instance_norm(de_conv(r6, output_shape=[self.batch_size,
                                                                           self.output_size//2, self.output_size//2, 128], name='gen_deconv1'), scope="gen_in"))
            # for 1
            g_deconv_1_1 = tf.nn.relu(instance_norm(de_conv(g_deconv1,
                        output_shape=[self.batch_size, self.output_size, self.output_size, 32], name='g_deconv_1_1'), scope='gen_in_1_1'))

            g_deconv_1_1_x = tf.concat([g_deconv_1_1, x_var], axis=3)
            x_tilde1 = conv2d(g_deconv_1_1_x, output_dim=self.channel, k_w=7, k_h=7, d_h=1, d_w=1, name='gen_conv_1_2')

            return tf.nn.tanh(x_tilde1)

    def get_Mask(self, eye_pos, flag=0):

        eye_pos = eye_pos
        #logger.info eye_pos
        batch_mask = []
        for i in range(self.batch_size):

            current_eye_pos = eye_pos[i]
            #eye
            if flag == 0:
                #left eye, y
                mask = np.zeros(shape=[self.output_size, self.output_size, self.channel])
                scale = current_eye_pos[0] - 25 #current_eye_pos[3] / 2
                down_scale = current_eye_pos[0] + 25 #current_eye_pos[3] / 2
                l1_1 =int(scale)
                u1_1 =int(down_scale)
                #x
                scale = current_eye_pos[1] - 35 #current_eye_pos[2] / 2
                down_scale = current_eye_pos[1] + 35 #current_eye_pos[2] / 2
                l1_2 = int(scale)
                u1_2 = int(down_scale)

                mask[l1_1:u1_1, l1_2:u1_2, :] = 1.0
                #right eye, y
                scale = current_eye_pos[4] - 25 #current_eye_pos[7] / 2
                down_scale = current_eye_pos[4] + 25 #current_eye_pos[7] / 2

                l2_1 = int(scale)
                u2_1 = int(down_scale)

                #x
                scale = current_eye_pos[5] - 35 #current_eye_pos[6] / 2
                down_scale = current_eye_pos[5] + 35 #current_eye_pos[6] / 2
                l2_2 = int(scale)
                u2_2 = int(down_scale)

                mask[l2_1:u2_1, l2_2:u2_2, :] = 1.0

            batch_mask.append(mask)

        return np.array(batch_mask)

# ---------------------------------------------------------
# Main.py...
# ---------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES']= '12'
# tf.reset_default_graph()

OPER_FLAG = 0 # "flag of opertion, test or train")
OPER_NAME = "Experiment_6_21_6" #"name of the experiment")
path = '/home/hlcv_team001/data_aligned' # "path of training data")
batch_size= 4 # "size of single batch")
max_iters= 34000 # "number of total iterations for G")
learn_rate = 0.0001# "learning rate for g and d")
test_step= 34000# "loading setp model for testing")
is_load= False# "whether loading the pretraining model for training")
use_sp = True# "whether using spectral normalization")
lam_recon= 1# "weight for recon loss")
lam_gp= 10# "weight for gradient penalty")
beta1= 0.5# "beta1 of Adam optimizer")
beta2= 0.999# "beta2 of Adam optimizer")
n_critic= 1# "iters of g for every d")

if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG, filename="logfile.txt", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # consoleHandler = logging.StreamHandler(sys.stdout)
    # consoleHandler.setLevel(logging.DEBUG)
    # consoleHandler.setFormatter(formatter)

    fileHandler = handlers.RotatingFileHandler(filename="error.log",maxBytes=1024000, backupCount=10, mode="a")
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(formatter)

    smtpHandler = handlers.SMTPHandler(
                mailhost = ("smtp.gmail.com", 587),
                fromaddr = "email",
                toaddrs = "email",
                subject = "Project "+OPER_NAME,
                credentials = ('email','password'),
                secure = ()
                )
    smtpHandler.setLevel(logging.DEBUG)
    smtpHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(smtpHandler)
    tf.disable_v2_behavior()
   
    # logging.info(FLAGS)
    root_log_dir = "/home/hlcv_team001/Exemplar_GAN/output/log/logs{}".format(OPER_FLAG)
    checkpoint_dir = "/home/hlcv_team001/Exemplar_GAN/output/model_gan{}/".format(OPER_NAME)
    sample_path = "/home/hlcv_team001/Exemplar_GAN/output/sample{}/sample_{}".format(OPER_FLAG, OPER_NAME)

    mkdir_p(root_log_dir)
    mkdir_p(checkpoint_dir)
    mkdir_p(sample_path)
    m_ob = Eyes(path)

    eGan = ExemplarGAN(batch_size= batch_size, max_iters= max_iters,
                    model_path= checkpoint_dir, data_ob= m_ob, sample_path= sample_path , log_dir= root_log_dir,
                    learning_rate=  learn_rate, is_load=is_load, lam_recon=lam_recon, lam_gp=lam_gp,
                    use_sp=use_sp, beta1=beta1, beta2=beta2, n_critic=n_critic)
    print("Created GAN.")
    logger.debug("Created GAN.")
    if OPER_FLAG == 0:
        eGan.build_model_GAN()
        logger.debug("Starting Training.")
        eGan.train()
    if OPER_FLAG == 1:
        eGan.build_test_model_GAN()
        eGan.test(test_step=test_step)

