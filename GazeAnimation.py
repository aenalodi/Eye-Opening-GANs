#!/usr/bin/env python3
import os, errno
import logging
import logging.handlers as handlers
import sys
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
import argparse
import os
from abc import abstractmethod
import os
import imageio
import scipy.misc as misc
import scipy
import numpy as np
import cv2
import tensorflow.compat.v1 as tf
import tf_slim as slim
from tensorflow.keras.layers import BatchNormalization as batch_norm
from tensorflow.keras.regularizers import L2 as l2_regularizer
import functools
import random

def list_difference(a, b):
    '''
    :param a:
    :param b:
    :return:
    '''
    set_a = set(a)
    set_b = set(b)
    comparison = set_a.difference(set_b)

    return list(comparison)

def mkdir_p(path):
    '''
    :param path:
    :return:
    '''
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def makefolders(subfolders):
    '''
    create multiple folders
    :param subfolders:
    :return:
    '''
    assert isinstance(subfolders, list)

    for path in subfolders:
        if not os.path.exists(path):
            mkdir_p(path)

def setLogConfig():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger



class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--data_dir', type=str, default='/home/hlcv_team001/GazeAnimation/CelebAGaze/', help='path to images')
        parser.add_argument('--pretrain_path', type=str, default='/home/hlcv_team001/GazeAnimation/PAM/', help='pretrained model for pam module mentioned in the paper')
        parser.add_argument('--vgg_path', type=str, default='/home/hlcv_team001/GazeAnimation/vgg_16_2016_08_28/vgg_16.ckpt', help='vgg path for perceptual loss')
        parser.add_argument('--inception_path', type=str, default='/home/hlcv_team001/GazeAnimation/pretrained/')
        parser.add_argument('--img_size', type=int, default=256, help='scale images to this size')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--nef', type=int, default=32, help='# of encoder filters in frist decov layer')
        parser.add_argument('--ngf', type=int, default=16, help='# of generator filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=16, help='# of discriminator filters in first conv layer')
        parser.add_argument('--n_layers_e', type=int, default=3, help='layers of sae')
        parser.add_argument('--n_layers_g', type=int, default=5, help='layers of generator')
        parser.add_argument('--n_layers_d', type=int, default=5, help='layers of d model')
        parser.add_argument('--n_layers_r', type=int, default=3, help='layers of r model')
        parser.add_argument('--n_layers_sr', type=int, default=2, help='layers of sr model')
        parser.add_argument('--norm_sr', type=str, default='In', choices=['In', 'Bn', 'WO'],
                            help='the type of normalization in SR')
        parser.add_argument('--n_blocks', type=int, default=4, help='numbers of blocks')
        parser.add_argument('--gpu_id', type=str, default='2',
                            help='gpu ids: en_blocks.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--exper_name', type=str, default='GazeTraining_1',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--checkpoints_dir', type=str, default='/home/hlcv_team001/GazeAnimation/PAM/', help='models are saved here')
        parser.add_argument('--log_dir', type=str, default='/home/hlcv_team001/GazeAnimation/logs', help='logs for tensorboard')
        parser.add_argument('--capacity', type=int, default=1000, help='capacity for queue in training')
        parser.add_argument('--num_threads', type=int, default=10, help='thread for reading data in training')
        parser.add_argument('--sample_dir', type=str, default='/home/hlcv_team001/GazeAnimation/sample_dir', help='dir for sample images')
        parser.add_argument('--test_sample_dir', type=str, default='/home/hlcv_team001/GazeAnimation/test_sample_dir',
                            help='test sample images are saved here')
        parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
                            help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        parser.add_argument('--ds_n', type=int, default=4, help='scale for downsamping image')
        parser.add_argument('--ds_s', type=int, default=8, help='scale for downsamping ds_image')

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):

        opt.checkpoints_dir = os.path.join(opt.exper_name, opt.checkpoints_dir)
        opt.sample_dir = os.path.join(opt.exper_name, opt.sample_dir)

        opt.test_sample_dir = os.path.join(opt.exper_name, opt.test_sample_dir)
        opt.test_sample_dir0 = os.path.join(opt.test_sample_dir, '0')
        opt.test_sample_dir1 = os.path.join(opt.test_sample_dir, '1')
        opt.test_sample_dir2 = os.path.join(opt.test_sample_dir, '2')

        opt.log_dir = os.path.join(opt.exper_name, opt.log_dir)
        makefolders([opt.inception_path, opt.checkpoints_dir,
            opt.sample_dir, opt.test_sample_dir, opt.log_dir,
                     opt.test_sample_dir0, opt.test_sample_dir1, opt.test_sample_dir2])

        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'

        # save to the disk
        if opt.isTrain:
            file_name = os.path.join(opt.checkpoints_dir, 'opt.txt')
        else:
            file_name = os.path.join(opt.checkpoints_dir, 'test_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    @abstractmethod
    def parse(self):
        pass


class TestOptions(BaseOptions):

    def initialize(self, parser):

        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--pos_number', type=int, default=4, help='position')
        parser.add_argument('--use_sp', action='store_true', help='use spetral normalization')
        parser.add_argument('--crop_w', type=int, default=160, help='the size of cropped eye region')
        parser.add_argument('--crop_h', type=int, default=92, help='the size of crooped eye region')
        parser.add_argument('--crop_w_p', type=int, default=200, help='the padding version for cropped size')
        parser.add_argument('--crop_h_p', type=int, default=128, help='the padding version for crooped size')
        parser.add_argument('--test_num', type=int, default=300, help='the number of test samples')
        
        self.isTrain = False
        return parser

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain
        self.print_options(opt)
        self.opt = opt

        return self.opt


class TrainOptions(BaseOptions):

    def initialize(self, parser):

        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        parser.add_argument('--save_model_freq', type=int, default=10000, help='frequency of saving checkpoints')
        parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--niter', type=int, default=34000, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=50000, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--lr_d', type=float, default=0.0001, help='initial learning rate for Adam in d')
        parser.add_argument('--lr_g', type=float, default=0.0001, help='initial learning rate for Adam in g')
        parser.add_argument('--lr_r', type=float, default=0.0005, help='initial learning rate for Adam in r')
        parser.add_argument('--lr_sr', type=float, default=4e-4, help='initial learning rate for Adam in sr')
        parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam')
        parser.add_argument('--loss_type', type=str, default='softplus', choices=['hinge', 'gan', 'wgan', 'softplus', 'lsgan'], help='using type of gan loss')
        parser.add_argument('--lam_gp', type=float, default=10.0, help='weight for gradient penalty of gan')
        parser.add_argument('--lam_p', type=float, default=100.0, help='perception loss in g')
        parser.add_argument('--lam_r', type=float, default=1.0, help='weight for recon loss in g')
        parser.add_argument('--lam_ss', type=float, default=1, help='self-supervised loss in g')
        parser.add_argument('--lam_fp', type=float, default=0.1, help='fp loss for g')
        parser.add_argument('--use_sp', action='store_true', help='use spectral normalization')
        parser.add_argument('--pos_number', type=int, default=4, help='position')
        parser.add_argument('--crop_w', type=int, default=50, help='the size of cropped eye region')
        parser.add_argument('--crop_h', type=int, default=30, help='the size of crooped eye region')
        parser.add_argument('--crop_w_p', type=int, default=180, help='the padding version for cropped size')
        parser.add_argument('--crop_h_p', type=int, default=128, help='the padding version for crooped size')
        parser.add_argument('--test_num', type=int, default=300, help='the number of test samples')

        self.isTrain = True
        return parser

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain
        self.print_options(opt)
        self.opt = opt

        return self.opt


def save_as_gif(images_list, out_path, gif_file_name='all', save_image=False):

    if os.path.exists(out_path) == False:
        os.mkdir(out_path)
    # save as .png
    if save_image == True:
        for n in range(len(images_list)):
            file_name = '{}.png'.format(n)
            save_path_and_name = os.path.join(out_path, file_name)
            misc.imsave(save_path_and_name, images_list[n])
    # save as .gif
    out_path_and_name = os.path.join(out_path, '{}.gif'.format(gif_file_name))
    imageio.mimsave(out_path_and_name, images_list, 'GIF', duration=0.1)

def get_image(image_path, crop_size=128, is_crop=False, resize_w=140, is_grayscale=False):
    return transform(imread(image_path , is_grayscale), crop_size, is_crop, resize_w)

def transform(image, crop_size=64, is_crop=True, resize_w=140):

    image = scipy.misc.imresize(image, [resize_w, resize_w])
    if is_crop:
        cropped_image = center_crop(image, crop_size)
    else:
        cropped_image = image
        cropped_image = scipy.misc.imresize(cropped_image ,
                            [resize_w , resize_w])

    return np.array(cropped_image) / 127.5 - 1

def center_crop(x, crop_h, crop_w=None):

    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))

    rate = np.random.uniform(0, 1, size=1)
    if rate < 0.5:
        x = np.fliplr(x)

    return x[j:j+crop_h, i:i+crop_w]

def transform_image(image):
    return (image + 1) * 127.5

def save_images(images, image_path, is_verse=True):
    if is_verse:
        return imageio.imwrite(image_path, inverse_transform(images))
    else:
        return imageio.imwrite(image_path, images)

def resizeImg(img, size=list):
    return scipy.misc.imresize(img, size)

def imread(path, is_grayscale=False):

    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def merge(images, size):
    if len(images.shape) == 3:
        img = images
    else:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w: i * w + w, :] = image
    return img

def inverse_transform(image):
    result = ((image + 1) * 127.5).astype(np.uint8)
    return result

height_to_eyeball_radius_ratio = 1.1
eyeball_radius_to_iris_diameter_ratio = 1.0

def replace_eyes(image, local_left_eyes, local_right_eyes, start_left_point, start_right_point):

    img_size = image.shape[-2]
    copy_image = np.copy(image)
    for i in range(len(image)):
        #for left
        y_cen, x_cen = int(start_left_point[i][0]*img_size), np.abs(int(start_left_point[i][1]*img_size))
        local_height, local_width = int(local_left_eyes[i].shape[0]), int(local_left_eyes[i].shape[1])
        copy_image[i, y_cen:(y_cen + local_height), x_cen:(x_cen + local_width), :] = local_left_eyes[i]
        #for right
        y_cen, x_cen = int(start_right_point[i][0]*img_size), int(start_right_point[i][1]*img_size)
        local_height, local_width = int(local_right_eyes[i].shape[0]), int(local_right_eyes[i].shape[1])
        #print "local_width", local_width, local_height, x_cen, y_cen, i
        if x_cen + local_width > img_size:
            y_right = img_size
        else:
            y_right = x_cen + local_width
            # local_right_eyes[i] = Image.res(local_right_eyes[i], newshape=(local_height, new_width, 3))
        # resize_replace = np.transpose(resize_replace, axes=(1, 0, 2))
        copy_image[i, y_cen:(y_cen + local_height), x_cen:y_right, :] = local_right_eyes[i, :, 0:y_right-x_cen, :]

    return copy_image

def imageClose(image, left_eye, right_eye, left_eye_mask, right_eye_mask):

    batch_size = image.shape[0]
    ret = []
    for i in range(batch_size):

        _image = inverse_transform(image[..., [2,1,0]][i])
        cv2.imwrite("_image.jpg", _image)
        _left_eye = inverse_transform(left_eye[...,[2,1,0]][i])
        cv2.imwrite("_left_eye.jpg", _left_eye)
        _right_eye = inverse_transform(right_eye[...,[2,1,0]][i])
        cv2.imwrite("_right_eye.jpg", _right_eye)
        _left_eye_mask = (left_eye_mask[...,[2,1,0]][i] * 255).astype(np.uint8)
        cv2.imwrite("_left_eye_mask.jpg", _left_eye_mask)
        _right_eye_mask = (right_eye_mask[...,[2,1,0]][i] * 255).astype(np.uint8)
        cv2.imwrite("_right_eye_mask.jpg", _right_eye_mask)

        #for left eyes
        itemindex = np.where(_left_eye_mask == 255)
        center = (itemindex[1][0] // 2 + itemindex[1][-1] // 2, itemindex[0][0] // 2 + itemindex[0][-1] // 2)
        print(center)
        dstimg = cv2.inpaint(_image, _left_eye_mask[...,0], 1, cv2.INPAINT_TELEA)
        #cv2.imwrite("dstimg.jpg", dstimg)
        out_left = cv2.seamlessClone(_left_eye, dstimg, _left_eye_mask, center, cv2.NORMAL_CLONE)

        #for right eyes
        itemindex = np.where(_right_eye_mask == 255)
        center = (itemindex[1][0] // 2 + itemindex[1][-1] // 2, itemindex[0][0] // 2 + itemindex[0][-1] // 2)
        print(center)
        dstimg = cv2.inpaint(out_left, _right_eye_mask[...,0], 1, cv2.INPAINT_TELEA)
        out_right = cv2.seamlessClone(_right_eye, dstimg, _right_eye_mask, center, cv2.NORMAL_CLONE)
        out_right = out_right[..., [2, 1, 0]] / 127.5 - 1
        ret.append(out_right)

    return np.array(ret)

def from_gaze2d(gaze, output_size, scale=1.0):

    """Generate a normalized pictorial representation of 3D gaze direction."""
    gazemaps = []
    oh, ow = np.round(scale * np.asarray(output_size)).astype(np.int32)
    oh_2 = int(np.round(0.5 * oh))
    ow_2 = int(np.round(0.5 * ow))
    r = int(height_to_eyeball_radius_ratio * oh_2)
    theta, phi = gaze
    theta = -theta
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # Draw iris
    eyeball_radius = int(height_to_eyeball_radius_ratio * oh_2)
    iris_radius_angle = np.arcsin(0.5 * eyeball_radius_to_iris_diameter_ratio)
    iris_radius = eyeball_radius_to_iris_diameter_ratio * eyeball_radius
    iris_distance = float(eyeball_radius) * np.cos(iris_radius_angle)
    iris_offset = np.asarray([
        -iris_distance * sin_phi * cos_theta,
        iris_distance * sin_theta,
    ])
    iris_centre = np.asarray([ow_2, oh_2]) + iris_offset
    angle = np.degrees(np.arctan2(iris_offset[1], iris_offset[0]))
    ellipse_max = eyeball_radius_to_iris_diameter_ratio * iris_radius
    ellipse_min = np.abs(ellipse_max * cos_phi * cos_theta)
    #gazemap = np.zeros((oh, ow), dtype=np.float32)

    # Draw eyeball
    gazemap = np.zeros((oh, ow), dtype=np.float32)
    gazemap = cv2.ellipse(gazemap, box=(iris_centre, (ellipse_min, ellipse_max), angle),
                         color = 1.0 , thickness=-1, lineType=cv2.LINE_AA)
    #outout = cv2.circle(test_gazemap, (ow_2, oh_2), r, color=1, thickness=-1)
    gazemaps.append(gazemap)

    gazemap = np.zeros((oh, ow), dtype=np.float32)
    gazemap = cv2.circle(gazemap, (ow_2, oh_2), r, color=1, thickness=-1)
    gazemaps.append(gazemap)

    return np.asarray(gazemaps)


def get_gan_losses_fn():
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    def d_loss_fn(r_logit, f_logit):
        r_loss = bce(tf.ones_like(r_logit), r_logit)
        f_loss = bce(tf.zeros_like(f_logit), f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = bce(tf.ones_like(f_logit), f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn

def get_hinge_loss():
    def loss_hinge_dis(d_real_logits, d_fake_logits):
        loss = tf.reduce_mean(tf.nn.relu(1.0 - d_real_logits))
        loss += tf.reduce_mean(tf.nn.relu(1.0 + d_fake_logits))
        return loss

    def loss_hinge_gen(d_fake_logits):
        loss = - tf.reduce_mean(d_fake_logits)
        return loss

    return loss_hinge_dis, loss_hinge_gen

def get_softplus_loss():

    def loss_dis(d_real_logits, d_fake_logits):
        l1 = tf.reduce_mean(tf.nn.softplus(-d_real_logits))
        l2 = tf.reduce_mean(tf.nn.softplus(d_fake_logits))
        return l1 + l2

    def loss_gen(d_fake_logits):
        return tf.reduce_mean(tf.nn.softplus(-d_fake_logits))

    return loss_dis, loss_gen

def get_lsgan_loss():

    def d_lsgan_loss(d_real_logits, d_fake_logits):
        return tf.reduce_mean((d_real_logits - 0.9)*2) \
               + tf.reduce_mean((d_fake_logits)*2)

    def g_lsgan_loss(d_fake_logits):
        return tf.reduce_mean((d_fake_logits - 0.9)*2)

    return d_lsgan_loss, g_lsgan_loss

def get_wgan_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = - tf.reduce_mean(r_logit)
        f_loss = tf.reduce_mean(f_logit)
        return r_loss + f_loss

    def g_loss_fn(f_logit):
        f_loss = - tf.reduce_mean(f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn

def get_adversarial_loss(mode):

    if mode == 'gan':
        return get_gan_losses_fn()
    elif mode == 'hinge':
        return get_hinge_loss()
    elif mode == 'lsgan':
        return get_lsgan_loss()
    elif mode == 'softplus':
        return get_softplus_loss()
    elif mode == 'wgan_gp':
        return get_wgan_losses_fn()



def bilinear_sample(input, flow, name):
    # reference to spatial transform network
    # 1.details can be found in office release:
    #   https://github.com/tensorflow/models/blob/master/research/transformer/spatial_transformer.py
    # 2.maybe another good implement can be found in:
    #   https://github.com/kevinzakka/spatial-transformer-network/blob/master/transformer.py
    #   but this one maybe contain some problems, go to --> https://github.com/kevinzakka/spatial-transformer-network/issues/10
    with tf.variable_scope(name):
        N, iH, iW, iC = input.get_shape().as_list()
        _, fH, fW, fC = flow.get_shape().as_list()

        assert iH == fH and iW == fW
        # re-order & reshape: N,H,W,C --> N,C,H*W , shape= ( 16,2,3500 )
        flow = tf.reshape(tf.transpose(flow, [0, 3, 1, 2]), [-1, fC, fH * fW])
        # get mesh-grid, 2,H*W
        indices_grid = meshgrid(iH, iW)
        transformed_grid = tf.add(flow, indices_grid)
        x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])  # x_s should be (16,1,3500)
        y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])  # y_s should be ( 16,1,3500)
        # look tf.slice with ctrl , to figure out its meanning
        x_s_flatten = tf.reshape(x_s, [-1])  # should be (16*3500)
        y_s_flatten = tf.reshape(y_s, [-1])  # should be (16*3500)
        transformed_image = interpolate(input, x_s_flatten, y_s_flatten, iH, iW, 'interpolate')
        # print(transformed_image.get_shape().as_list())
        transformed_image = tf.reshape(transformed_image, [N, iH, iW, iC])

        return transformed_image

def meshgrid(height, width, ones_flag=None):

    with tf.variable_scope('meshgrid'):
        y_linspace = tf.linspace(-1., 1., height)
        x_linspace = tf.linspace(-1., 1., width)
        x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
        x_coordinates = tf.reshape(x_coordinates, shape=[-1])   #[H*W]
        y_coordinates = tf.reshape(y_coordinates, shape=[-1])   #[H*W]
        if ones_flag is None:
            indices_grid = tf.stack([x_coordinates, y_coordinates], axis=0) #[2, H*W]
        else:
            indices_grid = tf.stack([x_coordinates, y_coordinates, tf.ones_like(x_coordinates)], axis=0)

        return indices_grid


def interpolate(input, x, y, out_height, out_width, name):
    # parameters: input is input image,which has shape of (batchsize,height,width,3)
    # x,y is flattened coordinates , which has shape of (16*3500) = 56000
    # out_heigth,out_width = height,width
    with tf.variable_scope(name):
        N, H, W, C = input.get_shape().as_list()  #64, 40, 72, 3

        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        H_f = tf.cast(H, dtype=tf.float32)
        W_f = tf.cast(W, dtype=tf.float32)
        # note that x,y belongs to [-1,1] before
        x = (x + 1.0) * (W_f - 1) * 0.5 # x now is [0,2]*0.5*[width-1],is [0,1]*[width-1]
                                        # shape 16 * 3500
        y = (y + 1.0) * (H_f - 1) * 0.5
        # get x0 and x1 in bilinear interpolation
        x0 = tf.cast(tf.floor(x), tf.int32) # cast to int ,discrete
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1

        # clip the coordinate value
        max_y = tf.cast(H - 1, dtype=tf.int32)
        max_x = tf.cast(W - 1, dtype=tf.int32)
        zero = tf.constant([0], shape=(1,), dtype=tf.int32)

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        # note x0,x1,y0,y1 have same shape 16 * 3500
        # go to method , look tf.clip_by_value,
        # realizing restrict op
        flat_image_dimensions = H * W
        pixels_batch = tf.range(N) * flat_image_dimensions
        # note N is batchsize, pixels_batch has shape [16]
        # plus, it's value is [0,1,2,...15]* 3500
        flat_output_dimensions = out_height * out_width
        # a scalar
        base = repeat(pixels_batch, flat_output_dimensions)
        # return 16 * 3500, go to see concrete value.

        base_y0 = base + y0 * W
        # [0*3500,.....1*3500,....2*3500,....]+[]
        base_y1 = base + y1 * W
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        # gather every pixel value
        flat_image = tf.reshape(input, shape=(-1, C))
        flat_image = tf.cast(flat_image, dtype=tf.float32)

        pixel_values_a = tf.gather(flat_image, indices_a)
        pixel_values_b = tf.gather(flat_image, indices_b)
        pixel_values_c = tf.gather(flat_image, indices_c)
        pixel_values_d = tf.gather(flat_image, indices_d)

        x0 = tf.cast(x0, tf.float32)
        x1 = tf.cast(x1, tf.float32)
        y0 = tf.cast(y0, tf.float32)
        y1 = tf.cast(y1, tf.float32)

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)

        output = tf.add_n([area_a * pixel_values_a,
                           area_b * pixel_values_b,
                           area_c * pixel_values_c,
                           area_d * pixel_values_d])
        #for mask the interpolate part which pixel don't move
        mask = area_a + area_b + area_c + area_d
        output = (1 - mask) * flat_image + mask * output

        return output

def repeat(x, n_repeats):
    # parameters x: list [16]
    #            n_repeats : scalar,3500
    with tf.variable_scope('_repeat'):
        rep = tf.reshape(tf.ones(shape=tf.stack([n_repeats, ]), dtype=tf.int32), (1, n_repeats))
        # just know rep has shape (1,3500), and it's value is 1
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        # after reshape , matmul is (16,1)X(1,3500)
        # in matrix multi, result has shape ( 16,3500)
        # plus, in each row i, has same value  i * 3500
        return tf.reshape(x, [-1])  # return 16* 3500


def gradient_penalty(f, real, fake, mode='wgan-gp'):
    def _gradient_penalty(f, real, fake=None):
        def _interpolate(a, b=None):
            if b is None:   # interpolation in DRAGAN
                beta = tf.random_uniform(shape=tf.shape(a), minval=0., maxval=1.)
                b = a + 0.5 * tf.math.reduce_std(a) * beta
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.shape)
            return inter

        x = _interpolate(real, fake)
        pred = f(x)
        grad = tf.gradients(pred, x)[0]
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)

        return gp

    if mode == 'none':
        gp = tf.constant(0, dtype=real.dtype)
    elif mode == 'dragan':
        gp = _gradient_penalty(f, real)
    elif mode == 'wgan-gp':
        gp = _gradient_penalty(f, real, fake)
    else:
        raise("your mode is not correct")

    return gp


def getfeature_matching_loss(feature1, feature2):
    return tf.reduce_mean(tf.abs(
        tf.reduce_mean(feature1, axis=[1, 2]) - tf.reduce_mean(feature2, axis=[1, 2])))

def SSCE(logits, labels) :
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return loss

def SCE(logits, labels) :
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                          (labels=labels, logits=logits))
    return loss

def cosine(f1, f2):
    f1_norm = tf.nn.l2_normalize(f1, dim=0)
    f2_norm = tf.nn.l2_normalize(f2, dim=0)
    return tf.losses.cosine_distance(f1_norm, f2_norm, dim=0)

def MSE(i1, i2):
    return tf.reduce_mean(tf.square(i1 - i2))

def L1(i1, i2):
    return tf.reduce_mean(tf.abs(i1 - i2))

def TV_loss(i1):
    shape = i1.get_shape().as_list()
    return tf.reduce_mean(tf.image.total_variation(i1)) / (shape[1]*shape[2]*shape[3])


def log_sum_exp(x, axis=1):
    m = tf.reduce_max(x, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x - m), axis=axis))

def lrelu(x, alpha=0.2, name="LeakyReLU"):
    with tf.variable_scope(name):
        return tf.maximum(x , alpha*x)

# Get residual for laplacian pyramid
def pyrDown(input, scale):
    d_input = avgpool2d(input, k=scale)
    ds_input = upscale(d_input, scale=scale)
    residual = input - ds_input
    return d_input, ds_input, residual

def conv2d(input_, output_dim, kernel=4, stride=2, use_sp=False, padding='SAME', scope="conv2d", use_bias=True):
    with tf.variable_scope(scope):
        w = tf.get_variable('w', [kernel, kernel, input_.get_shape()[-1], output_dim],
                            initializer=tf.keras.initializers.VarianceScaling(), regularizer=l2_regularizer(l2=0.0001))
        if use_sp != True:
            conv = tf.nn.conv2d(input_, w, strides=[1, stride, stride, 1], padding=padding)
        else:
            conv = tf.nn.conv2d(input_, spectral_norm(w), strides=[1, stride, stride, 1], padding=padding)

        if use_bias:
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return conv

def instance_norm(input, scope="instance_norm", affine=True):
    with tf.variable_scope(scope):
        depth = input.get_shape()[-1]

        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        if affine:
            scale = tf.get_variable("scale", [depth],
                                    initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
            offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
            return scale * normalized + offset
        else:
            return normalized

def Resblock(x_init, o_dim=256, relu_type="lrelu", use_IN=True, ds=True, scope='resblock'):

    dim = x_init.get_shape().as_list()[-1]
    conv1 = functools.partial(conv2d, output_dim=dim, kernel=3, stride=1)
    conv2 = functools.partial(conv2d, output_dim=o_dim, kernel=3, stride=1)
    In = functools.partial(instance_norm)

    input_ch = x_init.get_shape().as_list()[-1]
    with tf.variable_scope(scope):

        def relu(relu_type):
            relu_dict = {
                "relu": tf.nn.relu,
                "lrelu": lrelu
            }
            return relu_dict[relu_type]

        def shortcut(x):
            if input_ch != o_dim:
                x = conv2d(x, output_dim=o_dim, kernel=1, stride=1, scope='conv', use_bias=False)
            if ds:
                x = avgpool2d(x, k=2)
            return x

        if use_IN:
            x = conv1(relu(relu_type)(In(x_init, scope='bn1')), padding='SAME', scope='c1')
            if ds:
                x = avgpool2d(x, k=2)
            x = conv2(relu(relu_type)(In(x, scope='bn2')), padding='SAME', scope='c2')
        else:
            x = conv1(relu(relu_type)(x_init), padding='SAME', scope='c1')
            if ds:
                x = avgpool2d(x, k=2)
            x = conv2(relu(relu_type)(x), padding='SAME', scope='c2')

        if input_ch != o_dim or ds:
            x_init = shortcut(x_init)

        return (x + x_init) / tf.sqrt(2.0)  #unit variance

def de_conv(input_, output_shape,
             k_h=4, k_w=4, d_h=2, d_w=2, use_sp=False,
             scope="deconv2d", with_w=False):

    with tf.variable_scope(scope):

        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], dtype=tf.float32,
                            initializer=tf.keras.initializers.VarianceScaling())
        if use_sp:
            deconv = tf.nn.conv2d_transpose(input_, spectral_norm(w), output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])
        else:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], tf.float32, initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def avgpool2d(x, k=2):
    return tf.nn.avg_pool(x, ksize=[1, k, k ,1], strides=[1, k, k, 1], padding='SAME')

def Adaptive_pool2d(x, output_size=1):
    input_size = get_conv_shape(x)[-1]
    stride = int(input_size / (output_size))
    kernel_size = input_size - (output_size - 1) * stride
    return tf.nn.avg_pool(x, ksize=[1, kernel_size, kernel_size, 1], strides=[1, kernel_size, kernel_size, 1], padding='SAME')

def upscale(x, scale, is_up=True):
    _, h, w, _ = get_conv_shape(x)
    if is_up:
        return resize_nearest_neighbor(x, (h * scale, w * scale))
    else:
        return resize_nearest_neighbor(x, (h // scale, w // scale))

def get_conv_shape(tensor):
    shape = int_shape(tensor)
    return shape

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def resize_nearest_neighbor(x, new_size):
    x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def fully_connect(input_, output_size, scope=None, use_sp=False,
                  bias_start=0.0, with_w=False):

  shape = input_.get_shape().as_list()
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 initializer=tf.keras.initializers.VarianceScaling(), regularizer=l2_regularizer(0.0001))
    bias = tf.get_variable("bias", [output_size], tf.float32,
      initializer=tf.constant_initializer(bias_start))

    if use_sp:
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
    y_reshaped = tf.reshape(y, [y_shapes[0], 1, 1, y_shapes[-1]])
    return tf.concat([x , y_reshaped*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2] , y_shapes[-1]])], 3)

def batch_normal(input, is_training=True, scope="scope", reuse=False):
    return batch_norm(input, epsilon=1e-5, decay=0.9, scale=True, is_training=is_training,
                      scope=scope, reuse=reuse, fused=True, updates_collections=None)

def identity(x, scope):
    return tf.identity(x, name=scope)

def _l2normalize(v, eps=1e-12):
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def spectral_norm(W, collections=None, return_norm=False, name='sn'):
    shape = W.get_shape().as_list()
    if len(shape) == 1:
        sigma = tf.reduce_max(tf.abs(W))
    else:
        if len(shape) == 4:
            _W = tf.reshape(W, (-1, shape[3]))
            shape = (shape[0] * shape[1] * shape[2], shape[3])
        else:
            _W = W
        u = tf.get_variable(
            name=name + "_u",
            shape=(_W.shape.as_list()[-1], shape[0]),
            initializer=tf.random_normal_initializer,
            collections=collections,
            trainable=False
        )

        _u = u
        for _ in range(1):
            _v = tf.nn.l2_normalize(tf.matmul(_u, _W), 1)
            _u = tf.nn.l2_normalize(tf.matmul(_v, tf.transpose(_W)), 1)
        _u = tf.stop_gradient(_u)
        _v = tf.stop_gradient(_v)
        sigma = tf.reduce_mean(tf.reduce_sum(_u * tf.transpose(tf.matmul(_W, tf.transpose(_v))), 1))
        update_u_op = tf.assign(u, _u)
        with tf.control_dependencies([update_u_op]):
            sigma = tf.identity(sigma)

    if return_norm:
        return W / sigma, sigma
    else:
        return W / sigma

def getWeight_Decay(scope='discriminator'):
    return tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope))

def getTrainVariable(vars, scope='discriminator'):
    return [var for var in vars if scope in var.name]

def GetOptimizer(lr, loss, vars, beta1, beta2):
    return tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2). \
        minimize(loss=loss, var_list=vars)


class Vgg(object):

    def __init__(self):
        self.content_layer_name = ["vgg_16/conv5/conv5_3"]
        self.style_layers = ["vgg_16/conv1/conv1_2", "vgg_16/conv2/conv2_2",
                        "vgg_16/conv3/conv3_3", "vgg_16/conv4/conv4_3"]

    def content_loss(self, endpoints_mixed, content_layers):

        loss = 0
        for layer in content_layers:
            feat_a, feat_b = tf.split(endpoints_mixed[layer], 2, 0)
            size = tf.size(feat_a)
            loss += tf.nn.l2_loss(feat_a - feat_b) * 2 / tf.to_float(size)

        return loss

    def style_loss(self, endpoints_mixed, style_layers):

        loss = 0
        for layer in style_layers:
            feat_a, feat_b = tf.split(endpoints_mixed[layer], 2, 0)
            size = tf.size(feat_a)
            loss += tf.nn.l2_loss(
                self.gram(feat_a) - self.gram(feat_b)) * 2 / tf.to_float(size)

        return loss

    def gram(self, layer):

        shape = tf.shape(layer)
        num_images = shape[0]
        width = shape[1]
        height = shape[2]
        num_filters = shape[3]
        features = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
        denominator = tf.to_float(width * height * num_filters)
        grams = tf.matmul(features, features, transpose_a=True) / denominator

        return grams

    def vgg_content_loss(self, fake, real):
        """
        build the sub graph of perceptual matching network
        return:
            c_loss: content loss
            s_loss: style loss
        """
        _, endpoints_mixed = self.vgg_16(tf.concat([fake, real], 0))
        c_loss = self.content_loss(endpoints_mixed, self.content_layer_name)

        return c_loss

    def vgg_style_loss(self, fake, real):
        """
        build the sub graph of perceptual matching network
        return:
            c_loss: content loss
        """
        _, endpoints_mixed = self.vgg_16(tf.concat([fake, real], 0))
        s_loss = self.style_loss(endpoints_mixed, self.style_layers)

        return s_loss

    def percep_loss(self, fake, real):
        return self.vgg_style_loss(fake, real) + self.vgg_content_loss(fake, real)

    def vgg_16(self, inputs, scope='vgg_16'):

        # repeat_net = functools.partial(slim.repeat, )
        with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=tf.AUTO_REUSE) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope(
                    [slim.conv2d, slim.fully_connected, slim.max_pool2d],
                    outputs_collections=end_points_collection):
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                # Convert end_points_collection into a end_point dict
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

        return net, end_points




class Dataset(object):

    def __init__(self, config):
        super(Dataset, self).__init__()

        self.data_dir = config.data_dir
        self.attr_0_txt = '0.txt'
        self.attr_1_txt = '1.txt'
        self.height, self.width= config.img_size, config.img_size
        self.channel = config.output_nc
        self.capacity = config.capacity
        self.batch_size = config.batch_size
        self.num_threads = config.num_threads
        self.test_number = config.test_num

        self.train_images_list, self.train_eye_pos, \
        self.train_images_list2, self.train_eye_pos2, \
                self.test_images_list, self.test_eye_pos, \
                    self.test_images_list0, self.test_eye_pos0 = self.readfilenames()

        print(len(self.train_images_list), len(self.train_eye_pos),
              len(self.train_images_list2), len(self.train_eye_pos2),
              len(self.train_images_list2), len(self.train_eye_pos2),
              len(self.test_images_list), len(self.test_eye_pos),
                len(self.test_images_list0), len(self.test_eye_pos0))

        assert len(self.train_images_list) == len(self.train_eye_pos)
        assert len(self.train_images_list2) == len(self.train_eye_pos2)
        assert len(self.test_images_list) == len(self.test_eye_pos)
        assert len(self.test_images_list0) == len(self.test_eye_pos0)

        assert len(self.train_images_list) > 0

    def readfilenames(self):

        train_eye_pos = []
        train_images_list = []
        fh = open(os.path.join(self.data_dir, self.attr_1_txt))

        for f in fh.readlines():
            f = f.strip('\n')
            filenames = f.split(' ', 5)
            if os.path.exists(os.path.join(self.data_dir, "1/"+filenames[0]+".jpg")):
                train_images_list.append(os.path.join(self.data_dir, "1/"+filenames[0]+".jpg"))
                train_eye_pos.append([int(value) for value in filenames[1:5]])
        fh.close()
        logger.debug("Read training data of length: {}".format(len(train_images_list)))
        fh = open(os.path.join(self.data_dir, self.attr_0_txt))
        test_images_list = []
        test_eye_pos = []

        for f in fh.readlines():
            f = f.strip('\n')
            filenames = f.split(' ', 5)
            if os.path.exists(os.path.join(self.data_dir, "0/"+filenames[0]+".jpg")):
                test_images_list.append(os.path.join(self.data_dir,"0/"+filenames[0]+".jpg"))
                test_eye_pos.append([int(value) for value in filenames[1:5]])
                #print test_eye_pos
        fh.close()
        logger.debug("Read testing images of length {}".format(len(test_images_list)))
        train_images_list2 = test_images_list[0:-self.test_number]
        train_eye_pos2 = test_eye_pos[0:-self.test_number]

        test_images_list = test_images_list[-self.test_number:]
        test_eye_pos = test_eye_pos[-self.test_number:]

        test_images_list0 = train_images_list[-100:]
        test_eye_pos0 = train_eye_pos[-100:]

        train_images_list = train_images_list[0:-100]
        train_eye_pos = train_eye_pos[0:-100]

        return train_images_list, train_eye_pos, \
               train_images_list2, train_eye_pos2, \
               test_images_list, test_eye_pos, \
               test_images_list0, test_eye_pos0

    def read_images(self, input_queue):

        content = tf.read_file(input_queue)
        # Decode a JPEG-encoded image to a uint8 tensor
        image = tf.image.decode_jpeg(content, channels=self.channel)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_images(image, size=(self.height, self.width))
        return image / 127.5 - 1.0

    def input(self):

        train_images = tf.convert_to_tensor(self.train_images_list, dtype=tf.string)
        train_eye_pos = tf.convert_to_tensor(self.train_eye_pos, dtype=tf.int32)
        train_queue = tf.train.slice_input_producer([train_images, train_eye_pos], shuffle=True)
        train_eye_pos_queue = train_queue[1]
        train_images_queue = self.read_images(input_queue=train_queue[0])

        train_images2 = tf.convert_to_tensor(self.train_images_list2, dtype=tf.string)
        train_eye_pos2 = tf.convert_to_tensor(self.train_eye_pos2, dtype=tf.int32)
        train_queue2 = tf.train.slice_input_producer([train_images2, train_eye_pos2], shuffle=True)
        train_eye_pos_queue2 = train_queue2[1]
        train_images_queue2 = self.read_images(input_queue=train_queue2[0])

        test_images = tf.convert_to_tensor(self.test_images_list, dtype=tf.string)
        test_eye_pos = tf.convert_to_tensor(self.test_eye_pos, dtype=tf.int32)
        test_queue = tf.train.slice_input_producer([test_images, test_eye_pos], shuffle=False)
        test_eye_pos_queue = test_queue[1]
        test_images_queue = self.read_images(input_queue=test_queue[0])

        batch_image1, batch_eye_pos1, \
                batch_image2, batch_eye_pos2= tf.train.shuffle_batch([train_images_queue, train_eye_pos_queue,
                                                            train_images_queue2, train_eye_pos_queue2],
                                                batch_size=self.batch_size,
                                                capacity=self.capacity,
                                                num_threads=self.num_threads,
                                                min_after_dequeue=500)

        batch_image3, batch_eye_pos3 = tf.train.batch([test_images_queue, test_eye_pos_queue],
                                                batch_size=self.batch_size,
                                                capacity=100,
                                                num_threads=1)

        return batch_image1, batch_eye_pos1, batch_image2, batch_eye_pos2, batch_image3, \
               batch_eye_pos3


class Gaze_GAN(object):

    # build model
    def __init__(self, dataset, opt):

        self.dataset = dataset
        self.opt = opt
        # placeholder
        self.x_left_p = tf.placeholder(tf.float32, [self.opt.batch_size, self.opt.pos_number])
        self.x_right_p = tf.placeholder(tf.float32, [self.opt.batch_size, self.opt.pos_number])
        self.y_left_p = tf.placeholder(tf.float32, [self.opt.batch_size, self.opt.pos_number])
        self.y_right_p = tf.placeholder(tf.float32, [self.opt.batch_size, self.opt.pos_number])

        self.x = tf.placeholder(tf.float32,
                                [self.opt.batch_size, self.opt.img_size, self.opt.img_size, self.opt.input_nc])
        self.xm = tf.placeholder(tf.float32,
                                [self.opt.batch_size, self.opt.img_size, self.opt.img_size, self.opt.output_nc])
        self.y = tf.placeholder(tf.float32,
                                [self.opt.batch_size, self.opt.img_size, self.opt.img_size, self.opt.input_nc])
        self.ym = tf.placeholder(tf.float32,
                                [self.opt.batch_size, self.opt.img_size, self.opt.img_size, self.opt.input_nc])

        self.alpha = tf.placeholder(tf.float32, [self.opt.batch_size, 1])
        self.lr_decay = tf.placeholder(tf.float32, None, name='lr_decay')

        self.vgg = Vgg()

    def build_model(self):

        def build_x_model():
            
            xc = self.x * (1 - self.xm)  # corrputed images
            xl_left, xl_right = self.crop_resize(self.x, self.x_left_p, self.x_right_p)
            xl_left_fp = self.Gr(xl_left, use_sp=False)
            xl_right_fp = self.Gr(xl_right, use_sp=False)
            xl_fp_content = tf.concat([xl_left_fp, xl_right_fp], axis=-1)
            xo = self.Gx(xc, self.xm, xl_fp_content, use_sp=False)

            return xc, xl_left, xl_right, xo

        def build_y_model():

            yc = self.y * (1 - self.ym)
            yl_left, yl_right = self.crop_resize(self.y, self.y_left_p, self.y_right_p)
            yl_fp = self.encode(tf.concat([yl_left, yl_right], axis=-1))
            yl_content_left = self.Gr(yl_left, use_sp=False)
            yl_content_right = self.Gr(yl_right, use_sp=False)
            yl_content = tf.concat([yl_content_left, yl_content_right], axis=-1)
            yo = self.Gy(yc, self.ym, yl_fp, yl_content, use_sp=False)

            yo_left, yo_right = self.crop_resize(yo, self.y_left_p, self.y_right_p)
            yo_fp = self.encode(tf.concat([yo_left, yo_right], axis=-1))

            yo_content_left = self.Gr(yo_left, use_sp=False)
            yo_content_right = self.Gr(yo_right, use_sp=False)

            yo_content = tf.concat([yo_content_left, yo_content_right], axis=-1)

            return yc, yl_left, yl_right, yl_fp, yl_content, yo, yo_fp, yo_content

        def build_yx_model():
            
            y2x = self.Gx(self.yc, self.ym, self.yl_content, use_sp=False)  # output
            y2x_left, y2x_right = self.crop_resize(y2x, self.y_left_p, self.y_right_p)

            y2x_content_left = self.Gr(y2x_left, use_sp=False)
            y2x_content_right = self.Gr(y2x_right, use_sp=False)
            
            #
            y2x_content = tf.concat([y2x_content_left, y2x_content_right], axis=-1)

            y2x_fp = self.encode(tf.concat([y2x_left, y2x_right], axis=-1))
            # Learn the angle related features
            y2x_ = self.Gy(self.yc, self.ym, y2x_fp, y2x_content, use_sp=False)

            y2x_left_, y2x_right_ = self.crop_resize(y2x_, self.y_left_p, self.y_right_p)

            y2x_content_left_ = self.Gr(y2x_left_, use_sp=False)
            y2x_content_right_ = self.Gr(y2x_right_, use_sp=False)

            y2x_content_ = tf.concat([y2x_content_left_, y2x_content_right_], axis=-1)

            y2x_fp_ = self.encode(tf.concat([y2x_left_, y2x_right_], axis=-1))

            return y2x, y2x_left, y2x_right, y2x_fp, y2x_content, y2x_, y2x_fp_, y2x_content_

        self.xc, self.xl_left, self.xl_right, self.xo = build_x_model()
        self.yc, self.yl_left, self.yl_right, self.yl_fp, \
                self.yl_content, self.yo, self.yo_fp, self.yo_content = build_y_model()

        self.y2x, self.y2x_left, self.y2x_right, self.y2x_fp, self.y2x_content, self.y2x_, \
                            self.y2x_fp_, self.y2x_content_ = build_yx_model()

        self._xl_left, self._xl_right = self.crop_resize(self.xo, self.x_left_p, self.x_right_p)
        self._yl_left, self._yl_right = self.crop_resize(self.yo, self.y_left_p, self.y_right_p)
        self._y2x_left_, self._y2x_right_ = self.crop_resize(self.y2x_, self.y_left_p, self.y_right_p)

        self.dx_logits = self.D(self.x, self.xl_left, self.xl_right, scope='Dx')
        self.gx_logits = self.D(self.xo, self._xl_left, self._xl_right, scope='Dx')

        self.dy_logits = self.D(self.y, self.yl_left, self.yl_right, scope='Dy')
        self.gy_logits = self.D(self.yo, self._yl_left, self._yl_right, scope='Dy')

        # self.dyx_logits = self.D(self.x, self.y2x_left, self.y2x_right)
        self.gyx_logits = self.D(self.y2x_, self._y2x_left_, self._y2x_right_, scope='Dx')
        d_loss_fun, g_loss_fun = get_adversarial_loss(self.opt.loss_type)

        self.dx_gan_loss = d_loss_fun(self.dx_logits, self.gx_logits)
        self.gx_gan_loss = g_loss_fun(self.gx_logits)

        self.dy_gan_loss = d_loss_fun(self.dy_logits, self.gy_logits)
        self.gy_gan_loss = g_loss_fun(self.gy_logits)

        self.dyx_gan_loss = d_loss_fun(self.dx_logits, self.gyx_logits)
        self.gyx_gan_loss = g_loss_fun(self.gyx_logits)

        self.recon_loss_x = self.Local_L1(self.xo, self.x)
        self.recon_loss_y = self.Local_L1(self.yo, self.y)
        self.recon_loss_y_angle = self.Local_L1(self.y2x, self.y2x_)

        self.percep_loss_x = self.vgg.percep_loss(self.xl_left, self._xl_left) \
                             + self.vgg.percep_loss(self.xl_right, self._xl_right)

        self.percep_loss_y = self.vgg.percep_loss(self.yl_left, self._yl_left) + \
                             self.vgg.percep_loss(self.yl_right, self._yl_right) + \
                             self.vgg.percep_loss(self.y2x_left, self._y2x_left_) + \
                             self.vgg.percep_loss(self.y2x_right, self._y2x_right_)

        # fp loss
        self.recon_fp_content = L1(self.y2x_content, self.y2x_content_) + L1(self.yl_content, self.yo_content)

        # self.recon_fp_angle = L1(self.y2x_fp, self.y2x_fp_) + L1(self.yl_fp, self.yo_fp)
        self.Dx_loss = self.dx_gan_loss + self.dyx_gan_loss
        self.Dy_loss = self.dy_gan_loss
        self.Gx_loss = self.gx_gan_loss + self.opt.lam_r * self.recon_loss_x + self.opt.lam_p * self.percep_loss_x
        self.Gy_loss = self.gy_gan_loss + self.gyx_gan_loss + self.opt.lam_r * self.recon_loss_y \
                       + self.opt.lam_r * self.recon_loss_y_angle + self.recon_fp_content + self.opt.lam_p * self.percep_loss_y

    def build_test_model(self):

        def build_x_model():
            xc = self.x * (1 - self.xm)  # corrputed images
            xl_left, xl_right = self.crop_resize(self.x, self.x_left_p, self.x_right_p)
            xl_left_fp = self.Gr(xl_left, use_sp=False)
            xl_right_fp = self.Gr(xl_right, use_sp=False)
            xl_fp_content = tf.concat([xl_left_fp, xl_right_fp], axis=-1)
            xl_fp = self.encode(tf.concat([xl_left, xl_right], axis=-1))
            xo = self.Gx(xc, self.xm, xl_fp_content, use_sp=False)

            return xc, xl_left, xl_right, xl_fp, xl_fp_content, xo

        def build_y_model():

            yc = self.y * (1 - self.ym)
            yl_left, yl_right = self.crop_resize(self.y, self.y_left_p, self.y_right_p)
            yl_fp = self.encode(tf.concat([yl_left, yl_right], axis=-1))
            yl_content_left = self.Gr(yl_left, use_sp=False)
            yl_content_right = self.Gr(yl_right, use_sp=False)
            yl_content = tf.concat([yl_content_left, yl_content_right], axis=-1)
            yo = self.Gy(yc, self.ym, yl_fp, yl_content, use_sp=False)

            return yc, yl_left, yl_right, yl_fp, yl_content, yo

        def build_yx_model():

            y2x = self.Gx(self.yc, self.ym, self.yl_content, use_sp=False)  # output
            y2x_left, y2x_right = self.crop_resize(y2x, self.y_left_p, self.y_right_p)

            y2x_content_left = self.Gr(y2x_left, use_sp=False)
            y2x_content_right = self.Gr(y2x_right, use_sp=False)

            y2x_content = tf.concat([y2x_content_left, y2x_content_right], axis=-1)
            y2x_fp = self.encode(tf.concat([y2x_left, y2x_right], axis=-1))

            # Learn the angle related features
            y2x_ = self.Gy(self.yc, self.ym, y2x_fp, y2x_content, use_sp=False)

            return y2x, y2x_left, y2x_right, y2x_fp, y2x_content, y2x_

        self.xc, self.xl_left, self.xl_right, self.xl_fp, self.xl_content, self.xo = build_x_model()
        self.yc, self.yl_left, self.yl_right, self.yl_fp, self.yl_content, self.yo = build_y_model()

        self._xl_left, self._xl_right = self.crop_resize(self.xo, self.x_left_p, self.x_right_p)
        self._yl_left, self._yl_right = self.crop_resize(self.yo, self.y_left_p, self.y_right_p)

        yo_content_left = self.Gr(self._yl_left, use_sp=False)
        yo_content_right = self.Gr(self._yl_right, use_sp=False)

        self.yo_content = tf.concat([yo_content_left, yo_content_right], axis=-1)
        self.y2x, self.y2x_left, self.y2x_right, self.y2x_fp, self.y2x_content, self.y2x_ = build_yx_model()

        self._y2x_left, self._y2x_right = self.crop_resize(self.y2x_, self.x_left_p, self.x_right_p)
        y2x_content_left_ = self.Gr(self._y2x_left, use_sp=False)
        y2x_content_right_ = self.Gr(self._y2x_right, use_sp=False)
        self.y2x_content_ = tf.concat([y2x_content_left_, y2x_content_right_], axis=-1)

        self.y2x_fp_inter = self.y2x_fp * self.alpha + (1 - self.alpha) * self.yl_fp
        self.y2x_content_inter = self.y2x_content * self.alpha + (1 - self.alpha) * self.yl_content
        self._y2x_inter = self.Gy(self.yc, self.ym, self.y2x_fp_inter, self.y2x_content_inter, use_sp=False)

    def crop_resize(self, input, boxes_left, boxes_right):

        shape = [int(item) for item in input.shape.as_list()]
        return tf.image.crop_and_resize(input, boxes=boxes_left, box_ind=list(range(0, shape[0])),
                                        crop_size=[int(shape[-3] / 4), int(shape[-2] / 4)]), \
               tf.image.crop_and_resize(input, boxes=boxes_right, box_ind=list(range(0, shape[0])),
                                        crop_size=[int(shape[-3] / 4), int(shape[-2] / 4)])

    def Local_L1(self, l1, l2):
        loss = tf.reduce_mean(tf.reduce_sum(tf.abs(l1 - l2), axis=[1, 2, 3])
                              / (self.opt.crop_w * self.opt.crop_h * self.opt.output_nc))
        return loss

    def test(self):
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            self.saver = tf.train.Saver()

            ckpt = tf.train.get_checkpoint_state(self.opt.checkpoints_dir)
            print('Load checkpoint')
            logger.debug('Load checkpoint!')
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Load Succeed!')
                logger.debug('Load Succeed!')
            else:
                print('Do not exists any checkpoint, Load Failed!')
                logger.debug('Do not exists any checkpoint, Load Failed!')
                exit()

            trainbatch, trainmask, _, _, testbatch, testmask = self.dataset.input()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            batch_num = self.opt.test_num
            for j in range(batch_num):
                x_img, x_img_pos, y_img, y_img_pos = sess.run([trainbatch, trainmask, testbatch, testmask])
                x_m, x_left_pos, x_right_pos = self.get_Mask_and_pos(x_img_pos)
                y_m, y_left_pos, y_right_pos = self.get_Mask_and_pos(y_img_pos)

                f_d = {self.x: x_img,
                       self.xm: x_m,
                       self.x_left_p: x_left_pos,
                       self.x_right_p: x_right_pos,
                       self.y: y_img,
                       self.ym: y_m,
                       self.y_left_p: y_left_pos,
                       self.y_right_p: y_right_pos
                       }

                output = sess.run([self.x, self.xc, self.xo, self.yc,
                                   self.y, self.yo, self.y2x, self.y2x_], feed_dict=f_d)
                output_concat = self.Transpose(np.array([output[0], output[1], output[2],
                                                         output[3], output[4], output[5], output[6], output[7]]))
                local_output = sess.run([self.xl_left, self.xl_right, self.yl_left, self.yl_right,
                                         self._xl_left, self._xl_right, self._yl_left, self._yl_right, self.y2x_left,
                                         self.y2x_right], feed_dict=f_d)
                local_output_concat = self.Transpose(
                    np.array([local_output[0], local_output[1], local_output[2], local_output[3],
                              local_output[4], local_output[5], local_output[6], local_output[7],
                              local_output[8], local_output[9]]))

                inter_results = [y_img, np.ones(shape=[self.opt.batch_size,
                                                       self.opt.img_size, self.opt.img_size, 3])]
                inter_results1 = [y_img, np.ones(shape=[self.opt.batch_size,
                                                        self.opt.img_size, self.opt.img_size, 3])]
                inter_results2 = [y_img, np.ones(shape=[self.opt.batch_size,
                                                        self.opt.img_size, self.opt.img_size, 3])]
                inter_results3 = []

                for i in range(0, 11):
                    f_d = {self.x: x_img,
                           self.xm: x_m,
                           self.x_left_p: x_left_pos,
                           self.x_right_p: x_right_pos,
                           self.y: y_img,
                           self.ym: y_m,
                           self.y_left_p: y_left_pos,
                           self.y_right_p: y_right_pos,
                           self.alpha: np.reshape([i / 10.0], newshape=[self.opt.batch_size, 1])
                           }
                    output = sess.run(self._y2x_inter, feed_dict=f_d)
                    inter_results.append(output)

                for i in range(0, 15):
                    f_d = {self.x: x_img,
                           self.xm: x_m,
                           self.x_left_p: x_left_pos,
                           self.x_right_p: x_right_pos,
                           self.y: y_img,
                           self.ym: y_m,
                           self.y_left_p: y_left_pos,
                           self.y_right_p: y_right_pos,
                           self.alpha: np.reshape([i / 10.0], newshape=[self.opt.batch_size, 1])
                           }
                    output = sess.run(self._y2x_inter, feed_dict=f_d)
                    inter_results1.append(output)

                for i in range(11, 22):
                    f_d = {self.x: x_img,
                           self.xm: x_m,
                           self.x_left_p: x_left_pos,
                           self.x_right_p: x_right_pos,
                           self.y: y_img,
                           self.ym: y_m,
                           self.y_left_p: y_left_pos,
                           self.y_right_p: y_right_pos,
                           self.alpha: np.reshape([i / 10.0], newshape=[self.opt.batch_size, 1])
                           }
                    output = sess.run(self._y2x_inter, feed_dict=f_d)
                    inter_results2.append(output)

                for i in range(-10, 0):
                    f_d = {self.x: x_img,
                           self.xm: x_m,
                           self.x_left_p: x_left_pos,
                           self.x_right_p: x_right_pos,
                           self.y: y_img,
                           self.ym: y_m,
                           self.y_left_p: y_left_pos,
                           self.y_right_p: y_right_pos,
                           self.alpha: np.reshape([i / 10.0], newshape=[self.opt.batch_size, 1])
                           }
                    output = sess.run(self._y2x_inter, feed_dict=f_d)
                    inter_results3.append(output)

                save_images(output_concat,
                            '{}/{:02d}.jpg'.format(self.opt.test_sample_dir, j))
                save_images(local_output_concat,
                            '{}/{:02d}_local.jpg'.format(self.opt.test_sample_dir, j))
                save_images(self.Transpose(np.array(inter_results)),
                            '{}/{:02d}inter1.jpg'.format(self.opt.test_sample_dir, j))
                save_images(self.Transpose(np.array(inter_results1)),
                            '{}/{:02d}inter1_1.jpg'.format(self.opt.test_sample_dir, j))
                logger.debug("Wrote image: {}".format(j))
            coord.request_stop()
            coord.join(threads)

    def train(self):

        self.t_vars = tf.trainable_variables()
        self.dx_vars = [var for var in self.t_vars if 'Dx' in var.name]
        self.dy_vars = [var for var in self.t_vars if 'Dy' in var.name]
        self.gx_vars = [var for var in self.t_vars if 'Gx' in var.name]
        self.gy_vars = [var for var in self.t_vars if 'Gy' in var.name]
        self.e_vars = [var for var in self.t_vars if 'encode' in var.name]
        self.gr_vars = [var for var in self.t_vars if 'Gr' in var.name]
        self.vgg_vars = [var for var in self.t_vars if 'vgg_16' in var.name]

        assert len(self.t_vars) == len(self.dx_vars + self.dy_vars + self.gx_vars
                                      + self.gy_vars + self.e_vars + self.gr_vars + self.vgg_vars)

        self.saver = tf.train.Saver(max_to_keep=10)
        self.p_saver = tf.train.Saver(self.gr_vars)
        opti_Dx = tf.train.AdamOptimizer(self.opt.lr_d * self.lr_decay, beta1=self.opt.beta1, beta2=self.opt.beta2). \
            minimize(loss=self.Dx_loss, var_list=self.dx_vars)
        opti_Dy = tf.train.AdamOptimizer(self.opt.lr_d * self.lr_decay, beta1=self.opt.beta1, beta2=self.opt.beta2). \
            minimize(loss=self.Dy_loss, var_list=self.dy_vars)
        opti_Gx = tf.train.AdamOptimizer(self.opt.lr_g * self.lr_decay, beta1=self.opt.beta1, beta2=self.opt.beta2). \
            minimize(loss=self.Gx_loss, var_list=self.gx_vars)
        opti_Gy = tf.train.AdamOptimizer(self.opt.lr_g * self.lr_decay, beta1=self.opt.beta1, beta2=self.opt.beta2). \
            minimize(loss=self.Gy_loss, var_list=self.gy_vars + self.e_vars)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            start_step = 0

            variables_to_restore = slim.get_variables_to_restore(include=['vgg_16'])
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.opt.vgg_path)

            ckpt = tf.train.get_checkpoint_state(self.opt.checkpoints_dir)
            if ckpt and ckpt.model_checkpoint_path:
                start_step = int(ckpt.model_checkpoint_path.split('model_', 2)[1].split('.', 2)[0])
                logger.debug("New starting point: {}".format(start_step))
                logger.debug("Trying to restore model from: "+ckpt.model_checkpoint_path)
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                try:
                    ckpt = tf.train.get_checkpoint_state(self.opt.pretrain_path)
                    self.p_saver.restore(sess, ckpt.model_checkpoint_path)
                    logger.debug("Read PAM model from: "+ckpt)
                except:
                    logger.debug(" PAM ckpt path may not be correct:")

            step = start_step
            lr_decay = 1

            logger.debug("Start read dataset")
            train_images_x, train_eye_pos_x, train_images_y, train_eye_pos_y, \
                                test_images, test_eye_pos = self.dataset.input()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            real_test_batch, real_test_pos = sess.run([test_images, test_eye_pos])
            logger.debug("Created session.")
            while step <= self.opt.niter:

                if step > 20000 and step % 2000 == 0:
                    lr_decay = (self.opt.niter - step) / float(self.opt.niter - 20000)

                x_data, x_p_data = sess.run([train_images_x, train_eye_pos_x])
                y_data, y_p_data = sess.run([train_images_y, train_eye_pos_y])

                flip_random = random.uniform(0, 1)
                if flip_random > 0.5:
                    x_data = np.flip(x_data, axis=2)
                    y_data = np.flip(y_data, axis=2)
                    x_p_data[:,0] = self.opt.img_size - x_p_data[:, 0]
                    x_p_data[:,2] = self.opt.img_size - x_p_data[:, 2]

                xm_data, x_left_p_data, x_right_p_data = self.get_Mask_and_pos(x_p_data)
                ym_data, y_left_p_data, y_right_p_data = self.get_Mask_and_pos(y_p_data)

                f_d = {self.x: x_data,
                       self.xm: xm_data,
                       self.x_left_p: x_left_p_data,
                       self.x_right_p: x_right_p_data,
                       self.y: y_data,
                       self.ym: ym_data,
                       self.y_left_p: y_left_p_data,
                       self.y_right_p: y_right_p_data,
                       self.lr_decay: lr_decay}

                sess.run(opti_Dx, feed_dict=f_d)
                sess.run(opti_Dy, feed_dict=f_d)
                sess.run(opti_Gx, feed_dict=f_d)
                sess.run(opti_Gy, feed_dict=f_d)

                if step % 100 == 0:
                    output_loss = sess.run(
                        [self.Dx_loss + self.Dy_loss, self.Gx_loss, self.Gy_loss, self.opt.lam_r * self.recon_loss_x,
                         self.opt.lam_r * self.recon_loss_y], feed_dict=f_d)
                    print(
                        "step %d D_loss=%.4f, Gx_loss=%.4f, Gy_loss=%.4f, Recon_loss_x=%.4f, Recon_loss_y=%.4f, lr_decay=%.4f" %
                        (
                            step, output_loss[0], output_loss[1], output_loss[2], output_loss[3], output_loss[4],
                            lr_decay))
                    logger.debug(
                            "step {}/{} D_loss={:.4f}, Gx_loss={:.4f}, Gy_loss={:.4f}, Recon_loss_x={:.4f}, Recon_loss_y={:.4f}, lr_decay={:.4f}".format(step,self.opt.niter, output_loss[0], output_loss[1], output_loss[2], output_loss[3], output_loss[4],lr_decay))

                if np.mod(step, 1000) == 0:
                    o_list = sess.run([self.xl_left, self.xl_right, self.xc, self.xo,
                                       self.yl_left, self.yl_right, self.yc, self.yo,
                                       self.y2x, self.y2x_], feed_dict=f_d)

                    batch_masks, batch_left_eye_pos, batch_right_eye_pos = self.get_Mask_and_pos(real_test_pos)
                    # for test
                    f_d = {self.x: real_test_batch, self.xm: batch_masks,
                           self.x_left_p: batch_left_eye_pos, self.x_right_p: batch_right_eye_pos,
                           self.y: real_test_batch, self.ym: batch_masks,
                           self.y_left_p: batch_left_eye_pos, self.y_right_p: batch_right_eye_pos,
                           self.lr_decay: lr_decay}

                    t_o_list = sess.run([self.xc, self.xo, self.yc, self.yo], feed_dict=f_d)
                    train_trans = self.Transpose(
                        np.array([x_data, o_list[2], o_list[3], o_list[6], o_list[7], o_list[8],
                                  o_list[9]]))
                    l_trans = self.Transpose(np.array([o_list[0], o_list[1], o_list[4], o_list[5]]))
                    test_trans = self.Transpose(np.array([real_test_batch, t_o_list[0],
                                                          t_o_list[1], t_o_list[2], t_o_list[3]]))
                    save_images(l_trans, '{}/{:02d}_lo_{}.jpg'.format(self.opt.sample_dir, step, self.opt.exper_name))
                    save_images(train_trans,
                                '{}/{:02d}_tr_{}.jpg'.format(self.opt.sample_dir, step, self.opt.exper_name))
                    save_images(test_trans,
                                '{}/{:02d}_te_{}.jpg'.format(self.opt.sample_dir, step, self.opt.exper_name))

                if np.mod(step, 1000) == 0:
                    self.saver.save(sess, os.path.join(self.opt.checkpoints_dir, 'model_{:06d}.ckpt'.format(step)))

                step += 1

            save_path = self.saver.save(sess, os.path.join(self.opt.checkpoints_dir, 'model_{:06d}.ckpt'.format(step)))
            # summary_writer.close()

            coord.request_stop()
            coord.join(threads)

            print("Model saved in file: %s" % save_path)

    def Transpose(self, list):
        refined_list = np.transpose(np.array(list), axes=[1, 2, 0, 3, 4])
        refined_list = np.reshape(refined_list, [refined_list.shape[0] * refined_list.shape[1],
                                                 refined_list.shape[2] * refined_list.shape[3], -1])
        return refined_list

    def D(self, x, xl_left, xl_right, scope='D'):

        fc = functools.partial(fully_connect, use_sp=self.opt.use_sp)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            xg_fp = self.global_d(x)
            xl_fp = self.local_d(tf.concat([xl_left, xl_right], axis=-1))
            # Concatenation
            ful = tf.concat([xg_fp, xl_fp], axis=1)
            ful = tf.nn.relu(fc(ful, output_size=1024, scope='fc1'))
            logits = fc(ful, output_size=1, scope='fc2')

            return logits

    def local_d(self, x):

        conv2d_base = functools.partial(conv2d, use_sp=self.opt.use_sp)
        fc = functools.partial(fully_connect, use_sp=self.opt.use_sp)
        with tf.variable_scope("d1", reuse=tf.AUTO_REUSE):
            for i in range(self.opt.n_layers_d):
                output_dim = np.minimum(self.opt.ndf * np.power(2, i + 1), 512)
                x = lrelu(conv2d_base(x, output_dim=output_dim, scope='d{}'.format(i)))
            x = tf.reshape(x, shape=[self.opt.batch_size, -1])
            fp = fc(x, output_size=output_dim, scope='fp')
            return fp

    def global_d(self, x):

        conv2d_base = functools.partial(conv2d, use_sp=self.opt.use_sp)
        fc = functools.partial(fully_connect, use_sp=self.opt.use_sp)
        with tf.variable_scope("d2", reuse=tf.AUTO_REUSE):
            # Global Discriminator Dg
            for i in range(self.opt.n_layers_d):
                dim = np.minimum(self.opt.ndf * np.power(2, i + 1), 256)
                x = lrelu(conv2d_base(x, output_dim=dim, scope='d{}'.format(i)))
            x = tf.reshape(x, shape=[self.opt.batch_size, -1])
            fp = fc(x, output_size=dim, scope='fp')

            return fp

    def Gy(self, input_x, img_mask, fp_local, fp_content, use_sp=False):

        conv2d_first = functools.partial(conv2d, kernel=7, stride=1, use_sp=use_sp)
        conv2d_base = functools.partial(conv2d, kernel=4, stride=2, use_sp=use_sp)
        fc = functools.partial(fully_connect, use_sp=use_sp)
        conv2d_final = functools.partial(conv2d, kernel=7, stride=1, use_sp=use_sp, output_dim=self.opt.output_nc)
        with tf.variable_scope("Gy", reuse=tf.AUTO_REUSE):

            x = tf.concat([input_x, img_mask], axis=3)
            u_fp_list = []
            x = lrelu(instance_norm(conv2d_first(x, output_dim=self.opt.ngf, scope='conv'), scope='IN'))
            for i in range(self.opt.n_layers_g):
                c_dim = np.minimum(self.opt.ngf * np.power(2, i + 1), 256)
                x = lrelu(instance_norm(conv2d_base(x, output_dim=c_dim, scope='conv{}'.format(i)), scope='IN{}'.format(i)))
                u_fp_list.append(x)

            bottleneck = tf.reshape(x, shape=[self.opt.batch_size, -1])
            bottleneck = fc(bottleneck, output_size=256, scope='FC1')
            bottleneck = tf.concat([bottleneck, fp_local, fp_content], axis=1)

            h, w = x.shape.as_list()[-3], x.shape.as_list()[-2]
            de_x = lrelu(fc(bottleneck, output_size=256 * h * w, scope='FC2'))
            de_x = tf.reshape(de_x, shape=[self.opt.batch_size, h, w, 256])

            ngf = c_dim
            for i in range(self.opt.n_layers_g):
                c_dim = np.maximum(int(ngf / np.power(2, i)), 16)
                de_x = tf.concat([de_x, u_fp_list[len(u_fp_list) - (i + 1)]], axis=3)
                de_x = tf.nn.relu(instance_norm(de_conv(de_x,
                                                        output_shape=[self.opt.batch_size, h * pow(2, i + 1),
                                                                      w * pow(2, i + 1), c_dim], use_sp=use_sp,
                                                        scope='deconv{}'.format(i)), scope='IN_{}'.format(i)))
            de_x = conv2d_final(de_x, scope='output_conv')

            return input_x + tf.nn.tanh(de_x) * img_mask

    def Gx(self, input_x, img_mask, fp, use_sp=False):

        conv2d_first = functools.partial(conv2d, kernel=7, stride=1, use_sp=use_sp)
        conv2d_base = functools.partial(conv2d, kernel=4, stride=2, use_sp=use_sp)
        fc = functools.partial(fully_connect, use_sp=use_sp)
        conv2d_final = functools.partial(conv2d, kernel=7, stride=1, use_sp=use_sp, output_dim=self.opt.output_nc)
        with tf.variable_scope("Gx", reuse=tf.AUTO_REUSE):

            x = tf.concat([input_x, img_mask], axis=3)
            u_fp_list = []
            x = lrelu(instance_norm(conv2d_first(x, output_dim=self.opt.ngf, scope='conv'), scope='IN'))
            for i in range(self.opt.n_layers_g):
                c_dim = np.minimum(self.opt.ngf * np.power(2, i + 1), 256)
                x = lrelu(instance_norm(conv2d_base(x, output_dim=c_dim, scope='conv{}'.format(i)), scope='IN{}'.format(i)))
                u_fp_list.append(x)

            bottleneck = tf.reshape(x, shape=[self.opt.batch_size, -1])
            bottleneck = fc(bottleneck, output_size=256, scope='FC1')

            bottleneck = tf.concat([bottleneck, fp], axis=-1)
            h, w = x.shape.as_list()[-3], x.shape.as_list()[-2]
            de_x = lrelu(fc(bottleneck, output_size=256 * h * w, scope='FC2'))
            de_x = tf.reshape(de_x, shape=[self.opt.batch_size, h, w, 256])
            ngf = c_dim
            for i in range(self.opt.n_layers_g):
                c_dim = np.maximum(int(ngf / np.power(2, i)), 16)
                de_x = tf.concat([de_x, u_fp_list[len(u_fp_list) - (i + 1)]], axis=3)
                de_x = tf.nn.relu(instance_norm(de_conv(de_x, output_shape=[self.opt.batch_size, h * pow(2, i + 1),
                                                                            w * pow(2, i + 1), c_dim], use_sp=use_sp,
                                                        scope='deconv{}'.format(i)), scope='IN_{}'.format(i)))

            de_x = conv2d_final(de_x, scope='output_conv')
            return input_x + tf.nn.tanh(de_x) * img_mask

    def Gr(self, input_x, use_sp=False):
        print(input_x.shape)
        conv2d_first = functools.partial(conv2d, kernel=7, stride=1, use_sp=use_sp)
        conv2d_base = functools.partial(conv2d, kernel=4, stride=2, use_sp=use_sp)
        fc = functools.partial(fully_connect, use_sp=use_sp)
        with tf.variable_scope("Gr", reuse=tf.AUTO_REUSE):
            x = input_x
            x = lrelu(instance_norm(conv2d_first(x, output_dim=self.opt.ngf, scope='conv'), scope='IN'))
            for i in range(self.opt.n_layers_r):
                c_dim = np.minimum(self.opt.ngf * np.power(2, i + 1), 128)
                x = lrelu(instance_norm(conv2d_base(x, output_dim=c_dim, scope='conv{}'.format(i)), scope='IN{}'.format(i)))

            bottleneck = tf.reshape(x, shape=[self.opt.batch_size, -1])
            bottleneck = fc(bottleneck, output_size=256, scope='FC1')

            return bottleneck

    def encode(self, x):

        conv2d_first = functools.partial(conv2d, kernel=7, stride=1)
        conv2d_base = functools.partial(conv2d, kernel=4, stride=2)
        with tf.variable_scope("encode", reuse=tf.AUTO_REUSE):
            nef = self.opt.nef
            x = tf.nn.relu(instance_norm(conv2d_first(x, output_dim=nef, scope='e_c1'), scope='e_in1'))
            for i in range(self.opt.n_layers_e):
                x = tf.nn.relu(instance_norm(conv2d_base(x, output_dim=min(nef * pow(2, i + 1), 128), scope='e_c{}'.format(i + 2)),
                                  scope='e_in{}'.format(i + 2)))
            bottleneck = tf.reshape(x, [self.opt.batch_size, -1])
            content = fully_connect(bottleneck, output_size=2, scope='e_ful1')

            return content

    def get_Mask_and_pos(self, eye_pos):

        def get_pos(eye_pos):
            o_eye_pos = np.zeros(shape=(self.opt.batch_size, 4), dtype=np.int32)
            o_eye_pos[:, 3] = (eye_pos[:, 0] + self.opt.crop_w / 2)
            o_eye_pos[:, 2] = (eye_pos[:, 1] + self.opt.crop_h / 2)
            o_eye_pos[:, 1] = (eye_pos[:, 0] - self.opt.crop_w / 2)
            o_eye_pos[:, 0] = (eye_pos[:, 1] - self.opt.crop_h / 2)

            return o_eye_pos

        def get_Mask(left_eye_pos, right_eye_pos):
            batch_mask = np.zeros(shape=(self.opt.batch_size, self.opt.img_size, self.opt.img_size, self.opt.output_nc))
            # x, y = np.meshgrid(range(img_size), range(img_size))
            for i in range(self.opt.batch_size):
                batch_mask[i,
                left_eye_pos[i][0]:left_eye_pos[i][2],
                left_eye_pos[i][1]:left_eye_pos[i][3], :] = 1
                batch_mask[i,
                right_eye_pos[i][0]:right_eye_pos[i][2],
                right_eye_pos[i][1]:right_eye_pos[i][3], :] = 1

            return batch_mask

        left_eye_pos = get_pos(eye_pos[:, 0:2])
        right_eye_pos = get_pos(eye_pos[:, 2:4])
        mask = get_Mask(left_eye_pos, right_eye_pos)

        return mask, left_eye_pos / float(self.opt.img_size), \
               right_eye_pos / float(self.opt.img_size)

import setproctitle
setproctitle.setproctitle("GazeGAN")
opt = TrainOptions().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

if __name__ == "__main__":
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileHandler = handlers.RotatingFileHandler(filename="error.log",maxBytes=1024000, backupCount=10, mode="a")
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(formatter)

    smtpHandler = handlers.SMTPHandler(
                mailhost = ("smtp.gmail.com", 587),
                fromaddr = "email",
                toaddrs = "email",
                subject = "Project :: ExGAN FID Stats",
                credentials = ('email','password'),
                secure = ()
                )
    smtpHandler.setLevel(logging.DEBUG)
    smtpHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(smtpHandler)
    tf.disable_v2_behavior()
    logger.debug("Creating Dataset.")
    dataset = Dataset(opt)
    gaze_gan = Gaze_GAN(dataset, opt)
    gaze_gan.build_model()
    #gaze_gan.build_test_model()
    logger.debug("Starting testing...")
    gaze_gan.test()


