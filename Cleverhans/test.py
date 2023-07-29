# Code based on https://github.com/tensorflow/cleverhans/blob/master/examples/nips17_adversarial_competition/sample_attacks/fgsm/attack_fgsm.py

import os
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.attacks import LBFGS
import numpy as np
from PIL import Image
from  cv2 import imread
#from cv2 import imsave
import tensorflow as tf
from keras.contrib.slim.nets import inception

from urllib.request import urlretrieve
import os
import json
import matplotlib.pyplot as plt


slim = tf.contrib.slim
tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')
tf.flags.DEFINE_string(
    'checkpoint_path', './model/inception_v3.ckpt', 'Path to checkpoint for inception network.')
# tf.flags.DEFINE_string(
#     'checkpoint_path', './checkpoint/adv_inception_v3_2017_08_18/adv_inception_v3.ckpt', 'Path to checkpoint for inception network.')
# tf.flags.DEFINE_string(
#     'checkpoint_path', './checkpoint/ens4_adv_inception_v3_2017_08_18/ens4_adv_inception_v3.ckpt', 'Path to checkpoint for inception network.')
tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')
tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')
tf.flags.DEFINE_float(
    'max_epsilon', 4.0, 'Maximum size of adversarial perturbation.')
tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')
tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')
tf.flags.DEFINE_integer(
    'batch_size', 1, 'How many images process at one time.')
FLAGS = tf.flags.FLAGS

nb_images = 10

def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    # Limit to first 20 images for this example
    for filepath in sorted(tf.gfile.Glob(os.path.join(input_dir, '*.png')))[:nb_images]:
        with tf.gfile.Open(filepath, "rb") as f:
            images[idx, :, :, :] = imread(f, mode='RGB').astype(np.float)*2.0/255.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:#16
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images

def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')

class InceptionModel(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        _, end_points = inception.inception_v3(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    output = end_points['Predictions']
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs
#___________________________________________________

def testinception(testimage,reuse):
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, _ =inception.inception_v3(
            testimage, 1001, is_training=False, reuse=reuse)
        print('i am here:',logits)
        logits = logits[:,1:] # ignore background class
        probs = tf.nn.softmax(logits) # probabilities
        print('i am here:', probs)

    return logits, probs
imagenet_json = 'model/imagenet.json'
with open(imagenet_json) as f:
    imagenet_labels = json.load(f)

def classify(img):
    p = sess.run(probs, feed_dict={testimage: img})[0]
    topk = list(p.argsort()[-1:][::-1])#reverse

    for i in topk:
        label=imagenet_labels[i]
        print(label)
    return topk[0]

def show_classify(img):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    fig.sca(ax1)
    p = sess.run(probs, feed_dict={testimage: img})[0]
    img = (((img[0, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
    ax1.imshow(img)
    fig.sca(ax1)

    topk = list(p.argsort()[-10:][::-1])
    topprobs = p[topk]
    ax2.bar(range(10), topprobs)

    plt.sca(ax2)
    plt.ylim([0, 1.1])
    plt.xticks(range(10),
               [imagenet_labels[i][:15] for i in topk],
               rotation='vertical')
    fig.subplots_adjust(bottom=0.2)
    plt.show()
    return imagenet_labels[topk[0]]

#___________________________________________________

eps = 2.0 * 16.0 / 255.0 #0.125
batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
num_classes = 1001



tf.logging.set_verbosity(tf.logging.INFO)

y_target = 1

with tf.Graph().as_default():
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    model = InceptionModel(num_classes)

    mim = MomentumIterativeMethod(model)

    y_target = tf.one_hot([y_target], 1001)
    x_adv = mim.generate(x_input,y_target = y_target, eps=eps, clip_min=-1, clip_max=1)

    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=FLAGS.checkpoint_path,
        master=FLAGS.master)


    testimage = tf.placeholder(tf.float32, shape=[1, FLAGS.image_height, FLAGS.image_width, 3])

    logits,probs = testinception(testimage, reuse=True)

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:

        for filenames, images in load_images("./images/", batch_shape):

            reshape_image=np.reshape(images[0, :, :, :],[1, FLAGS.image_height, FLAGS.image_width, 3])
            print("Original image: ", show_classify(reshape_image), "\n")
            adv_images = sess.run(x_adv, feed_dict={x_input: images})

            save_images(adv_images, filenames , "./output")
            reshape_image=np.reshape(adv_images[0, :, :, :],[1, FLAGS.image_height, FLAGS.image_width, 3])
            print("Adversarial example: ", show_classify(reshape_image), "\n")
