import numpy as np
import keras
from keras import backend
from keras.models import load_model
import tensorflow as tf
from cleverhans.utils_keras import KerasModelWrapper
from keras.applications import inception_v3
from keras.applications import imagenet_utils
import cleverhans.attacks
from cleverhans.attacks import CarliniWagnerL2
import scipy.misc


keras_model = inception_v3.InceptionV3(weights='imagenet')

# Set the learning phase to false, the model is pre-trained.
backend.set_learning_phase(False)

# Set TF random seed to improve reproducibility
tf.set_random_seed(1234)

if keras.backend.image_dim_ordering() != 'tf':
    keras.backend.set_image_dim_ordering('tf')
    print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
          "'th', temporarily setting to 'tf'")

# Retrieve the tensorflow session
sess = backend.get_session()

image = scipy.misc.imread("cat.png")

# Resizing the image to be of size 299 * 299
image = np.array(scipy.misc.imresize(image, (299, 299)),
                 dtype=np.float32)

# converting each pixel to the range [0,1] (Normalization)

image = np.array([image / 255.0])

wrap = KerasModelWrapper(keras_model)

cw = cleverhans.attacks.CarliniWagnerL2(wrap, sess = sess)

# carlini and wagner
cw_params = {'batch_size': 1,
             'confidence': 10,
             'learning_rate': 0.1,
             'binary_search_steps': 5,
             'max_iterations': 1000,
             'abort_early': True,
             'initial_const': 0.01,
             'clip_min': 0,
             'clip_max': 1}

adv_cw = cw.generate_np(image, **cw_params)
scipy.misc.imsave("adversarial_cw.png", adv_cw[0])
