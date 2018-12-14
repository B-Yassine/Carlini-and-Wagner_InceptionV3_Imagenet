import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import inception_v3
from keras.applications import imagenet_utils
import matplotlib.pyplot as plt

# Load the Inception_V3 model
inception_model = inception_v3.InceptionV3(weights='imagenet')


filename = "adversarial_cat_cw.png"
# load an image in PIL format
original = load_img(filename, target_size=(299, 299))

numpy_image = img_to_array(original)

image_batch = np.expand_dims(numpy_image, axis=0)

# prepare the image for the inceptionv3 model
processed_image = inception_v3.preprocess_input(image_batch.copy())

# classify the image

preds = inception_model.predict(processed_image)

P = imagenet_utils.decode_predictions(preds)

#classified label
print("===")
print("GoogleNet is {:.2f} % sure that it's a {}".format(P[0][0][2] * 100, P[0][0][1]))
print("===")


plt.figure(1)
plt.subplot(121)

plt.title("Input image")
plt.imshow(np.uint8(numpy_image))

labels = [ P[0][0][1] + '['+str(P[0][0][2]*100)+' %]', P[0][1][1]+ '['+str(P[0][1][2]*100)+' %]', P[0][2][1]+ '['+str(P[0][2][2]*100)+' %]',
           P[0][3][1] +'[' + str(P[0][3][2]*100) + ' %]', P[0][4][1]+ '['+str(P[0][4][2]*100)+' %]']

sizes = [P[0][0][2], P[0][1][2], P[0][2][2], P[0][3][2], P[0][4][2]]
colors = ['red', 'blue', 'green','yellow', 'brown']

plt.subplot(122)
patches, texts = plt.pie(sizes, colors=colors, startangle=90)

plt.legend(patches, labels, loc='best')
plt.title("Top 5 Classification accuracy")
plt.axis('equal')
plt.tight_layout()
plt.show()
