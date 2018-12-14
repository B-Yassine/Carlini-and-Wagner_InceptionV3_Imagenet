# C-W_InceptionV3_Imagenet
Simple implementation of the C&amp;W attack on a pre-trained Keras's InceptionV3 on Imagenet

# Adversarial Examples
Adversarial examples are inputs that has been slightly modified to be imperceptible by the human and cause a misclassification
Formalization often used: for a clean input x, an input x’ is an adversarial example if it is misclassified and d(x, x’) < eps.

For instance: For our example here is what we get using the C&W attack:

![](adversarial_example.png)
