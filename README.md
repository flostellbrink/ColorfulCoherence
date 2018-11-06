# ColorfulCoherence

## Architecture

First part is a copy of the Colorful Image Colorization architecture (CIC).

That part is trained with categorical crossentropy compared to binned input in 64x64x313.

We transform the categorical layer, using the grayscale input, into a coherent categorical layer.
The color loss is regularized using softmax division (softmaxing the new dist, multiplying that with values in old (dot style) and dividing 1 (sum of softmax) by that number).
This is checked with upscaled colorization (256, 256, 313)

Finally the coherent output has an objective function that again applies softmax to the dist, multiplies that with the color lookup (dot style) and computes the absolute error of the image gradients.
Applied to coherent color and original image so, (256, 256, 3) with diff of gradients compressed into (256, 256, 1)

The softmax functions with temperature allow us to approximate an actual max function (that will be used to select the final value) while still having gradients.

# Idea

Working definition of coherence: similarity of gradients
Objective function does not need to compromise colorfullness! (Yeah!)