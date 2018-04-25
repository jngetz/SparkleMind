# SparkleMind

## Producing Data
to use 28x28 images, run reflect.py followed by the source file name and the destination file name.
In order to modify to use larger images, reflect.py must have all the environment elements use the proper size
The assumption in the design of the program is NxN images, camera, and mirror. As such, the elements must
match the size of the input images.

## Using data in Convolutional Neural Network
A list of absolute paths to the images and labels being used must be provided to the convolutional neural network.
The path to this list must be used in cnn_sparkle.py under the helper functions eval_input_fn and data_input_fn which
read in the image data and associated label data

## Running the Convolutional Neural Network
running cnn_sparkle.py will run the neural network. The network will store itself at the location of the variable directory
in main of cnn_sparkle.py

Predictions can be produced if using the command "cnn_sparkle.py P". predictions are saved in the directory specified in
the image save function.
