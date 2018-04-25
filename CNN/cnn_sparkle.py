from __future__ import absolute_import #
from __future__ import division        #
from __future__ import print_function  #

# importing tensorflow and numpy
from PIL import Image as I
import tensorflow as tf
import numpy as np
import sys

#logging
tf.logging.set_verbosity(tf.logging.INFO)


#helper funcitons
def file_to_list(filename):
    img=open(filename)
    img_temp_list = img.readlines()
    img_list = []
    line = img_temp_list[0]
    for line in img_temp_list:
        line = line.strip("\n")
        img_list.append(line)
    return img_list

def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    label_string = tf.read_file(label)
    label_decode = tf.image.decode_jpeg(label_string,channels=3)
    label = tf.cast(label_decode, tf.float32)
    return image, label

def _parse_single(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    return image

def data_input_fn():
    # loading training data
    img_names = tf.constant(file_to_list('../data/rmnist/imglist.txt'))
    img_labels = tf.constant(file_to_list('../data/mnist/imglist.txt'))
    dataset = tf.data.Dataset.from_tensor_slices((img_names, img_labels))
    dataset = dataset.shuffle(buffer_size=7000) #remove order bias
    dataset = dataset.map(_parse_function)
    dataset_batch = dataset.batch(1)
    data_iter = dataset_batch.make_one_shot_iterator()
    train_data, label = data_iter.get_next()
    train_data = tf.reduce_mean(train_data, 3,True, name="train_data")
    label = tf.reduce_mean(label, 3,True, name="label")
    return (train_data, label)

def eval_input_fn():
    # eval data
    eval_names = tf.constant(file_to_list('../data/eval_rmnist/imglist.txt'))
    eval_labels = tf.constant(file_to_list('../data/eval_mnist/imglist.txt'))
    evalset = tf.data.Dataset.from_tensor_slices((eval_names, eval_labels))
    evalset = evalset.shuffle(buffer_size=7000)
    evalset = evalset.map(_parse_function)
    evalset_batch = evalset.batch(1)
    eval_iter = evalset_batch.make_one_shot_iterator()
    eval_data, label = eval_iter.get_next()
    eval_data = tf.reduce_mean(eval_data, 3, True)
    label = tf.reduce_mean(label, 3, True)
    return (eval_data, label)


def predict_input_fn():
    # predict data
    prediction_in = tf.constant(file_to_list("../data/eval_rmnist/imglist.txt"))
    pred_set = tf.data.Dataset.from_tensor_slices((prediction_in))
    pred_set = pred_set.map(_parse_single)
    pred_batch = pred_set.batch(1)
    pred_iter = pred_batch.make_one_shot_iterator()
    pred_img = pred_iter.get_next()
    pred_img = tf.reduce_mean(pred_img, 3, True)
    return pred_img

#end helper functions

def cnn_model_fn(features, labels, mode):
    # # normalize batch
    tf.layers.batch_normalization(features)
    # input layer
    ## paramater 2 [batch size, image_height, image_width, channels]
    input_layer = tf.reshape(features, [1, 28, 28, 1])
    # convolution layer #1
    # strides 1 pixel in x and y direction
    num_filters = 15
    convolution1 = tf.layers.conv2d(inputs = input_layer, filters=num_filters, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
    # Conversion to dense layer
    flat = tf.reshape(convolution1, [-1,28*28*num_filters])

    # convert to some form of answer. output layer
    logits = tf.layers.dense(inputs=flat, units=28*28)


    # compile predictions based on logits layer
    prediction = {
        "grayscale": tf.clip_by_value(tf.multiply(tf.divide(tf.subtract(logits,tf.reduce_min(logits)),tf.subtract(tf.reduce_max(logits),tf.reduce_min(logits))),255.0), 0.0, 255.0, name="grayscale")
    }
    fake_logit = tf.reshape(logits, [1,28*28], name="logit_layer")

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=prediction)
    # loss calculation
    loss = tf.multiply(tf.losses.mean_squared_error(tf.reshape(labels, [1,784]), tf.reshape(prediction['grayscale'], [1,784])), 1.0, name="mse")

    # training operation for TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)



    eval_metric_ops = {
        "accuracy": tf.metrics.mean_squared_error(
            labels=tf.reshape(labels,[1,784]), predictions=tf.reshape(prediction["grayscale"], [1,784])
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )

def main(unused_argv):


    # estimator creation
    directory = "./noflip/neg_6_learn/1_conv_layer/cnn_sparkle_15"
    results_file = open("./results.txt", "a")
    sparkle_estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=directory
    )
    logging_hook = tf.train.LoggingTensorHook(
        tensors={"mse":"mse"}, every_n_iter=100
    )

    if len(sys.argv) == 2 and sys.argv[1] == "P":
        print("in if stmt")
        predicts = list(sparkle_estimator.predict(
            input_fn=predict_input_fn
        ))
        count = 1
        for i in predicts:
            img = I.fromarray(np.uint8(np.reshape(i['grayscale'], [28,28])))
            img.save("../ray_trace/draft_2/prediction3_results/"+str(count)+".jpg")
            count += 1

    else:
        sparkle_estimator.train(
            input_fn=data_input_fn,
            steps=42000,
            hooks=[]
        )
        eval_results = sparkle_estimator.evaluate(
            input_fn=eval_input_fn,
            hooks=[]
        )
        print("done evaluating")
        print(eval_results)
        print(eval_results, file=results_file)


if __name__ == '__main__':
    tf.app.run()
