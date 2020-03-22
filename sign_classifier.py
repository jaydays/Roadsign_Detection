#
# Train the top categories model (Text, Photo, Drawing, Abstract)
# Using Keras
#
# This module could also be used to train another model with binary categories
# given a csv file with image pathnames and the labels
#
# More work needs to be done to extend this to train a model with non-binary categories
# May need to detect the number of category values per category and change build_model() and train() accordingly
# Also modify export_model_to_tensorflow to include a classification signature and human-readable labels potentially

import csv
import math
import os
import shutil

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.python.keras.layers import Dense, Reshape, Conv1D, Activation, Input, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.core.example import example_pb2, feature_pb2
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def, classification_signature_def

import image_ops as custom_image


# siamese helpers
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


# end siamese helpers


def get_class_names():
    # TODO: return the actual sign names
    return list(range(1, 48))


def get_class_weights(csv_path, label_suffix='', im_per_row=1):
    with open(csv_path, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        header = next(reader, None)

        class_labels = [label + label_suffix for label in header[im_per_row:]]
        class_weights = {}
        for label_name in class_labels:
            class_weights[label_name] = {}

        for row in reader:
            if len(row) == 0:
                continue

            label = [int(i) for i in row[im_per_row:]]

            if len(label) != len(class_labels):
                raise ValueError("There are a different number of labels in " + csv_path)

            for idx, label_name in enumerate(class_labels):
                class_weights[label_name][label[idx]] = class_weights[label_name].get(label[idx], 0) + 1

        # normalize class_counts by label name
        for label_name in class_labels:
            sum_counts = float(sum(class_weights[label_name].values()))
            for label_value in class_weights[label_name].keys():
                class_weights[label_name][label_value] = sum_counts / class_weights[label_name][label_value]

        return class_weights


def get_last_checkpoint(saved_model_dir, file_prefix):
    """
    get the most recent epoch of saved models with filenames of the form file_prefix.{epoch:02d}-{val_loss:.2f}.hdf5
    file_prefix should not have any periods
    :param saved_model_dir:
    :param file_prefix: what the filename starts with
    :return: tuple of file path and latest epoch
    """

    if '.' in file_prefix:
        raise ValueError('file_prefix should not contain any periods')

    # extract out the epoch from the files in this directory beginning with the file_prefix
    file_names = [f for f in os.listdir(saved_model_dir)
                  if os.path.isfile(os.path.join(saved_model_dir, f)) and f.startswith(file_prefix)]

    epochs = [int(f.split('.')[1].split('-')[0]) for f in file_names]

    if len(epochs) == 0:
        return '', 0

    last_epoch = max(epochs)

    return os.path.join(saved_model_dir, file_names[epochs.index(last_epoch)]), last_epoch


def print_model_calibration_stats(model_filename, csv_path_validation):
    original_model = load_model(model_filename)
    categories = get_class_names()
    category_order = [layer.name for layer in original_model.layers if layer.name in categories]

    image_generator = custom_image.CustomImageDataGenerator(10, 0.05, 0.05, 0.1, 0.1, 0,
                                                            'nearest', 0.5, False, False, False,
                                                            preprocess_input, (299, 299), 1)

    batch_size = 50
    batches_per_epoch_validation = get_batches_per_epoch(csv_path_validation, True, batch_size)
    num_epochs = 50
    total_batches = num_epochs * batches_per_epoch_validation

    # test the calibration with fixed bin size
    bin_size = 0.1
    num_bins = 1 / bin_size

    # for each predicted confidence bin, compute the accuracy of that bin
    generator = image_generator.flow_from_csv(csv_path_validation, ',', True, batch_size, True)

    accuracies = {}
    accuracies_totals = {}
    for confidence in np.arange(0, 1.01, bin_size):
        accuracies[confidence] = 0
        accuracies_totals[confidence] = 0

    for batch_num in range(0, total_batches):
        print('Batch ' + str(batch_num) + '/' + str(total_batches))
        batch_x, batch_y = next(generator)
        batch_predictions_y = original_model.predict_on_batch(batch_x)
        for category_idx, category_batch_prediction_y in enumerate(batch_predictions_y):
            category = category_order[category_idx]
            category_batch_y = batch_y[category]
            for idx, category_prediction_y in enumerate(category_batch_prediction_y):
                category_y = category_batch_y[idx]
                if len(category_prediction_y) == 1:
                    confidence = math.floor(category_prediction_y * num_bins) * bin_size
                    if (confidence >= 0.5 and category_y >= 0.5) or (confidence < 0.5 and category_y < 0.5):
                        accuracies[confidence] += 1
                else:
                    predicted_class = np.argmax(category_prediction_y)
                    confidence = math.floor(category_prediction_y[predicted_class] * num_bins) * bin_size
                    if predicted_class == category_y:
                        accuracies[confidence] += 1
                accuracies_totals[confidence] += 1

    for confidence in np.arange(0, 1.01, bin_size):
        if accuracies_totals[confidence] > 0:
            accuracies[confidence] /= accuracies_totals[confidence]
        print(str(confidence) + ':' + str(accuracies[confidence]))


def calibrate_model(model_filename, csv_path_validation, csv_path_test, save_dir, learning_rate=0.001):
    log_dir = '{model_dir}/logs_calibrate'.format(model_dir=save_dir)
    calibrated_save_dir = save_dir + '/calibrated_weights'
    save_file = calibrated_save_dir + '/calibrated_weights.{epoch:02d}-{loss:.2f}.hdf5'

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    if not os.path.isdir(calibrated_save_dir):
        os.mkdir(calibrated_save_dir)

    categories = get_class_names(csv_path_validation)
    class_weights = get_class_weights(csv_path_validation, '_activation')

    # modify the model by adding a calibration layer before the
    # output layer activation function
    calibrated_model = build_model(categories, input_tensor=None, add_calibration_layer=True)
    calibrated_model.load_weights(model_filename, by_name=True)

    # fix all of the weights, except for the calibration layer
    calibrated_names = [name + '_calibrated' for name in categories]
    for layer in calibrated_model.layers:
        if layer.name not in calibrated_names:
            layer.trainable = False
        else:
            layer.trainable = True

    image_generator = custom_image.CustomImageDataGenerator(10, 0.05, 0.05, 0.1, 0.1, 0,
                                                            'nearest', 0.5, False, False, False,
                                                            preprocess_input, (299, 299), 1)

    batch_size = 50
    batches_per_epoch_validation = get_batches_per_epoch(csv_path_validation, True, batch_size)
    num_epochs = 10
    batches_per_epoch_test = get_batches_per_epoch(csv_path_test, True, batch_size)

    calibration_callbacks = [keras.callbacks.ModelCheckpoint(save_file), keras.callbacks.TensorBoard(log_dir=log_dir)]

    # now do some calibration based on the validation set
    optimizer = optimizers.RMSprop(lr=learning_rate)
    calibrated_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

    calibrated_model.fit_generator(
        image_generator.flow_from_csv(csv_path_validation, ',', True, batch_size, True, label_suffix='_activation'),
        batches_per_epoch_validation,
        num_epochs,
        1,
        calibration_callbacks,
        image_generator.flow_from_csv(csv_path_validation, ',', True, batch_size, True, label_suffix='_activation'),
        batches_per_epoch_validation,
        class_weights,
        batches_per_epoch_test,
        9,
        True,
        0
    )

    # test it
    print(calibrated_model.evaluate_generator(
        image_generator.flow_from_csv(csv_path_test, ',', True, batch_size, True, label_suffix='_activation'),
        batches_per_epoch_test * num_epochs,
        batches_per_epoch_test,
        9,
        True
    ))


def build_model(categories, input_tensor=None, add_calibration_layer=False):
    """
    Load the Inception model that has been pre-trained on ImageNet
    :param categories: a list of the category names that you want to predict. this method assumes binary categories
    :param input_tensor: input tensor that the model should take (e.g. if want to add preprocessing step.
                         if None, it'll take the default input)
    :param add_calibration_layer: add a calibration layer before the final activation layer
    :return: tensor representing the image where channel values are scaled between [-1,1]
             and resized to 299x299
    """

    # load the ImageNet model, it also adds a pooling layer
    base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_tensor=input_tensor)

    label_layers = []

    # add a fully connected layer to predict each binary category
    for idx, category in enumerate(categories):
        if not add_calibration_layer:
            label_layer = Dense(1, activation='sigmoid', name=category)(base_model.output)
        else:
            label_layer = Dense(1, activation=None, name=category)(base_model.output)

            # this adds a temperature scaling layer before the sigmoid function, which can be tuned on the validation
            # set to better calibrate the confidence scores
            label_layer = Reshape((1, 1), name=category + "_precalibrate")(label_layer)
            label_layer = Conv1D(1, kernel_size=1, use_bias=False, kernel_initializer='ones',
                                 name=category + "_calibrated")(label_layer)
            label_layer = Reshape((1,), name=category + "_postcalibrate")(label_layer)
            label_layer = Activation(activation='sigmoid', name=category + "_activation")(label_layer)
        label_layers.append(label_layer)

    model = Model(inputs=base_model.input, outputs=label_layers)

    return model


def train(csv_path, new_learning_rate, old_learning_rate, model_label, debug, im_per_row=1):
    model_dir = os.path.join('models', model_label)
    log_dir_finetune = '{model_dir}/logs_finetune'.format(model_dir=model_dir)
    log_dir_fulltune = '{model_dir}/logs_fulltune'.format(model_dir=model_dir)
    save_dir = '{model_dir}/saved_models'.format(model_dir=model_dir)

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
        os.mkdir(log_dir_finetune)
        os.mkdir(log_dir_fulltune)
        os.mkdir(save_dir)

    batch_size = int(32 / im_per_row)
    num_epochs_finetune = 100
    num_epochs_fulltune = 100

    # either finetune_weights or fulltune_weights
    start_mode = 'fulltune_weights'

    # get the current state
    model_filename, start_epoch = get_last_checkpoint(save_dir, 'fulltune_weights')

    if start_epoch >= num_epochs_fulltune:
        # we're done
        print('Already finished training')
        return

    if start_epoch == 0:
        model_filename, start_epoch = get_last_checkpoint(save_dir, 'finetune_weights')
        if start_epoch < num_epochs_finetune:
            start_mode = 'finetune_weights'
        else:
            start_epoch = 0

    if model_filename:
        print('Loading ' + model_filename)

    # the categories to predict
    categories = get_class_names()

    if start_mode == 'finetune_weights' and start_epoch == 0:
        model = build_model(categories)
    else:
        # load weights
        model = load_model(model_filename)

    # initialize the image generator and number of batches needed
    image_generator = custom_image.CustomImageDataGenerator(10, 0.05, 0.05, 0.1, 0.1, 0,
                                                            'nearest', 0.5, False, False, False,
                                                            preprocess_input, (299, 299), 1, im_per_row)

    batches_per_epoch_train = get_batches_per_epoch(csv_path_train, True, batch_size)
    batches_per_epoch_validation = get_batches_per_epoch(csv_path_validation, True, batch_size)

    print('batches ' + str(batches_per_epoch_train))

    save_file_finetune = save_dir + '/finetune_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    save_file_fulltune = save_dir + '/fulltune_weights.{epoch:02d}-{val_loss:.2f}.hdf5'

    callbacks_finetune = [
        keras.callbacks.ModelCheckpoint(save_file_finetune),
        keras.callbacks.TensorBoard(log_dir=log_dir_finetune)]

    callbacks_fulltune = [
        keras.callbacks.ModelCheckpoint(save_file_fulltune),
        keras.callbacks.TensorBoard(log_dir=log_dir_fulltune)]

    class_weights = get_class_weights(csv_path_train, '', im_per_row)

    if debug:
        batches_per_epoch_train = 10
        num_epochs_finetune = 10
        num_epochs_fulltune = 10
        start_epoch = 0
        callbacks_finetune = []
        callbacks_fulltune = []

    '''
    print('Petar debug')
    ig = image_generator.flow_from_csv(csv_path_train, ',', True, batch_size, True)
    print(ig)
    print(ig.image_paths[0:3], len(ig.image_paths))
    print(ig.image_labels, len(ig.image_labels))
    print('class weights')
    print(class_weights)
    gg = image_generator.flow_from_csv(csv_path_train, ',', True, batch_size, True)
    bx, by = gg._get_batches_of_transformed_samples([0, 1, 2])
    print(bx[0], "bx[0]", len(bx[0]))
    print(by, "by", len(by))
    return
    '''

    if im_per_row == 2:  # siamese training
        input_a = Input(shape=(299, 299, 3))
        input_b = Input(shape=(299, 299, 3))

        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        processed_a = model(input_a)
        processed_b = model(input_b)

        distance = Lambda(euclidean_distance,
                          output_shape=eucl_dist_output_shape)([processed_a, processed_b])

        model = Model([input_a, input_b], distance)

        # train
        rms = optimizers.RMSprop()
        model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])

        model.fit_generator(
            generator=image_generator.flow_from_csv(csv_path_train, ',', True, batch_size, True),
            steps_per_epoch=batches_per_epoch_train,
            epochs=num_epochs_finetune,
            verbose=1,
            callbacks=callbacks_finetune,
            validation_data=image_generator.flow_from_csv(csv_path_validation, ',', True, batch_size, True),
            validation_steps=batches_per_epoch_validation,
            # class_weight=class_weights,
            # max_queue_size=batches_per_epoch_train,
            workers=9,
            use_multiprocessing=True,
            initial_epoch=start_epoch)

        return

    if start_mode == 'finetune_weights':

        # first train only the new layers
        for idx, layer in enumerate(model.layers):
            if layer.name not in categories:
                layer.trainable = False

        new_optimizer = optimizers.RMSprop(lr=new_learning_rate)
        model.compile(optimizer=new_optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

        print('Finetuning last layer from epoch ' + str(start_epoch))
        if not debug:
            model.fit_generator(
                generator=image_generator.flow_from_csv(csv_path_train, ',', True, batch_size, True),
                steps_per_epoch=batches_per_epoch_train,
                epochs=num_epochs_finetune,
                verbose=1,
                callbacks=callbacks_finetune,
                validation_data=image_generator.flow_from_csv(csv_path_validation, ',', True, batch_size, True),
                validation_steps=batches_per_epoch_validation,
                class_weight=class_weights,
                # max_queue_size = batches_per_epoch_train,
                workers=9,
                use_multiprocessing=True,
                initial_epoch=start_epoch)
        else:
            model.fit_generator(
                generator=image_generator.flow_from_csv(csv_path_train, ',', True, batch_size, True, "./debug_images",
                                                        "train"),
                steps_per_epoch=batches_per_epoch_train,
                epochs=num_epochs_finetune,
                verbose=1,
                callbacks=callbacks_finetune,
                validation_data=image_generator.flow_from_csv(csv_path_validation, ',', True, batch_size, True,
                                                              "./debug_images",
                                                              "train"),
                validation_steps=batches_per_epoch_validation,
                class_weight=class_weights,
                # max_queue_size = batches_per_epoch_train,
                workers=1,
                use_multiprocessing=False,
                initial_epoch=start_epoch)
    else:
        print('Finetuning all layers from epoch ' + str(start_epoch))

        # then train all layers at very low learning rate
        for idx, layer in enumerate(model.layers):
            layer.trainable = True

        old_optimizer = optimizers.SGD(lr=old_learning_rate)
        model.compile(optimizer=old_optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

        model.fit_generator(
            generator=image_generator.flow_from_csv(csv_path_train, ',', True, batch_size, True),
            steps_per_epoch=batches_per_epoch_train,
            epochs=num_epochs_fulltune,
            verbose=1,
            callbacks=callbacks_fulltune,
            validation_data=image_generator.flow_from_csv(csv_path_validation, ',', True, batch_size, True),
            validation_steps=batches_per_epoch_validation,
            class_weight=class_weights,
            # max_queue_size = batches_per_epoch_train,
            workers=9,
            use_multiprocessing=True,
            initial_epoch=start_epoch
        )


def get_batches_per_epoch(csv_path, has_header, batch_size):
    num_lines = sum(1 for _ in open(csv_path))
    if has_header and num_lines > 0:
        num_lines -= 1
    return int(round(num_lines / batch_size, 0))


def preprocess_encoded_image(encoded_image):
    """
    :param encoded_image: encoded image string tensor
    :return: tensor representing the image where channel values are scaled between [-1,1]
             and resized to 299x299
    """
    # image = tf.decode_base64(encoded_image)
    image = tf.image.decode_image(encoded_image, channels=3)

    # convert the image values to be between [0,1]
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # the resize function only takes batches of image, so make this
    # a batch of 1 image, and then extract the one image
    batch = tf.expand_dims(image, 0)
    batch = tf.image.resize_bilinear(batch, (299, 299), align_corners=False)

    processed_image = tf.squeeze(batch, [0])

    # clarify that the image has only 3 channels
    # decode_image does not set this in the image shape
    processed_image.set_shape((299, 299, 3))

    # now scale the values between [-1,1]
    processed_image = tf.subtract(processed_image, 0.5)
    processed_image = tf.multiply(processed_image, 2.0)

    return processed_image


def binary_score_to_classification(x, category):
    """
    Convert the binary score to a human-readable label
    :param x: 1-D tensor with score for the positive class
    :param category: string for category name
    :return: yes if the score > 0.5, otherwise no
    """

    return tf.where(x > 0.5, tf.constant([category]), tf.constant(['no_' + category]))


def binary_score_to_class_score(x):
    """
    Convert the binary score to a class score 0-1
    :param x: 1-D tensor with score for the positive class
    :return: score for the predicted class
    """

    return tf.where(x > 0.5, x, 1 - x)


def export_model_to_tensorflow(categories, keras_model_filepath, output_model_dir, has_calibration=True,
                               overwrite_dir=False):
    if overwrite_dir and os.path.isdir(output_model_dir):
        shutil.rmtree(output_model_dir)

    # building the graph for inference
    # need to add preprocessing step as nodes in the graph
    # convert jpeg/png to preprocessed tensor

    # set the learning phase in Keras to False, otherwise the graph will
    # expect this as an input
    K.set_learning_phase(False)

    # the classification api uses serialized examples as inputs
    # the predict api just uses the encoded tensor
    # look in tensorflow_serving/example/inception_saved_model.py for an example of
    # exporting a classification model

    serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    feature_configs = {
        'image/encoded': tf.FixedLenFeature(shape=[], dtype=tf.string)
    }
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)

    encoded_images = tf_example['image/encoded']

    images = tf.map_fn(preprocess_encoded_image, encoded_images, dtype=tf.float32)

    # append this preprocessing step to the graph
    lower_model = build_model(categories, images, has_calibration)
    lower_model.load_weights(keras_model_filepath, True)

    # the new predict signature of the full model should have the encoded image as input
    # and the original output as the output
    # we also need to convert the model output to text labels for classification
    outputs = {}
    classification_labels = None
    classification_scores = None

    for idx, category in enumerate(categories):
        logits = lower_model.output[idx]
        outputs[category] = logits
        labels = tf.map_fn(lambda x: binary_score_to_classification(x, category), logits, dtype=tf.string)
        scores = tf.map_fn(binary_score_to_class_score, logits, dtype=tf.float32)
        if classification_labels is None:
            classification_labels = labels
            classification_scores = scores
        else:
            classification_labels = tf.concat([classification_labels, labels], axis=1)
            classification_scores = tf.concat([classification_scores, scores], axis=1)

    builder = saved_model_builder.SavedModelBuilder(output_model_dir)
    predict_signature = predict_signature_def(inputs={'images': encoded_images}, outputs=outputs)
    classification_signature = classification_signature_def(examples=serialized_tf_example,
                                                            classes=classification_labels,
                                                            scores=classification_scores)

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(sess=sess,
                                             tags=[tag_constants.SERVING],
                                             signature_def_map={
                                                 'predict': predict_signature,
                                                 tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                                     classification_signature}
                                             )
        builder.save()


def test_exported_model(output_model_dir):
    test_file = '/home/ZAZZLE.COM/sharon.lin/Pictures/redfox.jpg'

    with open(test_file, 'rb') as f:
        encoded_image = f.read()

        # if we ever use serialized examples for the classification api:
        test_examples = [
            example_pb2.Example(features=feature_pb2.Features(feature={
                'image/encoded':
                    feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=[encoded_image]))
            }))
        ]
        test_serialized_examples = [example.SerializeToString() for example in test_examples]

        test_images = [encoded_image]

    # reload it and test
    signatures = ['predict', 'serving_default']
    feed = [test_images, test_serialized_examples]

    with tf.Session(graph=tf.Graph()) as sess:
        test_metadata = tf.saved_model.loader.load(sess, [tag_constants.SERVING], output_model_dir)
        for idx, signature in enumerate(signatures):
            test_alias_to_output_info = test_metadata.signature_def[signature].outputs
            test_output_names = [test_alias_to_output_info[alias].name for alias in test_alias_to_output_info]
            test_outputs = {}
            for name in test_output_names:
                test_outputs[name] = tf.get_default_graph().get_tensor_by_name(name)

            test_alias_to_input_info = test_metadata.signature_def[signature].inputs

            test_input_names = [test_alias_to_input_info[alias].name for alias in test_alias_to_input_info]
            test_result = sess.run(test_outputs, feed_dict={test_input_names[0]: feed[idx]})
            print(test_result)


def test_imagenet():
    img = image.load_img('/home/ZAZZLE.COM/sharon.lin/Pictures/cat.jpg')
    img = img.resize((299, 299))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)
    x = preprocess_input(x)

    # load the ImageNet model
    model = InceptionV3(weights='imagenet', include_top=True)

    prediction = model.predict(x, 1, 0)
    index = np.argmax(prediction)
    print(index)


def main_export():
    export_model_to_tensorflow(get_class_names('/storage/Images/All-292/imagetypes-train-keras.txt'),
                               'models/no_jpeg/best_model.hdf5',
                               'models/no_jpeg/tf_category_model',
                               has_calibration=False,
                               overwrite_dir=True)

    test_exported_model('models/no_jpeg/tf_category_model')


def main_train():
    train('/storage/Images/All-292/imagetypes-train-keras.txt',
          '/storage/Images/All-292/imagetypes-validation-keras.txt',
          0.001, 0.0001, "no_jpeg_2", False)
    # print(get_class_weights('/storage/Images/All-292/imagetypes-train-keras.txt'))


def siamese_train():
    # https://hackernoon.com/one-shot-learning-with-siamese-networks-in-pytorch-8ddaab10340e
    train('/storage/Images/All-292/imagetypes-train-keras-siamese.txt',
          '/storage/Images/All-292/imagetypes-validation-keras-siamese.txt',
          0.001, 0.0001, "siamese", False, 2)
    # print(get_class_weights('/storage/Images/All-292/imagetypes-train-keras-siamese.txt'))


def print_calibration_weights(model_path):
    model = load_model(model_path)
    categories = ['text_calibrated', 'photo_calibrated', 'draw_calibrated', 'abstract_calibrated']
    print([model.get_layer(category).get_weights() for category in categories])


def main_calibrate():
    calibrate_model('models/no_jpeg/best_model.hdf5',
                    '/storage/Images/All-292/imagetypes-validation-keras.txt',
                    '/storage/Images/All-292/imagetypes-test-keras.txt',
                    'models/no_jpeg',
                    0.001)


if __name__ == '__main__':
    # Train the model, then pick the "best" model
    # typically it's looking at the validation error as the epoch increases
    # and finding a point where the model is starting to overfit
    # (e.g. validation error is increasing)
    # currently done by hand, as there might be some ups and downs
    # but can explore a heuristic
    # the best model is taken out of the subfolder and renamed 'best_model.hdf5'
    main_train()
    # siamese_train()

    # calibrating the model in this case didn't seem to help
    # it was probably already fairly well calibrated
    # main_calibrate()

    # export and test the model
    # main_export()