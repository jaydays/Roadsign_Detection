#
# Train the product variation grouping model (is group or not) Using Keras
# The model is a siamese model, 2 input images are fed into a pretrained lower model with sigmoid embedding output
# layer(s) appended at the end. The euclidean distance and contrastive loss between the 2 output embeddings is
# calculated to train the lower model. When exporting, only the lower model which outputs the embeddings is exported
import csv
import os
import shutil
import random

import tensorflow as tf
import tensorflow.keras as keras
from keras import optimizers, Input
from keras import backend as K
from keras.models import Sequential
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, Conv2D, MaxPool2D, AveragePooling2D, Flatten, Dropout, BatchNormalization, Lambda
from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.core.example import example_pb2, feature_pb2
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image as pil_image

import image_ops as custom_image

IM_PER_ROW = 2
FINEGRAIN_EMBEDDING_NAME = "finegrain"
COARSEGRAIN_EMBEDDING_NAME = "coarsegrain"


# output methods
def euclidean_distance_train(vects):
    """
    Calculate the euclidean distance between 2 tensors for keras model training
    :param vects: list of tensors to calculate distance between (should be a tuple)
    :return: tensor representing the euclidean distance between 2 tensors
    """
    return euclidean_distance_on_axis(vects, 1)


def euclidean_distance_on_axis(vects, axis):
    """
    Calculate the euclidean distance between 2 tensors on specified axis
    :param vectors: list of tensors to calculate distance between (should be a tuple)
    :param axis: axis to sum squared distances
    :return: tensor representing the euclidean distance between 2 tensors
    """
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=axis, keepdims=True)
    return K.sqrt(K.maximum(sum_square, 0))


def contrastive_loss(y_true, y_pred):
    """
    Contrastive loss from Hadsell-et-al.'06, http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Calculate the contrastive the loss between ground truth and prediction
    :param y_true: ground truth classification tensor for a given training batch
    :param y_pred: output prediction classifications of neural network
    :return: tensor representing the contrastive loss between ground truth and prediction
    """
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    losses = y_true * square_pred + (1 - y_true) * margin_square
    return losses


def accuracy(y_true, y_pred):
    """
    Metric to display in TensorBoard, compute classification accuracy with a fixed threshold on distances.
    :param y_true: ground truth classification tensor for a given training batch
    :param y_pred: calculated euclidean distance tensor
    :return: tensor representing the mean accuracy
    """
    # Determine class of prediction
    # Cast 'y_pred < 0.5' as the y_true numpy object type (float32 in this case)
    # Check if prediction matches truth labels
    # Compute mean of prediction matches
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
# end output methods


def get_class_names():
    # TODO: return the actual sign names
    return list(range(1, 48))


def get_csv_data(csv_path, has_header):
    """
    Get all data from a csv file, not including headers
    :param csv_path: path to csv file
    :param has_header: flag for whether csv has header row, header row is skipped
    :return: A list of tuples, each tuple represents one csv row
    """
    csv_vals = []
    with open(csv_path, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        if has_header:
            # Skip the headers
            next(reader, None)

        for row in reader:
            csv_vals.append(row)
    return csv_vals


def set_model_trainability(model, trainability, ignore_layers=None):
    """
    Recursively iterate through model layers and set trainable flag
    :param model: keras model to modify
    :param trainability: flag for whether model layers will be trainable or not
    :param ignore_layers: list of layer names to ignore
    """
    for layer in model.layers:
        if ignore_layers is not None and layer.name in ignore_layers:
            continue

        if isinstance(layer, Model):
            set_model_trainability(layer, trainability, ignore_layers)
        else:
            layer.trainable = trainability


def build_model():
    numCategories = len(get_class_names())
    learning_rate = 0.001
    input_shape = (32, 32, 3)
    cnn = Sequential()

    # Convolutional Layers
    cnn.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    cnn.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'))
    cnn.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='valid', data_format=None))
    # cnn.add(Dropout(0.5))

    cnn.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    cnn.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='valid', data_format=None))
    # cnn.add(Dropout(0.5))

    cnn.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    cnn.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='valid', data_format=None))
    # cnn.add(Dropout(0.5))

    cnn.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    cnn.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid', data_format=None))
    # cnn.add(Dropout(0.5))

    # Fully Connected
    cnn.add(Flatten())

    # cnn.add(Dense(512, activation='sigmoid'))
    # cnn.add(BatchNormalization())
    # cnn.add(Dropout(0.5))

    cnn.add(Dense(512, activation='sigmoid'))
    # cnn.add(BatchNormalization())
    cnn.add(Dropout(0.5))

    # Output
    cnn.add(Dense(numCategories, activation='softmax'))

    print(cnn.summary())
    adam = Adam(lr=learning_rate)
    cnn.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return cnn


def build_new_siamese_model(output_finegrain, output_coarsegrain, finetune, reference_model_path=None, input_tensor=None):
    """
    Build and compile a new siamese model, copy over reference weights if specified
    :param output_finegrain: flag for if fine grain embedding will be included in output or not
    :param output_coarsegrain: flag for if coarse grain embedding will be included in output or not
    :param finetune: flag for weather non-output layers will be trainable or not
    :param reference_model_path: path to model to copy weights from if specified
    :param input_tensor: input tensor of lower model
    :return: A built and compiled keras siamese model
    """
    # Build lower model with correct outputs
    lower_model = None #build_lower_model(output_finegrain=output_finegrain, output_coarsegrain=output_coarsegrain, input_tensor=input_tensor)

    # Load weights from reference model if specified
    if reference_model_path is not None:
        # Load model and copy weights over
        print('Loading lower model weights from ' + reference_model_path)
        loaded_model = load_model(reference_model_path, custom_objects={'contrastive_loss': contrastive_loss})
        lower_model.set_weights(loaded_model.layers[2].get_weights())
    else:
        print('No model to load, build lower model with new weights')

    print('Build siamese model')
    input_a = Input(shape=(299, 299, 3))
    input_b = Input(shape=(299, 299, 3))
    lower_model_output_a = lower_model(input_a)
    lower_model_output_b = lower_model(input_b)

    # Normalize for 1 or many of outputs
    if not isinstance(lower_model_output_a, list):
        lower_model_output_a = [lower_model_output_a]
        lower_model_output_b = [lower_model_output_b]

    # Create output names
    output_names = list()
    if output_finegrain:
        output_names.append(FINEGRAIN_EMBEDDING_NAME+"_distance")
    if output_coarsegrain:
        output_names.append(COARSEGRAIN_EMBEDDING_NAME+"_distance")
    output_distance_funcs = []

    # Create output layers and metrics
    output_metrics = dict()
    for i in range(0, len(lower_model_output_a)):
        output_distance_funcs.append(
            Lambda(euclidean_distance_train, name=output_names[i])([lower_model_output_a[i], lower_model_output_b[i]]))
        output_metrics[output_names[i]] = accuracy

    # Create siamese model
    model = Model([input_a, input_b], output_distance_funcs)

    # Adjust trainable layers
    if finetune:
        print ('Freeze all model layers except for outputs')
        ignore_layers = []
        if output_coarsegrain:
            ignore_layers.append(COARSEGRAIN_EMBEDDING_NAME)
        if output_finegrain:
            ignore_layers.append(FINEGRAIN_EMBEDDING_NAME)
        set_model_trainability(model, False, ignore_layers)
    else:
        set_model_trainability(model, True)

    # compile model
    rms = optimizers.RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, weighted_metrics=output_metrics)

    return model


def append_coarse_grain_output_to_model(old_model_filepath, new_model_filepath=None):
    """
    Append a coarsegrain embedding output layer to the lower model of a siamese model .This is done by loading the
    old model, creating a new model with the output, then copying weights over from old model to new
    :param old_model_filepath: filepath of the model to append the new layer to
    :param new_model_filepath: filepath to save the new model to
    """
    # Load old model
    old_model = load_model(old_model_filepath, custom_objects={'contrastive_loss': contrastive_loss})
    print("OLD MODEL SUMMARY")
    old_model.summary()
    weights1 = old_model.layers[2].layers[len(old_model.layers[2].layers)-1].get_weights()
    print("weights1")
    print(weights1)

    new_model = build_new_siamese_model(True, True, False, reference_model_path=old_model_filepath)

    print("NEW MODEL SUMMARY")
    new_model.summary()
    weights2 = new_model.layers[2].layers[len(new_model.layers[2].layers)-2].get_weights()
    print("weights2")
    print(weights2)

    # Save model
    if new_model_filepath is None:
        new_model_filepath = old_model_filepath + "-coarse"
    new_model.save(new_model_filepath)

    return new_model


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
        return None, 0

    last_epoch = max(epochs)

    return os.path.join(saved_model_dir, file_names[epochs.index(last_epoch)]), last_epoch


def train(csv_path_train, csv_path_validation, model_dir, num_epochs, output_finegrain, output_coarsegrain,
          finetune=False, initial_model_path=None):
    """
    Build and train a siamese neural network with specified outputs
    :param csv_path: csv filepath that contains annotated data
    :param csv_path_validation: csv filepath that contains validation data
    :param model_dir: directory for outputs of this script
    :param num_epochs: number of epochs to train model
    :param output_finegrain: flag for whether finegrain layer will be an output or not
    :param output_coarsegrain: flag for whether coarsegrain layer will be an output or not
    :param finetune: non-output layers will be frozen if set to true
    :param initial_model_path: filepath of model to start training from, ignored if progress has already been made
    """
    start_mode = 'model'

    # Directories to save models and logs to
    save_dir = os.path.join(model_dir, 'saved_models')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    log_dir = os.path.join(model_dir, start_mode)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    # Get the current state
    model_filename, start_epoch = get_last_checkpoint(save_dir, start_mode)
    if start_epoch >= num_epochs:
        print('Already finished training')
        return

    # Override model with checkpoint if specified
    if start_epoch == 0:
        if initial_model_path is None:
            print("Start training new model from scratch")
        else:
            print("Start training from reference model ", initial_model_path)
            model_filename = initial_model_path

    # Create a new model to train
    model = build_model()

    # Initialize the image generator
    vary_images = True
    print("Vary images from image generator? ", vary_images)
    if vary_images:
        image_generator = custom_image.CustomImageDataGenerator(10, 0.05, 0.05, 0.1, 0.1, 40,
                                                                'nearest', 0, False, False, False,
                                                                preprocess_input, (299, 299), 1,
                                                                im_per_row=IM_PER_ROW, shift_hue=True,
                                                                invert_colours=True)
    else:
        shift_hue = False
        image_generator = custom_image.CustomImageDataGenerator(0, 0, 0, 0, 0, 0,
                                                                'nearest', 0, False, False, False,
                                                                preprocess_input, (299, 299), 1,
                                                                im_per_row=IM_PER_ROW, shift_hue=shift_hue,
                                                                invert_colours=False)

    # Initialize other training params
    batch_size = int(32 / IM_PER_ROW)
    batches_per_epoch_train = get_batches_per_epoch(csv_path_train, True, batch_size)
    batches_per_epoch_validation = get_batches_per_epoch(csv_path_validation, True, batch_size)
    # TODO: sheck this
    num_workers = 9
    use_multiprocessing = True

    save_file_format = os.path.join(save_dir, start_mode + '.{epoch:02d}-{val_loss:.2f}.hdf5')
    logs_callback = [
        keras.callbacks.ModelCheckpoint(save_file_format, period=1),
        keras.callbacks.TensorBoard(log_dir=log_dir)]

    # The number of target instances must equal the number of output layers
    num_target_instances = len(model.outputs)

    # Train the model
    model.fit_generator(
        generator=image_generator.flow_from_csv(csv_path_train, ',', True, batch_size, False,
                                                reweight_labels=True, num_target_instances=num_target_instances),
        steps_per_epoch=batches_per_epoch_train,
        epochs=num_epochs,
        verbose=1,
        callbacks=logs_callback,
        validation_data=image_generator.flow_from_csv(csv_path_validation, ',', True, batch_size, True,
                                                      reweight_labels=True, num_target_instances=num_target_instances),
        validation_steps=batches_per_epoch_validation,
        # max_queue_size=batches_per_epoch_train,
        workers=num_workers,
        use_multiprocessing=use_multiprocessing,
        initial_epoch=start_epoch)


def get_batches_per_epoch(csv_path, has_header, batch_size):
    """
    Calculate the batches per epoch for a given dataset
    :param csv_path: filepath to csv dataset
    :param has_header: flag for whether csv has a header row or not
    :param batch_size: the size of a batch
    :return: the number of batches in one epoch
    """
    num_lines = sum(1 for _ in open(csv_path))
    if has_header and num_lines > 0:
        num_lines -= 1
    return int(round(num_lines / batch_size, 0))


def test_keras_model(model, csv_path_test, incorrect_pred_dir, vary_images=False, save_im_dir=None):
    """
    Test a keras model and print incorrect predictions to files ina  directory
    :param model: model to test
    :param csv_path_test: filepath for csv with test data
    :param incorrect_pred_file: output filepath to print out incorrect predictions
    :param vary_images: flag for wether test images will be randomly transformed or not
    :param save_im_dir: filepath to save test images to
    """
    batch_size = int(32 / IM_PER_ROW)
    if vary_images:
        image_generator = custom_image.CustomImageDataGenerator(10, 0.05, 0.05, 0.1, 0.1, 40,
                                                                'nearest', 0, False, False, False,
                                                                preprocess_input, (299, 299), 1,
                                                                im_per_row=IM_PER_ROW, shift_hue=True,
                                                                invert_colours=True)
    else:
        image_generator = custom_image.CustomImageDataGenerator(0, 0, 0, 0, 0, 0,
                                                                'nearest', 0, False, False, False,
                                                                preprocess_input, (299, 299), 1,
                                                                im_per_row=IM_PER_ROW, shift_hue=False,
                                                                invert_colours=False)

    predictions = model.predict_generator(
        generator=image_generator.flow_from_csv(csv_path_test, ',', True, batch_size, False, reweight_labels=False,
                                                save_to_dir= save_im_dir,
                                                save_prefix='save', save_format='jpg'),
        workers=9,
        use_multiprocessing=True,
        verbose=1)

    # Normalize predictions data structure for different number of outputs
    if len(model.outputs) > 1:
        prediction_sets = predictions
    else:
        prediction_sets = [predictions]

    distance_output_sets = list()
    for prediction_set in prediction_sets:
        distance_set = []
        for prediction in prediction_set:
            euc_distance = prediction[0]
            distance_set.append(euc_distance)

        distance_output_sets.append(distance_set)

    print_test_result(csv_path_test, incorrect_pred_dir, distance_output_sets)


def print_test_result(csv_path_test, incorrect_pred_dir, distance_output_sets):
    """
    Print result of test run to directory
    :param model: model to test
    :param csv_path_test: filepath for csv with test data
    :param incorrect_pred_file: output filepath to print out incorrect predictions
    :param distance_output_sets: sets of distances for each output of the model
    """
    # Get ground truth data
    csv_vals = get_csv_data(csv_path_test, True)
    y_true = []
    for row in csv_vals:
        y_true.append(int(row[IM_PER_ROW]))

    # Format of confusion matrix (outputed later)
    # print('TP FN')
    # print('FP TN')

    # Create pred directory if it doesn't exist
    if not os.path.exists(incorrect_pred_dir):
        os.mkdir(incorrect_pred_dir)

    # Iterate through prediction sets and calculate result stats
    for set_index in range(0, len(distance_output_sets)):
        distance_output_set = distance_output_sets[set_index]
        if len(csv_vals) != len(distance_output_set):
            print("Error: Test data and test prediction different size")
            return

        # Distances and grouping predictions
        y_pred = []
        for euc_distance in distance_output_set:
            y_pred.append(int(euc_distance < 0.5))

        if len(y_true) != len(y_pred):
            print("Error: prediction and test data not the same length")
            return

        print('Confusion Matrix ', set_index)
        # TP FN
        # FP TN
        labels = [1, 0]
        print(confusion_matrix(y_true, y_pred, labels))
        print('Classification Report ', set_index)
        target_names = ['Is Group', 'Not Group']
        print(classification_report(y_true, y_pred, labels=labels, target_names=target_names))

        # print incorrect predictions to file
        curr_pred_file = os.path.join(incorrect_pred_dir, "out_{iter}.txt".format(iter=set_index))
        out_file = open(curr_pred_file, "w")
        line_format = "{0:49} {1:53} {2:3} {3:4} {4:8}\n"
        out_file.write(line_format.format("img1", "img2", "act", "pred", "euc_dist"))
        for j in range(0, len(y_true)):
            if y_true[j] != y_pred[j]:
                img1 = csv_vals[j][0]
                img2 = csv_vals[j][1]
                out_file.write(line_format.format(img1, img2, str(y_true[j]), str(y_pred[j]), str(distance_output_set[j])))
        out_file.close()


def load_and_test_keras_model(model_filepath, output_finegrain, output_coarsegrain, csv_path_test, incorrect_pred_dir,
                              vary_images=False, save_im_dir=None):
    """
    Build new siames model with specified outputs and test it using saved model weights
    :param model_filepath: filepath of saved model
    :param output_finegrain: flag for whether finegrain layer will be an output or not
    :param output_coarsegrain: flag for whether coarsegrain layer will be an output or not
    :param csv_path_test: filepath for csv with test data
    :param incorrect_pred_file: output filepath to print out incorrect predictions
    :param vary_images: flag for wether test images will be randomly transformed or not
    :param save_im_dir: filepath to save test images to
    """
    model = build_new_siamese_model(output_finegrain, output_coarsegrain, False, reference_model_path=model_filepath)
    test_keras_model(model, csv_path_test, incorrect_pred_dir, vary_images, save_im_dir)


def test_keras_model_as_saved(model_filepath, csv_path_test, incorrect_pred_dir, vary_images=False, save_im_dir=None):
    """
    Load model from file and test it as is (do not change outputs), print incorrect predictions to a file
    :param model_filepath: filepath of saved model
    :param csv_path_test: filepath for csv with test data
    :param incorrect_pred_file: output filepath to print out incorrect predictions
    :param vary_images: flag for wether test images will be randomly transformed or not
    :param save_im_dir: filepath to save test images to
    """
    model = load_model(model_filepath, custom_objects={'contrastive_loss': contrastive_loss})
    rms = optimizers.RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, weighted_metrics=[accuracy])
    test_keras_model(model, csv_path_test, incorrect_pred_dir, vary_images, save_im_dir)


def preprocess_encoded_image(encoded_image):
    """
    Preprocess image data for use as input into neural network
    :param encoded_image: encoded image string tensor
    :return: tensor representing the image where channel values are scaled between [-1,1] and resized to 299x299
    """

    # image_data = tf.decode_base64(encoded_image)
    image_data = tf.image.decode_jpeg(encoded_image, channels=3, dct_method='INTEGER_ACCURATE')

    # convert the image values to be between [0,1]
    image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
    batch = tf.expand_dims(image_data, 0)
    batch = tf.image.resize_nearest_neighbor(batch, (299, 299), align_corners=False)
    processed_image = tf.squeeze(batch, [0])

    # clarify that the image has only 3 channels
    # decode_image does not set this in the image shape
    processed_image.set_shape((299, 299, 3))

    # now scale the values between [-1,1]
    processed_image = tf.subtract(processed_image, 0.5)
    processed_image = tf.multiply(processed_image, 2.0)

    return processed_image


def preprocess_encoded_image_accurate(encoded_image):
    """
    Preprocess image data for use as input into neural network
    This is method matches the keras preprocessing method more accurately, but cannot be used as an input tensor when
    exporting the model
    :param encoded_image: encoded image string tensor
    :return: tensor representing the image where channel values are scaled between [-1,1] and resized to 299x299
    """
    image_data = tf.image.decode_jpeg(encoded_image, channels=3, dct_method='INTEGER_ACCURATE')

    image_data = tf.cast(image_data, dtype=tf.float32)

    with tf.Session() as sess:
        x = image_data.eval(session=sess)
        img = image.array_to_img(x, scale=False)
        width_height_tuple = (299, 299)
        if img.size != width_height_tuple:
            resample = pil_image.NEAREST
            img = img.resize(width_height_tuple, resample)
        img_array = image.img_to_array(img)
        image_data = tf.convert_to_tensor(img_array, dtype=tf.float32)

    return image_data


def export_model_to_tensorflow(keras_model_filepath, output_model_dir, output_finegrain, output_coarsegrain,
                               overwrite_dir=True):
    """
    Export keras model into tensorflow format and save it
    :param keras_model_filepath: filepath where keras model is saved
    :param output_model_dir: output directory of tensorflow model data
    :param output_finegrain: flag for whether finegrain layer will be an output or not
    :param output_coarsegrain: flag for whether coarsegrain layer will be an output or not
    :param overwrite_dir: flag for whether output_model_directory can be overridden or not
    :return: The keras model that was just exported
    """
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

    # Load model with new input function,
    keras_model = build_new_siamese_model(output_finegrain, output_coarsegrain, False,
                                          reference_model_path=keras_model_filepath, input_tensor=images)

    # extract lower model to export
    lower_model = keras_model.layers[2]

    # Output names
    output_names = []
    if output_finegrain:
        output_names.append(FINEGRAIN_EMBEDDING_NAME)
    if output_coarsegrain:
        output_names.append(COARSEGRAIN_EMBEDDING_NAME)

    # the new predict signature of the full model should have the encoded image as input
    # and the original output as the output
    # we also need to convert the model output to text labels for classification
    outputs = dict()
    for i in range(0, len(lower_model.output)):
        output_name = output_names[i] + '_output'
        outputs[output_name] = lower_model.output[i]

    builder = saved_model_builder.SavedModelBuilder(output_model_dir)
    predict_signature = predict_signature_def(inputs={'images': encoded_images}, outputs=outputs)
    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(sess=sess,
                                             tags=[tag_constants.SERVING],
                                             signature_def_map={
                                                 'predict': predict_signature}
                                             )
        builder.save()

    return keras_model


def test_exported_model(output_model_dir, images_dir, csv_path_test, incorrect_pred_dir):
    """
    Test an exported keras model, print incorrect predistions to file
    :param output_model_dir: directory where exported model is saved
    :param images_fir: directory where images from csv_path_test are saved
    :param csv_path_test: filepath for csv with test data
    :param incorrect_pred_file: output filepath to print out incorrect predictions
    """
    with tf.Session(graph=tf.Graph()) as sess:
        loaded_model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], output_model_dir)
        distance_output_sets = list()
        csv_vals = get_csv_data(csv_path_test, True)
        for row in csv_vals:
            imageFilepathA = os.path.join(images_dir, row[0])
            imageFilepathB = os.path.join(images_dir, row[1])
            distance_outputs = get_distances_from_exported_model(sess, loaded_model, imageFilepathA, imageFilepathB)
            for i in range(0, len(distance_outputs)):
                if i >= len(distance_output_sets):
                    distance_output_sets.append(list())
                distance_output_sets[i].append(distance_outputs[i])

    print_test_result(csv_path_test, incorrect_pred_dir, distance_output_sets)


def debug_exported_model(output_model_dir):
    """
    Test the output of an exported tensorflow model
    :param output_model_dir: output directory of tensorflow model data
    """
    test_file1 = "/storage/Images/prod_var_groupings/dataset/207123-256565611889276751-Done.jpg"
    test_file2 = "/storage/Images/prod_var_groupings/dataset/207123-256443297751993051-Done.jpg"
    with tf.Session(graph=tf.Graph()) as sess:
        loaded_model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], output_model_dir)
        get_distances_from_exported_model(sess, loaded_model, test_file1, test_file2, debug=True)


def get_distances_from_exported_model(sess, loaded_model, image_file1, image_file2, debug=False):
    """
    Get euclidean distances between 2 images using exported model for each output of model
    :param sess: current tensorflow session
    :param loaded_model: exported model that was loaded by sess
    :param image_file1: filepath to first image
    :param image_file2: filepath to second image
    :param debug: flag for whether debug info will be printed to console or not
    :return: Euclidean distances for each output of the model
    """
    test_files = list()
    test_files.append(image_file1)
    test_files.append(image_file2)

    result_vector_pairs = [[]]
    # reload it and test
    signatures = ['predict']
    for test_file in test_files:
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
        feed = [test_images, test_serialized_examples]

        for idx, signature in enumerate(signatures):
            test_alias_to_output_info = loaded_model.signature_def[signature].outputs
            test_output_names = [test_alias_to_output_info[alias].name for alias in test_alias_to_output_info]
            test_outputs = {}
            for name in test_output_names:
                test_outputs[name] = tf.get_default_graph().get_tensor_by_name(name)

            test_alias_to_input_info = loaded_model.signature_def[signature].inputs

            test_input_names = [test_alias_to_input_info[alias].name for alias in test_alias_to_input_info]
            test_result = sess.run(test_outputs, feed_dict={test_input_names[0]: feed[idx]})

            output_iter = 0
            for key, test_result_item in test_result.items():
                if output_iter > len(result_vector_pairs) - 1:
                    result_vector_pairs.append(list())
                vector = test_result_item[0]
                result_vector_pairs[output_iter].append(vector)
                output_iter += 1

    result_vector_pairs.sort(key=lambda rvp: len(rvp[0]), reverse=True)
    distances = list()
    for vector_pair in result_vector_pairs:
        if debug:
            print("Vector Pair")
            print(vector_pair)
        dist = euclidean_distance_on_axis(vector_pair, 0)
        with tf.Session():
            distance = dist.eval()[0]
            distances.append(distance)
            if debug:
                print("Distance")
                print(distance)

    return distances


def get_first_file_with_prefix(directory, prefix):
    """
    Get the first filepath that starts with a specified prefix from a directory
    :param directory: directory to search
    :param prefix: prefix of file
    :return: First file in directory that starts with prefix
    """
    file_paths = get_files_with_prefix(directory, prefix)
    return None if len(file_paths) == 0 else file_paths[0]


def get_files_with_prefix(directory, prefix):
    """
    Get all filepaths that starts with a specified prefix from a directory
    :param directory: directory to search
    :param prefix: prefix of file
    :return: List of files in directory that starts with prefix
    """
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory)
                  if os.path.isfile(os.path.join(directory, f)) and f.startswith(prefix)]
    return file_paths


def shuffle_split_file(input_path, output_path1, output_path2, split, has_header):
    if os.path.isfile(output_path1):
        os.remove(output_path1)
    if os.path.isfile(output_path2):
        os.remove(output_path2)

    with open(input_path, 'r') as f:
        lines = f.readlines()

    if has_header:
        lines.pop(0)

    random.seed(4)
    random.shuffle(lines)
    split_index = int(len(lines)*split)
    with open(output_path1, 'w') as f:
        f.writelines(lines[:split_index])
    with open(output_path2, 'w') as f:
        f.writelines(lines[split_index:])


def main():
    # Modify model by appending coarse grain output (do once only)
    # append_coarse_grain_output_to_lower_model(saved_model)

    # Train the model, then pick the "best" model
    # typically it's looking at the validation error as the epoch increases
    # and finding a point where the model is starting to overfit
    # (e.g. validation error is increasing)
    # currently done by hand, as there might be some ups and downs
    # but can explore a heuristic
    # the best model is taken out of the subfolder and renamed 'best_model.hdf5'

    project_dir = os.path.abspath(os.getcwd())

    # Created model directories
    models_dir = os.path.join(project_dir, 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    model_label = 'sign_classifier'
    model_dir = os.path.join(models_dir, model_label)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # Database directories/paths
    database_dir = os.path.join(project_dir, 'Database')
    if not os.path.exists(database_dir):
        raise ValueError('Database not found.')

    images_dir = os.path.join(database_dir, 'Data')
    if not os.path.exists(images_dir):
        raise ValueError('Images directory not found.')

    annotations_csv = os.path.join(database_dir, 'annotations.csv')
    if not os.path.exists(annotations_csv):
        raise ValueError('Annotations not found.')

    # Split annotations into training/validation
    training_csv = os.path.join(os.path.dirname(annotations_csv), 'training.csv')
    validation_csv = os.path.join(os.path.dirname(annotations_csv), 'validation.csv')
    training_val_split = 0.8
    shuffle_split_file(annotations_csv, training_csv, validation_csv, training_val_split, has_header=True)

    # General parameters
    saved_models_dir = os.path.join(model_dir, 'saved_models')
    saved_model = None
    print(saved_model)
    output_finegrain = True
    output_coarsegrain = True
    finetune = True

    # Train model
    num_epochs = 6
    train(training_csv, validation_csv, model_dir, num_epochs, output_finegrain, output_coarsegrain,
          finetune=finetune, initial_model_path=saved_model)

    # Test model
    csv_path_test = '/storage/Images/prod_var_groupings/dataset/test-even.txt'
    incorrect_pred_dir = os.path.join(model_dir, 'incorrect_predictions')
    vary_images = False
    save_im_dir = None
    #test_keras_model_as_saved(saved_model, csv_path_test, incorrect_pred_dir, vary_images, save_im_dir)
    #load_and_test_keras_model(saved_model, output_finegrain, output_coarsegrain, csv_path_test, incorrect_pred_dir, vary_images, save_im_dir)

    # Export
    export_dir = os.path.join(model_dir, 'exported_model')
    exported_model = export_model_to_tensorflow(saved_model, export_dir, output_finegrain, output_coarsegrain)
    # Test exported model
    images_dir = '/storage/Images/prod_var_groupings/dataset'
    debug_exported_model(export_dir)

    # epochs = [10]
    # for epoch in epochs:
    #     file_prefix = 'coarsegrain_finetune.' + format(epoch, "02")
    #     file_paths = get_files_with_prefix(saved_models_dir, file_prefix)
    #     if len(file_paths) > 0:
    #         print("!!!! NEW FILE TEST: ", file_prefix)
    #         load_and_test_keras_model(file_paths[0], csv_path_test, incorrect_pred_dir, vary_images=False, save_im_dir=None)

if __name__ == '__main__':
    main()
