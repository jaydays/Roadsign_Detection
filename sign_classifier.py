#
# Train the product variation grouping model (is group or not) Using Keras
# The model is a siamese model, 2 input images are fed into a pretrained lower model with sigmoid embedding output
# layer(s) appended at the end. The euclidean distance and contrastive loss between the 2 output embeddings is
# calculated to train the lower model. When exporting, only the lower model which outputs the embeddings is exported
import csv
import os
import random


import tensorflow.keras as keras

from tensorflow.python.keras import models
from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.applications.inception_v3 import preprocess_input
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, AveragePooling2D, Flatten, Dropout, BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix

import image_ops as custom_image
import numpy as np

IM_PER_ROW = 1  # TODO: get rid of this
IM_SIZE = (96, 96)
INPUT_SHAPE = (96, 96, 3)
UNKNOWN_CLASS_NUM = -1


def build_model():
    numCategories = len(get_class_names())
    learning_rate = 0.001
    cnn = Sequential()

    # Convolutional Layers
    filters = 16
    cnn.add(Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu', input_shape=INPUT_SHAPE))
    # cnn.add(Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu'))
    # cnn.add(BatchNormalization())
    cnn.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='valid', data_format=None))
    # cnn.add(Dropout(0.5))

    filters = 32
    # cnn.add(Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu'))
    cnn.add(Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu'))
    # cnn.add(BatchNormalization())
    cnn.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='valid', data_format=None))
    # cnn.add(Dropout(0.5))

    # filters = 64
    # cnn.add(Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu'))
    # cnn.add(Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu'))
    # cnn.add(BatchNormalization())
    # cnn.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='valid', data_format=None))
    # cnn.add(Dropout(0.5))

    # filters = 128
    # cnn.add(Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu'))
    # cnn.add(Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu'))
    # cnn.add(BatchNormalization())
    # cnn.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid', data_format=None))
    # cnn.add(Dropout(0.5))

    # Fully Connected
    cnn.add(Flatten())

    # cnn.add(Dense(512, activation='sigmoid'))
    # cnn.add(BatchNormalization())
    # cnn.add(Dropout(0.5))

    cnn.add(Dense(512, activation='sigmoid'))
    # cnn.add(BatchNormalization())
    # cnn.add(Dropout(0.5))

    # Output
    cnn.add(Dense(numCategories, activation='softmax'))

    print(cnn.summary())

    compile_model(cnn, learning_rate)
    return cnn


def compile_model(model, learning_rate=0.001):
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


def load_model(model_filepath):
    model = models.load_model(model_filepath)
    compile_model(model)
    return model


def get_class_names():
    class_names = ['addedLane', 'curveLeft', 'curveRight', 'dip', 'doNotEnter', 'doNotPass', 'intersection',
                   'keepRight', 'laneEnds',
                   'merge', 'noLeftTurn', 'noRightTurn', 'pedestrianCrossing', 'rampSpeedAdvisory20',
                   'rampSpeedAdvisory35', 'rampSpeedAdvisory40', 'rampSpeedAdvisory45', 'rampSpeedAdvisory50',
                   'rampSpeedAdvisoryUrdbl', 'rightLaneMustTurn', 'roundabout', 'school', 'schoolSpeedLimit25',
                   'signalAhead', 'slow', 'speedLimit15', 'speedLimit25', 'speedLimit30', 'speedLimit35',
                   'speedLimit40', 'speedLimit45', 'speedLimit50', 'speedLimit55', 'speedLimit65', 'truckSpeedLimit55',
                   'speedLimit15', 'speedLimit25', 'speedLimit30', 'speedLimit35', 'speedLimit40', 'speedLimit45',
                   'speedLimit50', 'speedLimit55', 'speedLimit65', 'speedLimitUrdbl', 'speedLimit15', 'speedLimit25',
                   'speedLimit30', 'speedLimit35', 'speedLimit40', 'speedLimit45', 'speedLimit50', 'speedLimit55',
                   'speedLimit65', 'speedLimitUrdbl', 'stop', 'stopAhead', 'thruMergeLeft', 'thruMergeRight',
                   'thruTrafficMergeLeft', 'truckSpeedLimit55', 'turnLeft', 'turnRight', 'yield', 'yieldAhead',
                   'zoneAhead25', 'zoneAhead45']
    return class_names


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


def create_training_validation_csv(input_csv, training_csv, validation_csv, training_split, has_header):
    if os.path.isfile(training_csv):
        os.remove(training_csv)
    if os.path.isfile(validation_csv):
        os.remove(validation_csv)

    with open(input_csv, 'r') as f:
        lines = f.readlines()

    header = None
    if has_header:
        header = lines.pop(0)

    random.seed(4)
    random.shuffle(lines)
    split_index = int(len(lines)*training_split)

    training = lines[:split_index]
    validation = lines[split_index:]
    if header is not None:
        training.insert(0, header)
        validation.insert(0, header)

    with open(training_csv, 'w') as f:
        f.writelines(training)
    with open(validation_csv, 'w') as f:
        f.writelines(validation)


def get_data_from_csv(csv_path, image_root_dir, delimiter=';', has_header=True):
    class_names = get_class_names()
    num_classes = len(class_names)

    image_paths = []
    image_labels = []
    label_bounds = []
    with open(csv_path, newline='') as csv_file:
        # get the directory path of the csv file
        csv_reader = csv.reader(csv_file, delimiter=delimiter)

        first_read = True
        for row in csv_reader:
            if has_header and first_read:
                first_read = False
                continue
            if len(row) == 0:
                continue
            image_paths.append(os.path.join(image_root_dir, row[0]))

            sign_class = row[1]
            class_index = class_names.index(sign_class)
            sign_one_hot = class_num_to_one_hot(class_index, num_classes)
            image_labels.append(sign_one_hot)

            bounds = list(map(int, row[2:6]))
            label_bounds.append(bounds)

    return image_paths, image_labels, label_bounds


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


def train(csv_path_train, csv_path_validation, images_dir, model_dir, model_name, num_epochs, initial_model_path=None):
    """
    Build and train a siamese neural network with specified outputs
    :param csv_path: csv filepath that contains annotated data
    :param csv_path_validation: csv filepath that contains validation data
    :param model_dir: directory for outputs of this script
    :param model_name: name to save the model information to
    :param num_epochs: number of epochs to train model
    :param output_finegrain: flag for whether finegrain layer will be an output or not
    :param output_coarsegrain: flag for whether coarsegrain layer will be an output or not
    :param finetune: non-output layers will be frozen if set to true
    :param initial_model_path: filepath of model to start training from, ignored if progress has already been made
    """

    # Directories to save models and logs to
    save_dir = os.path.join(model_dir, 'saved_models')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    log_dir = os.path.join(model_dir, 'logs')
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    # Get the current state
    current_model_path, start_epoch = get_last_checkpoint(save_dir, model_name)

    # Override model with checkpoint if specified
    if current_model_path is not None:
        if initial_model_path is None:
            initial_model_path = current_model_path
        else:
            print("Overriding current model with model at ", initial_model_path)

    if initial_model_path is None:
        print("Build new model from scratch")
        model = build_model()
    else:
        print("Load model at ", initial_model_path)
        model = load_model(initial_model_path)

    if start_epoch >= num_epochs:
        print('Already finished training')
        return model

    # Initialize the image generator
    vary_images = False
    print("Vary images from image generator? ", vary_images)
    if vary_images:
        image_generator = custom_image.CustomImageDataGenerator(10, 0.05, 0.05, 0.1, 0.1, 40,
                                                                'nearest', 0, False, False, False,
                                                                preprocess_input, IM_SIZE, 1,
                                                                shift_hue=True, invert_colours=True)
    else:
        shift_hue = False
        image_generator = custom_image.CustomImageDataGenerator(0, 0, 0, 0, 0, 0,
                                                                'nearest', 0, False, False, False,
                                                                preprocess_input, IM_SIZE, 1,
                                                                shift_hue=shift_hue, invert_colours=False)

    # Initialize other training params
    batch_size = 32
    batches_per_epoch_train = get_batches_per_epoch(csv_path_train, True, batch_size)
    batches_per_epoch_validation = get_batches_per_epoch(csv_path_validation, True, batch_size)
    # TODO: check this
    num_workers = 9
    use_multiprocessing = False

    save_file_format = os.path.join(save_dir, model_name + '.{epoch:02d}-{val_loss:.2f}.hdf5')
    logs_callback = [
        #keras.callbacks.ModelCheckpoint(save_file_format, period=1),
        keras.callbacks.ModelCheckpoint(save_file_format, save_freq='epoch'),
        keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1)]

    # The number of target instances must equal the number of output layers
    num_target_instances = len(model.outputs)

    paths_train, labels_train, train_bounds = get_data_from_csv(csv_path_train, images_dir)
    paths_val, labels_val, val_bounds = get_data_from_csv(csv_path_train, images_dir)

    # Train the model
    model.fit_generator(
        generator=image_generator.flow_from_iterator(paths_train, labels_train, None, batch_size, False,
                                                     reweight_labels=True, save_to_dir=None,#model_dir,
                                                     num_target_instances=num_target_instances,
                                                     label_bounds=train_bounds),
        steps_per_epoch=batches_per_epoch_train,
        epochs=num_epochs,
        verbose=1,
        callbacks=logs_callback,
        validation_data=image_generator.flow_from_iterator(paths_val, labels_val, None, batch_size, False,
                                                           reweight_labels=True, save_to_dir=None,
                                                           num_target_instances=num_target_instances,
                                                           label_bounds=val_bounds),
        validation_steps=batches_per_epoch_validation,
        # max_queue_size=batches_per_epoch_train,
        workers=num_workers,
        use_multiprocessing=use_multiprocessing,
        initial_epoch=start_epoch)
    return model


def test_keras_model(model, csv_path_test, images_dir, incorrect_pred_dir, vary_images=False, save_im_dir=None):
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
                                                                preprocess_input, IM_SIZE, 1,
                                                                im_per_row=IM_PER_ROW, shift_hue=True,
                                                                invert_colours=True)
    else:
        image_generator = custom_image.CustomImageDataGenerator(0, 0, 0, 0, 0, 0,
                                                                'nearest', 0, False, False, False,
                                                                preprocess_input, IM_SIZE, 1,
                                                                im_per_row=IM_PER_ROW, shift_hue=False,
                                                                invert_colours=False)

    paths_test, labels_test, bounds_test = get_data_from_csv(csv_path_test, images_dir)
    predictions = model.predict_generator(
        generator=image_generator.flow_from_iterator(paths_test, labels_test, None, batch_size, False,
                                                     reweight_labels=False, save_to_dir=None,
                                                     num_target_instances=1, label_bounds=bounds_test),
        workers=9,
        use_multiprocessing=False,
        verbose=1)

    y_true = [one_hot_to_class_num(label) for label in labels_test]
    y_pred = [one_hot_to_class_num(prediction) for prediction in predictions]

    print_test_result(incorrect_pred_dir, paths_test, y_true, y_pred)


def print_test_result(incorrect_pred_dir, x, y_true, y_pred):
    """
    Print result of test run to directory
    :param model: model to test
    :param csv_path_test: filepath for csv with test data
    :param incorrect_pred_file: output filepath to print out incorrect predictions
    :param distance_output_sets: sets of distances for each output of the model
    """
    # Format of confusion matrix (outputed later)
    # print('TP FN')
    # print('FP TN')

    print('Confusion Matrix ')
    labels = None#[1, 0] #TODO: proper labels and names
    print(confusion_matrix(y_true, y_pred, labels))

    target_names = None #get_class_names()
    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names))

    # print incorrect predictions to file
    incorrect_predictions_file = os.path.join(incorrect_pred_dir, "incorrect.txt")
    out_file = open(incorrect_predictions_file, "w")
    line_format = "{0:49} {1:5} {2:5}\n"
    out_file.write(line_format.format("img1", "act", "pred"))
    for i in range(0, len(y_true)):
        if y_true[i] != y_pred[i]:
            img = x[i]
            out_file.write(line_format.format(os.path.basename(img), str(y_true[i]), str(y_pred[i])))
    out_file.close()


def class_num_to_one_hot(class_num, num_classes):
    one_hot = np.zeros(num_classes)
    one_hot[class_num] = 1
    return one_hot


def one_hot_to_class_num(one_hot):
    max_val = max(one_hot)
    max_indexes = np.where(one_hot == max_val)[0]
    if len(max_indexes) > 1:
        return UNKNOWN_CLASS_NUM
    return max_indexes[0]


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

    model_name = 'classifier_base'

    project_dir = os.path.abspath(os.getcwd())

    # Created model directories
    models_dir = os.path.join(project_dir, 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    #model_label = 'sign_classifier'
    model_dir = os.path.join(models_dir, model_name)
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
    create_training_validation_csv(annotations_csv, training_csv, validation_csv, training_val_split, has_header=True)

    # General parameters
    saved_model = None
    if saved_model is not None:
        print("Model: ", saved_model)

    # Train model
    num_epochs = 20
    model = train(training_csv, validation_csv, images_dir, model_dir, model_name, num_epochs, initial_model_path=saved_model)

    # Test model
    incorrect_pred_dir = os.path.join(model_dir, 'incorrect_predictions')
    if not os.path.exists(incorrect_pred_dir):
        os.mkdir(incorrect_pred_dir)
    incorrect_images_dir = os.path.join(incorrect_pred_dir, 'images')
    if not os.path.exists(incorrect_images_dir):
        os.mkdir(incorrect_images_dir)
    csv_path_test = validation_csv
    vary_images = False
    test_keras_model(model, csv_path_test, images_dir, incorrect_pred_dir, vary_images, None)


if __name__ == '__main__':
    main()
