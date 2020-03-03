import tensorflow as tf
import tensorflow.contrib.keras as keras
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.contrib.keras import optimizers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import Iterator, ImageDataGenerator
import csv
import os.path
import numpy as np
from tensorflow.python.keras import backend as K
from scipy import linalg
import scipy.ndimage as ndi
from io import BytesIO
from PIL import Image as pil_image
import PIL.ImageOps
from enum import Enum
from ast import literal_eval
from math import sqrt, cos, sin, radians


class DataType(Enum):
    BINARY = 1
    ONE_HOT_CATEGORICAL = 2
    SOFT_CATEGORICAL = 3


class CustomImageDataGenerator(ImageDataGenerator):
    def __init__(self,
                 rotation_range,
                 width_shift_range,
                 height_shift_range,
                 shear_range,
                 zoom_range,
                 channel_shift_range,
                 fill_mode,
                 cval,
                 horizontal_flip,
                 vertical_flip,
                 jpeg_compression,
                 preprocessing_function,
                 image_size,
                 seed,
                 #siamese networks would have 2 images per row
                 im_per_row=1,
                 make_grayscale=True,
                 shift_hue=False,
                 invert_colours=False):
        super().__init__(zca_epsilon=0,
                         rotation_range=rotation_range,
                         width_shift_range=width_shift_range,
                         height_shift_range=height_shift_range,
                         shear_range=shear_range,
                         zoom_range=zoom_range,
                         channel_shift_range=channel_shift_range,
                         fill_mode=fill_mode,
                         cval=cval,
                         horizontal_flip=horizontal_flip,
                         vertical_flip=vertical_flip,
                         preprocessing_function=preprocessing_function,
                         data_format='channels_last')

        self.jpeg_compression = jpeg_compression
        self.seed = seed
        self.image_size = image_size
        self.im_per_row = im_per_row
        self.make_grayscale = make_grayscale
        self.shift_hue = shift_hue
        self.invert_colours = invert_colours

    def get_random_transform_params(self, inp, seed=None):
        random_transform_params = super().get_random_transform(inp.shape, seed)
        # make_grayscale = np.random.random() > 0.95
        # jpeg_compression = np.random.random() > 0.5
        # jpeg_quality =  np.random.randint(80, 100)
        # random_transform_params['make_grayscale'] = make_grayscale
        # random_transform_params['jpeg_compression'] = jpeg_compression
        # random_transform_params['jpeg_quality'] = jpeg_quality
        return random_transform_params

    def apply_transform(self, inp, override_params = None):
        transform_params = self.get_random_transform_params(inp)

        if override_params is not None:
            for key in override_params:
                transform_params[key] = override_params[key]

        result = super().apply_transform(inp, transform_params)

        greyscale = self.make_grayscale and len(result.shape) == 3 and np.random.random() <= 0.25
        if greyscale:
            # convert to grayscale
            data_format = K.image_data_format()
            if data_format == 'channels_last':
                gray = np.tensordot(result, [0.3, 0.59, 0.11], axes=(-1, 0))
                result = np.stack((gray,) * 3, axis=-1)
            else:
                gray = np.tensordot(result, [0.3, 0.59, 0.11], axes=(0, 0))
                result = np.stack((gray,) * 3)

        if self.jpeg_compression and np.random.random() > 0.5:
            # add jpeg compression
            random_quality = np.random.randint(80, 100)

            img = image.array_to_img(result)
            buffer = BytesIO()
            img.save(buffer, "JPEG", quality=random_quality)
            img = pil_image.open(buffer)

            result = image.img_to_array(img)
            buffer.close()

        if not greyscale and self.shift_hue and np.random.random()  <= 0.1:
            result = self.rotate_hue(result, np.random.randint(0, 359))

        if self.invert_colours and np.random.random()  <= 0.1:
            # invert image colours
            img = image.array_to_img(result)
            img = PIL.ImageOps.invert(img)
            result = image.img_to_array(img)

        return result


    def rotate_hue(self, img, degrees):
        # This is ~400x faster than doing it with pseudocode!
        if degrees == 0: return img
        rot_matrix = self.get_hue_rotation_matrix(degrees)

        im_length = len(img)
        im_width = len(img[0])
        num_pixels = im_length * im_width
        pixel_array = img.reshape(num_pixels, 3)
        # Transpose to get arrays by colour instead of by pixel
        rgb_arrays = pixel_array.T
        # Apply rotation to each pixel using matrix multiplication
        rot_rgb_arrays = np.clip(np.matmul(rot_matrix, rgb_arrays), 0, 255)

        rot_pixel_array = rot_rgb_arrays.T
        rot_image = rot_pixel_array.reshape(im_length, im_width, 3)
        return rot_image


    def get_hue_rotation_matrix(self, degrees):
        matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        cosA = cos(radians(degrees))
        sinA = sin(radians(degrees))
        matrix[0][0] = cosA + (1.0 - cosA) / 3.0
        matrix[0][1] = 1. / 3. * (1.0 - cosA) - sqrt(1. / 3.) * sinA
        matrix[0][2] = 1. / 3. * (1.0 - cosA) + sqrt(1. / 3.) * sinA
        matrix[1][0] = 1. / 3. * (1.0 - cosA) + sqrt(1. / 3.) * sinA
        matrix[1][1] = cosA + 1. / 3. * (1.0 - cosA)
        matrix[1][2] = 1. / 3. * (1.0 - cosA) - sqrt(1. / 3.) * sinA
        matrix[2][0] = 1. / 3. * (1.0 - cosA) - sqrt(1. / 3.) * sinA
        matrix[2][1] = 1. / 3. * (1.0 - cosA) + sqrt(1. / 3.) * sinA
        matrix[2][2] = cosA + 1. / 3. * (1.0 - cosA)

        return matrix

    def print_pixel_data(self, save_filepath, img):
        # For debugging purposes
        text = ""
        for row in img:
            for pixel in row:
                text += str(pixel) + "\n"
        out_file = open(save_filepath, "w")
        out_file.write(text)
        out_file.close()


    def parse_int(self, s):
        val = literal_eval(s)
        if isinstance(val, int) or (isinstance(val, float) and val.is_integer()):
            return val, True
        return 0, False

    def parse_soft_categorical(self, label, label_delimiter):
        fields = label.split(label_delimiter)
        result = [float(s) for s in fields]
        return result


    def determine_label_types_and_dims(self, raw_labels, label_delimiter):
        '''
        Get a list of the label types in the csv, as well as max number of
        classes if the type is categorical
        :param csv_reader:
        :param label_delimiter:
        :return: a list of DataTypes for each label, and a list of max values for each label if it is one hot categorical
        '''
        types = []
        label_dims = []
        first_read = True
        parsed_labels = []
        for labels in raw_labels:
            if first_read:
                first_read = False
                label_dims = [0 for _ in range(0, len(labels))]
                types = [DataType.BINARY for _ in range(0, len(labels))]

            parsed_row = []
            for idx, label in enumerate(labels):

                if len(label) == 0:
                    # it's an unknown label
                    parsed_row.append(None)
                    continue

                if types[idx] == DataType.SOFT_CATEGORICAL:
                    parsed_row.append(self.parse_soft_categorical(label, label_delimiter))
                    continue

                is_int = False
                if label_delimiter not in label:
                    value, is_int = self.parse_int(label)
                if is_int:
                    if types[idx] == DataType.ONE_HOT_CATEGORICAL:
                        label_dims[idx] = max(label_dims[idx], value+1)
                    else:
                        label_dims[idx] = max(label_dims[idx], value)
                    if label_dims[idx] > 1 and types[idx] == DataType.BINARY:
                        types[idx] = DataType.ONE_HOT_CATEGORICAL
                        label_dims[idx] += 1
                    parsed_row.append(value)
                else:
                    types[idx] = DataType.SOFT_CATEGORICAL
                    label_dims[idx] = len(label.split(label_delimiter))
                    parsed_row.append(self.parse_soft_categorical(label, label_delimiter))
            parsed_labels.append(parsed_row)

        return types, label_dims, parsed_labels


    def get_paths_and_labels_from_csv_reader(self, csv_reader, image_root_dir, has_header, label_suffix, label_delimiter='|', unknown_label_value=-1):
        '''
        Get list of paths and labels from the given csv-format iterator
        :param csv_reader: iterator over lists of values (image paths and their labels)
        :param image_root_dir: root path for the image paths
        :param has_header: if the csv_reader contains a header line
        :param label_suffix: a suffix that labels should be appended with
        :param label_delimiter: delimiter for an attr
        :param unknown_label_value: for an unknown label, put this value
        :return:
        '''
        num_labels = 0
        first_read = True
        im_per_row = self.im_per_row

        class_labels = []
        if has_header:
            class_labels = next(csv_reader, None)[im_per_row:]
            class_labels = [label + label_suffix for label in class_labels]

        image_paths = []
        image_string_labels = []

        for row in csv_reader:
            if len(row) == 0:
                continue
            image_paths.append(list(map(lambda s: os.path.join(image_root_dir, s), row[:im_per_row])))
            label = [s for s in row[im_per_row:]]
            if first_read:
                first_read = False
                num_labels = len(label)
            if len(label) != num_labels:
                print(row[:im_per_row])
                raise ValueError("There are a different number of labels " + str(len(label)) + " " + str(num_labels))
            image_string_labels.append(label)

        # process the raw image labels into
        # either multiple inputs or one input, and either
        # binary or categorical data
        # currently, we assume that the output will take the least number of bits to encode the label
        # (e.g. a label that can be interpreted as binary will be interpreted as binary)
        # TODO can extend to handle image labels or numerical data
        label_types, label_dims, image_raw_labels = self.determine_label_types_and_dims(image_string_labels,
                                                                                       label_delimiter)

        #print(label_dims)
        #print(class_labels)
        image_labels = []

        format_as_dict = has_header and num_labels > 1
        if format_as_dict:
            image_labels = {}

        if num_labels > 1:
            # append the labels as multiple arrays/outputs
            for label_idx, label_dim in enumerate(label_dims):
                if label_types[label_idx] == DataType.ONE_HOT_CATEGORICAL:
                    if not format_as_dict:
                        image_labels.append(np.zeros((len(image_raw_labels), label_dim)))
                        for row_idx, labels in enumerate(image_raw_labels):
                            if labels[label_idx] is None:
                                image_labels[-1][row_idx] = [unknown_label_value for _ in range(0, label_dim)]
                            else:
                                image_labels[-1][row_idx, labels[label_idx]] = 1
                    else:
                        class_label = class_labels[label_idx]
                        image_labels[class_label] = np.zeros((len(image_raw_labels), label_dim))
                        for row_idx, labels in enumerate(image_raw_labels):
                            if labels[label_idx] is None:
                                image_labels[class_label][row_idx] = [unknown_label_value for _ in range(0, label_dim)]
                            else:
                                image_labels[class_label][row_idx, labels[label_idx]] = 1
                else:
                    if label_dim > 1:
                        if not format_as_dict:
                            image_labels.append(np.zeros((len(image_raw_labels), label_dim)))
                            for row_idx, labels in enumerate(image_raw_labels):
                                if labels[label_idx] is None:
                                    image_labels[-1][row_idx] = [unknown_label_value for _ in range(0, label_dim)]
                                else:
                                    image_labels[-1][row_idx] = labels[label_idx]
                        else:
                            class_label = class_labels[label_idx]
                            image_labels[class_label] = np.zeros((len(image_raw_labels), label_dim))
                            for row_idx, labels in enumerate(image_raw_labels):
                                if labels[label_idx] is None:
                                    image_labels[class_labels[label_idx]][row_idx] = [unknown_label_value for _ in range(0, label_dim)]
                                else:
                                    image_labels[class_labels[label_idx]][row_idx] = labels[label_idx]
                    else:
                        if not format_as_dict:
                            image_labels.append(np.zeros(len(image_raw_labels)))
                            for row_idx, labels in enumerate(image_raw_labels):
                                if labels[label_idx] is None:
                                    image_labels[-1][row_idx] = unknown_label_value
                                else:
                                    image_labels[-1][row_idx] = labels[label_idx]
                        else:
                            class_label = class_labels[label_idx]
                            image_labels[class_label] = np.zeros(len(image_raw_labels))
                            for row_idx, labels in enumerate(image_raw_labels):
                                if labels[label_idx] is None:
                                    image_labels[class_labels[label_idx]][row_idx] = unknown_label_value
                                elif isinstance(labels[label_idx], float) or isinstance(labels[label_idx], int):
                                    image_labels[class_labels[label_idx]][row_idx] = labels[label_idx]
                                else:
                                    image_labels[class_labels[label_idx]][row_idx] = labels[label_idx][0] #TODOSHARON: look into how this occurs

        else:
            # there is only one label
            label_start_indices = []
            for label_idx, label_dim in enumerate(label_dims):
                if label_idx == 0:
                    label_start_indices.append(0)
                else:
                    label_start_indices.append(label_start_indices[-1] + label_dims[label_idx - 1])

            label_total_dim = sum(label_dims)

            image_labels = np.zeros((len(image_raw_labels), label_total_dim))

            for row_idx, labels in enumerate(image_raw_labels):
                for label_idx, start_idx in enumerate(label_start_indices):
                    if label_types[label_idx] == DataType.ONE_HOT_CATEGORICAL:
                        # categorical
                        if labels[label_idx] is None:
                            for d in range(0, label_dims[label_idx]):
                                image_labels[row_idx, start_idx + d] = unknown_label_value
                        else:
                            image_labels[row_idx, start_idx + labels[label_idx]] = 1
                    else:
                        # binary / soft categorical
                        if label_types[label_idx] == DataType.SOFT_CATEGORICAL:
                            for i in range(0, label_dims[label_idx]):
                                if labels[label_idx] is None:
                                    image_labels[row_idx, start_idx + i] = unknown_label_value
                                else:
                                    image_labels[row_idx, start_idx + i] = labels[label_idx][i]
                        else:
                            if labels[label_idx] is None:
                                image_labels[row_idx, start_idx] = unknown_label_value
                            else:
                                image_labels[row_idx, start_idx] = labels[label_idx]

            if image_labels.shape[1] == 1:
                image_labels = np.reshape(image_labels, image_labels.shape[0])

            image_labels = [image_labels]

        formatted_label_types = label_types
        if format_as_dict:
            formatted_label_types = {}
            for label_idx, label in enumerate(class_labels):
                formatted_label_types[label] = label_types[label_idx]

        return image_paths, image_labels, formatted_label_types

    def get_paths_and_labels_from_csv(self, csv_path, delimiter=',', has_header=True, label_suffix='', label_delimiter='|', unknown_label_value=-1):
        '''
        Read the paths and labels from the given csv file
        Return the ordered list of paths and labels
        :param csv_path: path to the csv file
        :param delimiter: delimiter used in the csv file
        :param has_header: whether the csv file has a header
        :param label_suffix: a suffix that should be appended to the label (only applies if there is a header)
        :param unknown_label_value: for an unknown label, put this value
        :return:
        '''

        with open(csv_path, newline='') as csv_file:
            # get the directory path of the csv file
            reader = csv.reader(csv_file, delimiter=delimiter)
            dir_path = os.path.dirname(csv_path)
            return self.get_paths_and_labels_from_csv_reader(reader, dir_path, has_header, label_suffix, label_delimiter, unknown_label_value)

    def flow_from_csv(self, csv_path, delimiter=',', has_header=True, batch_size=32, shuffle=True, save_to_dir=None,
                      save_prefix='', save_format='png', label_suffix='', label_delimiter='|', unknown_label_value=-1,
                      reweight_labels=False, num_target_instances=1):

        """
        :param csv_path: path to csv file to read, with image paths and their label(s). labels must be integer values.
        :param delimiter: delimiters used in the csv file
        :param has_header: if the csv file has a header line (which should be skipped)
        :param batch_size: batch size to iterate over
        :param shuffle: whether or not to shuffle the input after every full iteration
        :param save_to_dir: if not None, save transformed images to this directory
        :param save_prefix: prefix saved image filenames with this
        :param save_format: save images with this file format
        :param label_suffix: append this suffix to image labels (only matters if there is a header)
        :param label_delimiter: delimiter within each attribute
        :param unknown_label_value: for an unknown label value, put this value
        :param reweight_labels: reweight target labels dependant on their frequency
        :param num_target_instances: number of instances of target data in batch output, used if multiple outputs expect same target data
        :return: an iterator over dynamically augmented images and their labels
        """
        image_paths, image_labels, label_types = self.get_paths_and_labels_from_csv(csv_path,
                                                                       delimiter,
                                                                       has_header,
                                                                       label_suffix,
                                                                       label_delimiter,
                                                                       unknown_label_value)

        return ImageFileListIterator(image_paths,
                                     image_labels,
                                     batch_size,
                                     shuffle,
                                     self.seed,
                                     self,
                                     self.image_size,
                                     save_to_dir,
                                     save_prefix,
                                     save_format,
                                     self.im_per_row,
                                     label_types,
                                     unknown_label_value,
                                     reweight_labels,
                                     num_target_instances)


class ImageFileListIterator(Iterator):

    def __init__(self,
                 image_paths,
                 image_labels,
                 batch_size,
                 shuffle,
                 seed,
                 image_generator,
                 image_size,
                 save_to_dir,
                 save_prefix,
                 save_format,
                 im_per_row,
                 label_types = None,
                 unknown_label_value=-1,
                 reweight_labels=False,
                 num_target_instances=1):

        """
        :param image_paths: matrix of image paths. one column for each input
        :param image_labels: a list of image label arrays, one for each output
        :param batch_size: batch size to return (or if using partitions, number of unique values to return)
        :param shuffle: if the input should be shuffled after each full iteration
        :param seed: seed for randomizer
        :param image_generator: image data generator used to augment images
        :param image_size: image size on load
        :param im_per_row: number of images per row in the matrix
        :param partition_value: the label to partition by. if None, use no partitions. Should only be used for categorical labels
        :param num_per_partition: number of images per partition
        :param num_target_instances: number of instances of target data in batch output, used if multiple outputs expect same target data
        """

        #self.partition_value = partition_value
        #self.num_per_partition = num_per_partition
        self.length = len(image_paths)
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.unknown_label_value = unknown_label_value
        #self._partition_labels()

        super().__init__(self.length,
                         batch_size,
                         shuffle,
                         seed)

        self.image_generator = image_generator
        self.image_size = image_size
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.im_per_row = im_per_row
        self.label_types = label_types

        self.reweight_labels = reweight_labels
        self.sample_weights = self.determine_sample_weights()
        if self.reweight_labels:
            self._strip_unknown_labels()

        self.num_target_instances = num_target_instances

    def _partition_labels(self):
        self.value_order = []
        self.value_to_indices = {}
        if self.partition_value is not None:
            label_values = self.image_labels[self.partition_value]
            for i in range(0, len(label_values)):
                array = label_values[i]
                if np.all(array == self.unknown_label_value):
                    continue
                if len(array.shape) >= 1:
                    value = np.argmax(array)
                else:
                    if array < 0.5:
                        value = 0
                    else:
                        value = 1
                if value not in self.value_to_indices:
                    self.value_to_indices[value] = []
                self.value_to_indices[value].append(i)
            self.length = len(self.value_to_indices)
            self.value_order = [k for k in self.value_to_indices]

    def determine_sample_weights(self):
        sample_weights = {}

        keys = []
        if isinstance(self.image_labels, list):
            sample_weights = [0 for _ in range(0, len(self.image_labels))]
            keys = range(0, len(self.image_labels))
        else:
            # it's a dictionary
            keys = [k for k in self.image_labels]

        # map label name -> label value -> count
        label_to_value_to_counts = {}
        for key in keys:
            label_to_value_to_counts[key] = {}

            for array in self.image_labels[key]:
                if np.all(array == self.unknown_label_value):
                    continue
                if len(array.shape) >= 1:
                    value = np.argmax(array)
                else:
                    if array < 0.5:
                        value = 0
                    else:
                        value = 1
                if value not in label_to_value_to_counts[key]:
                    label_to_value_to_counts[key][value] = 0
                label_to_value_to_counts[key][value] += 1

        for key in keys:
            total = sum([label_to_value_to_counts[key][value] for value in label_to_value_to_counts[key]])
            sample_weights[key] = np.zeros(len(self.image_labels[key]))
            for idx, array in enumerate(self.image_labels[key]):
                if np.all(array == self.unknown_label_value):
                    sample_weights[key][idx] = K.epsilon()
                else:
                    if len(array.shape) >= 1:
                        value = np.argmax(array)
                    else:
                        if array < 0.5:
                            value = 0
                        else:
                            value = 1
                    weight = total / label_to_value_to_counts[key][value]
                    sample_weights[key][idx] = weight
        return sample_weights


    def _strip_unknown_label(self, label):
        if np.all(label == self.unknown_label_value):
            if isinstance(label, np.float64):
                # assume it's a binary category
                return 0.5
            else:
                return label / np.sum(label, axis=-1)
        else:
            return label


    def _strip_unknown_labels(self):
        # strip unknown labels from image labels
        image_labels = self.image_labels

        if isinstance(image_labels, dict):
            for label in image_labels:
                for i in range(0, len(image_labels[label])):
                    image_labels[label][i] = self._strip_unknown_label(image_labels[label][i])
        else:
            for label in range(0, len(image_labels)):
                for i in range(0, len(image_labels[label])):
                    image_labels[label][i] = self._strip_unknown_label(image_labels[label][i])

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = []
        batch_y = None
        sample_weights = []

        relevant_indices = index_array
        """if self.partition_value is not None:
            relevant_indices = []

            # for each partition, choose n random indices
            for i in range(0, len(index_array)):
                value = self.value_order[index_array[i]]
                partition_indices = self.value_to_indices[value]
                sample = partition_indices
                if len(sample) > self.num_per_partition:
                    sample = np.random.choice(partition_indices, self.num_per_partition, replace=False)
                relevant_indices.extend(sample)"""

        if isinstance(self.image_labels, dict):
            batch_y = {}
            sample_weights = {}
            for label_name in self.image_labels.keys():
                batch_y[label_name] = self.image_labels[label_name][relevant_indices]
                sample_weights[label_name] = self.sample_weights[label_name][relevant_indices]
        else:
            batch_y = [self.image_labels[i][relevant_indices] for i in range(0, len(self.image_labels))]
            sample_weights = [self.sample_weights[i][relevant_indices] for i in range(0, len(self.image_labels))]

        row_transform_params = dict()
        for image_idx in range(self.im_per_row):
            b_x = np.zeros((len(relevant_indices),) + self.image_size + (3,), dtype=K.floatx())

            for list_idx, row_idx in enumerate(relevant_indices):
                filename = self.image_paths[row_idx][image_idx]
                img = image.load_img(path=filename, target_size = self.image_size)
                x = image.img_to_array(img, self.image_generator.data_format)

                override_params = None
                if self.im_per_row > 1:
                    if row_idx in row_transform_params:
                        override_params = row_transform_params[row_idx]
                    else:
                        override_params = self.image_generator.get_random_transform_params(x, None)
                        override_params.pop('channel_shift_intensity') # Want unique channel shift intensity for images in a row
                        row_transform_params[row_idx] = override_params

                x = self.image_generator.apply_transform(x, override_params)
                x = self.image_generator.standardize(x)
                b_x[list_idx] = x

                if self.save_to_dir:
                    img = image.array_to_img(x, scale=True)
                    fname = '{prefix}_{index}_{im_index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                      index=row_idx,
                                                                      im_index=image_idx,
                                                                      hash=np.random.randint(1e4),
                                                                      format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))
            if hasattr(img, 'close'):
                img.close()
            batch_x.append(b_x)

        if self.im_per_row == 1:
            batch_x = batch_x[0]

        # Duplicate batch output data if applicable
        if isinstance(batch_y, dict):
            batch_y_out = batch_y
            sample_weights_out = sample_weights
        else:
            batch_y_out = []
            sample_weights_out = []
            for i in range(0, self.num_target_instances):
                for j in range(0, len(batch_y)):
                    batch_y_out.append(batch_y[j])
                    sample_weights_out.append(sample_weights[j])

        if self.reweight_labels:
            return batch_x, batch_y_out, sample_weights_out

        return batch_x, batch_y_out

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)

        return self._get_batches_of_transformed_samples(index_array)
