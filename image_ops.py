from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import Iterator, ImageDataGenerator
import csv
import os.path
import numpy as np
from tensorflow.python.keras import backend as K
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

    def apply_transform(self, inp, override_params=None):
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


    # TODO: rename since it doesn't involve a csv now
    def flow_from_csv(self, image_paths, image_labels, label_types, batch_size=32, shuffle=True, save_to_dir=None,
                      save_prefix='', save_format='png', unknown_label_value=-1,
                      reweight_labels=False, num_target_instances=1):

        """
        :param image_paths: list of image paths
        :param image_labels: labels associated with images
        :param label_types: TODO??
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
                 label_types=None,
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
        # Calculate counts for each classification found
        key_counts = {}
        for label in self.image_labels:
            #label = self.image_labels[im_index] # TODO: check dimension of list, this assumes 1 dimension right now
            if len(label.shape) >= 1:
                key = np.argmax(label)
            else:
                key = 0 if label < 0.5 else 1

            if key not in key_counts:
                key_counts[key] = 0
            key_counts[key] += 1

        # Calculate classification weights
        num_labels = len(self.image_labels)
        key_weights = {k: num_labels/v for k, v in key_counts.items()}
        min_weight = min(key_weights.values())
        key_weights = {k: v/min_weight for k, v in key_weights.items()}

        # Create weight for each label
        sample_weights = np.zeros(num_labels)
        for idx, label in enumerate(self.image_labels):
            if len(label.shape) >= 1:
                key = np.argmax(label)
            else:
                key = 0 if label < 0.5 else 1

            sample_weights[idx] = key_weights[key]

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

        relevant_indices = index_array
        batch_y = [self.image_labels[i] for i in relevant_indices]
        sample_weights = self.sample_weights[relevant_indices]

        b_x = np.zeros((len(relevant_indices),) + self.image_size + (3,), dtype=K.floatx())
        for list_idx, row_idx in enumerate(relevant_indices):
            filename = self.image_paths[row_idx]
            img = image.load_img(path=filename, target_size=self.image_size)
            x = image.img_to_array(img, self.image_generator.data_format)

            # TODO: can get rid of override param functionality
            transform_params = self.image_generator.get_random_transform_params(x, None)
            x = self.image_generator.apply_transform(x, transform_params)
            x = self.image_generator.standardize(x)
            b_x[list_idx] = x

            if self.save_to_dir:
                img = image.array_to_img(x, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=row_idx,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
            if hasattr(img, 'close'):
                img.close()
            batch_x.append(b_x)

        batch_y_out = []
        for i in range(0, self.num_target_instances):
                batch_y_out.append(batch_y)

        if self.reweight_labels:
            return batch_x, batch_y_out, sample_weights

        return batch_x, batch_y_out

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)

        return self._get_batches_of_transformed_samples(index_array)
