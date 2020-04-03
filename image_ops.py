from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import Iterator, ImageDataGenerator
import os.path
import numpy as np
from tensorflow.python.keras import backend as K
from io import BytesIO
from PIL import Image as pil_image
import PIL.ImageOps
from math import sqrt, cos, sin, radians


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
                 make_grayscale=False,
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

    def apply_transform(self, inp, transform_params=None):
        transform_params = self.get_random_transform_params(inp)
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

        if not greyscale and self.shift_hue and np.random.random() <= 0.1:
            result = self.rotate_hue(result, np.random.randint(0, 359))

        if not greyscale and self.invert_colours and np.random.random() <= 0.1:
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

    def flow_from_iterator(self, image_paths, image_labels, batch_size=32, shuffle=True, save_to_dir=None,
                           save_prefix='', save_format='png', unknown_label_value=-1,
                           reweight_labels=False, label_bounds=None):

        """
        :param image_paths: list of image paths
        :param image_labels: labels associated with images
        :param batch_size: batch size to iterate over
        :param shuffle: whether or not to shuffle the input after every full iteration
        :param save_to_dir: if not None, save transformed images to this directory
        :param save_prefix: prefix saved image filenames with this
        :param save_format: save images with this file format
        :param unknown_label_value: for an unknown label value, put this value
        :param reweight_labels: reweight target labels dependant on their frequency
        :param label_bounds: list of pre-crop boundaries for each image
        :return: ImageFileListIterator to provide input data for model training/validation
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
                                     unknown_label_value,
                                     reweight_labels,
                                     label_bounds)


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
                 unknown_label_value=-1,
                 reweight_labels=False,
                 label_bounds=None):

        self.length = len(image_paths)
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.unknown_label_value = unknown_label_value

        super().__init__(self.length,
                         batch_size,
                         shuffle,
                         seed)

        self.image_generator = image_generator
        self.image_size = image_size
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.reweight_labels = reweight_labels
        self.sample_weights = self.determine_sample_weights()
        self.label_bounds = label_bounds

    def determine_sample_weights(self):
        # Calculate counts for each classification found
        key_counts = {}
        for label in self.image_labels:
            if len(label.shape) >= 1:
                key = np.argmax(label)
            else:
                key = 0 if label < 0.5 else 1

            if key not in key_counts:
                key_counts[key] = 0
            key_counts[key] += 1

        # Calculate classification weights
        num_labels = len(self.image_labels)
        key_weights = {k: num_labels / v for k, v in key_counts.items()}
        min_weight = min(key_weights.values())
        key_weights = {k: v / min_weight for k, v in key_weights.items()}

        # Create weight for each label
        sample_weights = np.zeros(num_labels)
        for idx, label in enumerate(self.image_labels):
            if len(label.shape) >= 1:
                key = np.argmax(label)
            else:
                key = 0 if label < 0.5 else 1

            sample_weights[idx] = key_weights[key]

        return sample_weights

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = []

        relevant_indices = index_array
        batch_y = [self.image_labels[i] for i in relevant_indices]
        sample_weights = self.sample_weights[relevant_indices]

        b_x = np.zeros((len(relevant_indices),) + self.image_size + (3,), dtype=K.floatx())
        for list_idx, row_idx in enumerate(relevant_indices):
            filename = self.image_paths[row_idx]
            resample = pil_image.BICUBIC
            if self.label_bounds is not None:
                img = image.load_img(path=filename)
                img = img.crop(tuple(self.label_bounds[row_idx]))
                img = img.resize(self.image_size, resample=resample)
            else:
                img = image.load_img(path=filename, target_size=self.image_size)

            x = image.img_to_array(img, self.image_generator.data_format)

            transform_params = self.image_generator.get_random_transform_params(x, None)
            x = self.image_generator.apply_transform(x, transform_params)
            x = self.image_generator.standardize(x)
            b_x[list_idx] = x

            if self.save_to_dir:
                img = image.array_to_img(x, scale=True)
                fname = '{prefix}_{index}_{sampling}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=row_idx,
                                                                  sampling=resample,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
            if hasattr(img, 'close'):
                img.close()
            batch_x.append(b_x)

        batch_y_out = [batch_y]

        if self.reweight_labels:
            return batch_x, batch_y_out, sample_weights

        return batch_x, batch_y_out

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)

        return self._get_batches_of_transformed_samples(index_array)
