import sys
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from collections.abc import Iterable

########################################################################################################################
#
# Tensorflow Functions
#
########################################################################################################################

@tf.function
def tf_transform_batch(images, function, dtype=None, *args, **kwargs):
    """Applies function to a batch of images"""

    def fn(image):
        return function(image, *args, **kwargs)

    return tf.map_fn(fn, images, dtype=dtype)


@tf.function
def tf_gaussian_kernel(channels, kernel_size, sigma):
    """Generate 2D Gaussian kernel"""
    ax = tf.range(-kernel_size, kernel_size + 1.0)
    xx, yy = tf.meshgrid(ax, ax)
    kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / (tf.reduce_sum(kernel) + 1.0e-7)
    kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
    return kernel


@tf.function
def tf_gaussian_blur(image, sigma_range, prob):
    """Applies random gaussian blur with given probability"""
    if tf.random.uniform(shape=()) > prob:
        return image

    sigma = tf.random.uniform((), minval=sigma_range[0], maxval=sigma_range[0], dtype=tf.float32)
    size = tf.maximum(1.0, tf.floor(1.5 * sigma))
    channels_n = tf.shape(image)[-1]

    kernel = tf_gaussian_kernel(channels_n, size, sigma)
    kernel = kernel[..., tf.newaxis]

    image_float = tf.cast(image, tf.float32)
    image_padded = tf.pad(image_float, [(size, size), (size, size), (0, 0)], mode="REFLECT")
    image_padded = tf.expand_dims(image_padded, 0)

    image_output = tf.nn.depthwise_conv2d(image_padded, kernel, [1, 1, 1, 1], padding='VALID')
    return image_output[0]


@tf.function
def tf_random_crop(image, sides):
    """Performs random cropping of the batch images (each individually)"""
    images_shape = tf.shape(image)
    return tf.image.random_crop(image, (sides[0], sides[1], images_shape[-1]))


@tf.function
def tf_center_crop(images, sides):
    """Crops central region"""
    images_shape = tf.shape(images)
    top = (images_shape[1] - sides[0]) // 2
    left = (images_shape[2] - sides[1]) // 2
    return tf.image.crop_to_bounding_box(images, top, left, sides[0], sides[1])


@tf.function
def tf_center_pad(image, pad_rows, pad_cols, mode):
    """Pad image such that it remains in the center"""
    return tf.pad(image, [(pad_rows, pad_rows), (pad_cols, pad_cols), (0, 0)], mode=mode)


@tf.function
def tf_divisible_crop(image, divisor_rows, divisor_cols):
    """Crops image such that its sides are divisible by given factors"""
    image_shape = tf.shape(image)
    height = image_shape[0] - tf.math.mod(image_shape[0], divisor_rows)
    width = image_shape[1] - tf.math.mod(image_shape[1], divisor_cols)
    print(width, height)
    return tf_center_crop(tf.expand_dims(image, 0), (height, width))[0]


@tf.function
def tf_divisible_pad(image, divisor_rows, divisor_cols, mode):
    """Pad image such that its shape is divisible by given factors"""
    image_shape = tf.shape(image)
    pad_rows = divisor_rows - tf.math.mod(image_shape[0], divisor_rows)
    pad_cols = divisor_cols - tf.math.mod(image_shape[1], divisor_cols)
    pad_rows2 = pad_rows // 2
    pad_cols2 = pad_cols // 2
    return tf.pad(image, [(pad_rows2, pad_rows - pad_rows2), (pad_cols2, pad_cols - pad_cols2), (0, 0)], mode=mode)


@tf.function
def tf_generate_random_transform(center_x, center_y, transform_n, rotate_range=(0.0, 0.0),
                                 scale_range=(1.0, 1.0), seed=None):
    """Generate random random transformation matrix: tf.Tensor of the shape (transform_n, 8)"""
    # Generate random angles
    angle_min, angle_max = rotate_range[0], rotate_range[1]
    angles = tf.random.uniform(shape=(transform_n,), minval=angle_min, maxval=angle_max, seed=seed)

    # Generate random scaling factors
    scales = tf.random.uniform(shape=(transform_n,), minval=scale_range[0], maxval=scale_range[1])

    # Generate zero vector
    zero = tf.zeros(shape=(transform_n,))

    # Compute  sin and cos
    sin_angles, cos_angles = tf.sin(angles), tf.cos(angles)

    # Compute transformation components
    a0 = scales * cos_angles
    a1 = scales * sin_angles
    c0 = -scales * (center_x * cos_angles - center_y * sin_angles) + center_x
    c1 = -scales * (center_x * sin_angles + center_y * cos_angles) + center_y

    return tf.stack([a0, -a1, c0, a1, a0, c1, zero, zero], axis=-1)


@tf.function
def tf_random_affine_batch(images, rotate_range=(0.0, 0.0), scale_range=(1.0, 1.0), seed=None):
    """Performs random rotation and scaling of the batch"""
    images_shape = tf.shape(images)
    x_c = tf.cast(images_shape[2] / 2, tf.float32)
    y_c = tf.cast(images_shape[1] / 2, tf.float32)

    # Generate random transform
    transform = tf_generate_random_transform(x_c, y_c, images_shape[0], rotate_range, scale_range, seed)
    return tfa.image.transform(images, transform, output_shape=[images_shape[1], images_shape[2]])


@tf.function
def tf_normalize_mean_std(images):
    """Normalize images by subtracting mean and dividing by std"""
    out_images = tf.cast(images, tf.float32)
    return tf.image.per_image_standardization(out_images)


@tf.function
def tf_random_brightness_contrast(image, brightness, contrast):
    """Random brightness/contrast adjustment"""
    out_image = tf.cast(image, tf.float32)
    out_image = out_image / (tf.reduce_max(out_image) + 1.0e-7)
    out_image = (out_image - 0.5) * tf.random.uniform((), minval=contrast[0], maxval=contrast[1]) + 0.5
    out_image = out_image + tf.random.uniform((), minval=brightness[0], maxval=brightness[1])
    return tf.clip_by_value(out_image, 0.0, 1.0)


@tf.function
def tf_random_hsv(images, hue, saturation, value, seed=None):
    """Randomly adjust hue and saturation"""
    images_shape = tf.shape(images)
    batch_size = images_shape[0]

    # Rescale images
    images_rescaled = tf.cast(images, tf.float32)
    images_rescaled = images_rescaled / (tf.math.reduce_max(images_rescaled) + 1.0e-7)
    hsv_images = tf.image.rgb_to_hsv(images_rescaled)

    # Sample random HSV transformations
    random_hue = tf.random.uniform(shape=(batch_size, 1, 1), minval=hue[0], maxval=hue[1], seed=seed)
    random_value = tf.random.uniform(shape=(batch_size, 1, 1), minval=value[0], maxval=value[1], seed=seed)
    random_saturation = tf.random.uniform(shape=(batch_size, 1, 1), minval=saturation[0], maxval=saturation[1], seed=seed)

    # Augment image
    new_hue = tf.clip_by_value(hsv_images[:, :, :, 0] + random_hue, 0, 1)
    new_saturation = tf.clip_by_value(hsv_images[:, :, :, 1] * random_saturation, 0, 1)
    new_value = tf.clip_by_value(hsv_images[:, :, :, 2] + random_value, 0, 1)
    augmented = tf.stack([new_hue, new_saturation, new_value], axis=3)

    return tf.image.hsv_to_rgb(augmented)


@tf.function
def tf_box_blur(image, max_size, prob):
    """Random box blur with given probability"""
    if tf.random.uniform(shape=()) >= prob:
        return image

    size = tf.random.uniform((), minval=0, maxval=(max_size + 1) // 2, dtype=tf.int32)
    ksize = 2 * size + 1

    ch_n = tf.shape(image)[-1]
    kernel = tf.ones(shape=(ksize, ksize, ch_n, ch_n), dtype=tf.float32) / tf.cast(ksize * ksize, tf.float32)

    image_float = tf.cast(image, tf.float32)
    image_padded = tf.pad(image_float, [(size, size), (size, size), (0, 0)], mode="REFLECT")
    image_output = tf.nn.conv2d(tf.expand_dims(image_padded, 0), kernel, 1, "VALID")
    return image_output[0]


########################################################################################################################
#
# Augmentation Classes and Utility Functions
#
########################################################################################################################


def init_size(value):
    """Makes a tuple from the value"""
    if isinstance(value, Iterable):
        return value
    elif type(value) == int:
        return value, value
    else:
        raise RuntimeError("Type %s is not supported!" % (str(type(value))))


def init_range(value):
    """Initializes a range the value"""
    if isinstance(value, Iterable):
        return value
    elif type(value) == int or type(value) == float:
        return -value, value
    else:
        raise RuntimeError("Type %s is not supported!" % (str(type(value))))


def init_scale_range(value):
    """Initializes a range the scaling value"""
    if type(value) == int or type(value) == float:
        return min(value, 1.0), max(value, 1.0)
    elif isinstance(value, Iterable):
        return value
    else:
        raise RuntimeError("Type %s is not supported!" % (str(type(value))))


def tf_apply_transform_fn(function, batch_function=False, batch_data=False, dtype=None, *args, **kwargs):
    """Constructs Tensorflow transformation function

     Keyword arguments:
         function : is a tensorflow function to be applied
         batch_function (bool) : indicate True to state that the function handles batches (N, H, W, C)
         batch_data (bool) :  indicate True to state that input Data is a batch (N, H, W, C), otherwise it's an image
         dtype : type of the output (must be specified if it's different from the original)

    Returns:
        function, which can be used in tf.Data.Dataset.map

     """
    if (batch_data and batch_function) or (not batch_data and not batch_function):
        def fn(images, labels):
            return function(images, *args, **kwargs), labels
    elif batch_data and not batch_function:
        def fn(images, labels):
            return tf_transform_batch(images, function, dtype, *args, **kwargs), labels
    elif not batch_data and batch_function:
        def fn(images, labels):
            return function(tf.expand_dims(images, axis=0), *args, **kwargs)[0], labels
    return tf.function(lambda images, labels: fn(images, labels))


def tf_duplicated_transform_fn(function, batch_function=False, batch_data=False, dtype=None, *args, **kwargs):
    """Constructs Tensorflow transformation function

     Keyword arguments:
         function : is a tensorflow function to be applied
         batch_function (bool) : indicate True to state that the function handles batches (N, H, W, C)
         batch_data (bool) :  indicate True to state that input Data is a batch (N, H, W, C), otherwise it's an image
         dtype : type of the output (must be specified if it's different from the original)

     Returns:
        function, which can be used in tf.Data.Dataset.map

     """
    if (batch_data and batch_function) or (not batch_data and not batch_function):
        def fn(images, labels):
            combined = tf.concat([images, labels], axis=-1)
            proc_combined = function(combined, *args, **kwargs)
            shape = tf.shape(images)
            return proc_combined[..., :shape[-1]], proc_combined[..., shape[-1]:]
    elif batch_data and not batch_function:
        def fn(images, labels):
            combined = tf.concat([images, labels], axis=-1)
            proc_combined = tf_transform_batch(combined, function, dtype, *args, **kwargs)
            shape = tf.shape(images)
            return proc_combined[..., :shape[-1]], proc_combined[..., shape[-1]:]
    elif not batch_data and batch_function:
        def fn(images, labels):
            combined = tf.concat([images, labels], axis=-1)
            proc_combined = function(tf.expand_dims(combined, axis=0), *args, **kwargs)[0]
            shape = tf.shape(images)
            return proc_combined[..., :shape[-1]], proc_combined[..., shape[-1]:]
    return tf.function(lambda images, labels: fn(images, labels))


class Augmentor(object):
    """
    Generic class for augmentation operation. Each new operation should reload function `transform_fn`.

    Args:
        num_parallel_calls (int) : Number of parallel processes performing the operation. The default
            value is tf.Data.experimental.AUTOTUNE
        augment_label (bool) : if True the same transformation is applied to label.

    Attributes:
        num_parallel_calls (int) : See above
        augment_label (bool) : See above

    """
    def __init__(self, num_parallel_calls=tf.data.experimental.AUTOTUNE, augment_label=False):
        self.num_parallel_calls = num_parallel_calls
        self.augment_label = augment_label

    def transform_fn(self, batch_level=False):
        """Must return Tensorflow function for processing the Data.

        Args:
            batch_level (bool) : if True, returned function should process image batch, otherwise individual image

        """
        raise NotImplementedError("Operation is not implemented!")

    def augment(self, dataset, batch_level=False):
        """Extends Data pipeline with current transform

        Args:
            batch_level (bool) : if True, batch transformation is applied

        """
        return dataset.map(self.transform_fn(batch_level), num_parallel_calls=self.num_parallel_calls)


class RandomCrop(Augmentor):
    """
    Randomly crops an image region of the defined size.

    Args:
        size (int or tuple) : the size of the region, either single integer or tuple (height, width)
        num_parallel_calls (int) : the number of parallel processes (see documentation of Augmentor class)
        augment_label (bool) : if True the same transformation is applied to label.

    Attributes:
        size (tuple) : tuple of two integers (height and width of the crop region)

    """
    def __init__(self, size, num_parallel_calls=tf.data.experimental.AUTOTUNE, augment_label=False):
        super(RandomCrop, self).__init__(num_parallel_calls, augment_label)
        self.size = init_size(size)

    def transform_fn(self, batch_level=False):
        """Returns Tensorflow function which performs random cropping

        Args:
            batch_level (bool) : if True, returned function processes image batches, otherwise individual images

        """
        if self.augment_label:
            return tf_duplicated_transform_fn(tf_random_crop, False, batch_level, None, self.size)
        else:
            return tf_apply_transform_fn(tf_random_crop, False, batch_level, None, self.size)


class CenterCrop(Augmentor):
    """
    Crops central image region of the defined size.

    Args:
        size (int or tuple) : the size of the region, either single integer or tuple (height, width)
        num_parallel_calls (int) : the number of parallel processes (see documentation of Augmentor class)
        augment_label (bool) : if True the same transformation is applied to label.

    Attributes:
        size (tuple) : tuple of two integers (height and width of the crop region)

    """
    def __init__(self, size, num_parallel_calls=tf.data.experimental.AUTOTUNE, augment_label=False):
        super(CenterCrop, self).__init__(num_parallel_calls, augment_label)
        self.size = init_size(size)

    def transform_fn(self, batch_level=False):
        """Returns Tensorflow function which performs central cropping

        Args:
            batch_level (bool) : if True, returned function processes image batches, otherwise individual images

        """
        if self.augment_label:
            return tf_duplicated_transform_fn(tf_center_crop, True, batch_level, None, self.size)
        else:
            return tf_apply_transform_fn(tf_center_crop, True, batch_level, None, self.size)


class Affine(Augmentor):
    """
    Performs random rotation and scaling of an image or batch within the defined ranges.

    Args:
        rotation (float or tuple) : min and max value of rotation angle (in degrees)
        scale (float or tuple) : min and max scale value, 1.0 means no scaling, >1.0 - zoom out, <1.0 - zoom in
        num_parallel_calls (int) : the number of parallel processes (see documentation of Augmentor class)
        augment_label (bool) : if True the same transformation is applied to label.

    Attributes:
        size (tuple) : tuple of two integers (height and width of the crop region)

    """

    def __init__(self, rotation, scale, num_parallel_calls=tf.data.experimental.AUTOTUNE, augment_label=False):
        super(Affine, self).__init__(num_parallel_calls, augment_label)
        self.rot = init_range(rotation)
        self.rot = np.deg2rad(self.rot[0]), np.deg2rad(self.rot[1])
        self.scale = init_scale_range(scale)

    def transform_fn(self, batch_level=False):
        """Returns Tensorflow function which performs affine transform

        Args:
            batch_level (bool): if True, returned function processes image batches, otherwise individual images

        """
        if self.augment_label:
            return tf_duplicated_transform_fn(tf_random_affine_batch, True, batch_level, None, self.rot, self.scale)
        else:
            return tf_apply_transform_fn(tf_random_affine_batch, True, batch_level, None, self.rot, self.scale)


class NormalizeMeanStd(Augmentor):
    """
    Normalizes images by subtracting mean and dividing by standard deviation. Output has type tf.float32

    """

    def __init__(self, num_parallel_calls=tf.data.experimental.AUTOTUNE, augment_label=False):
        super(NormalizeMeanStd, self).__init__(num_parallel_calls, augment_label)

    def transform_fn(self, batch_level=False):
        """Returns Tensorflow function performing image normalization

        Args:
            batch_level (bool): if True, returned function processes image batches, otherwise individual images

        """
        if self.augment_label:
            return tf_duplicated_transform_fn(tf_normalize_mean_std, True, batch_level)
        else:
            return tf_apply_transform_fn(tf_normalize_mean_std, True, batch_level)


class BrightnessContrast(Augmentor):
    """
    Performs random adjustments of the image brightness and contrast.
    The resulting image is of type tf.float32 with intensities in the range [0.0, 1.0]

    Args:
        brightness (float or tuple) : range of random additive brightness (if single number, brightness is randomly
            sampled from the range (-brightness, brightness).
        contrast (float or tuple) : range of random multiplicative contrast (if single number, contrast is randomly
            sampled from the range (min(contrast, 1.0), max(contrast, 1.0))
        num_parallel_calls (int) : the number of parallel processes (see documentation of Augmentor class)
        augment_label (bool) : if True the same transformation is applied to label.

    Attributes:
        brightness (tuple) : tuple of two float values (minimum and maximum brightness factors)
        contrast (tuple) : tuple of two float values (minimum and maximum contrast factors)

    """
    def __init__(self, brightness=0.0, contrast=1.0, num_parallel_calls=tf.data.experimental.AUTOTUNE,
                 augment_label=False):
        super(BrightnessContrast, self).__init__(num_parallel_calls, augment_label)
        self.brightness = init_range(brightness)
        self.contrast = init_scale_range(contrast)

    def transform_fn(self, batch_level=False):
        """Returns Tensorflow function performing random adjustments of brightness / contrast

        Args:
            batch_level (bool): if True, returned function processes image batches, otherwise individual images

        """
        if self.augment_label:
            return tf_duplicated_transform_fn(tf_random_brightness_contrast, False, batch_level, None,
                                              self.brightness, self.contrast)
        else:
            return tf_apply_transform_fn(tf_random_brightness_contrast, False, batch_level, None,
                                         self.brightness, self.contrast)


class HSV(Augmentor):
    """
    Performs random adjustments of image hue, saturation and value.
    Input image must be tf.float32 with intensities in the range [0.0, 1.0]. The resulting image has the same format.

    Args:
        hue (float or tuple) : range of random additive hue (if single number, brightness is randomly
            sampled from the range (-brightness, brightness).
        saturation (float or tuple) : range of random multiplicative contrast (if single number, saturation is randomly
            sampled from the range (min(contrast, 1.0), max(contrast, 1.0))
        value (float or tuple) : range of random additive value (if single number, brightness is randomly
            sampled from the range (-value, value).

        num_parallel_calls (int) : the number of parallel processes (see documentation of Augmentor class)
        augment_label (bool) : if True the same transformation is applied to label.

    Attributes:
        hue (tuple) : tuple of two float values (minimum and maximum hue factors)
        saturation (tuple) : tuple of two float values (minimum and maximum saturation factors)

    """
    def __init__(self, hue=0.0, saturation=1.0, value=0.0, num_parallel_calls=tf.data.experimental.AUTOTUNE,
                 augment_label=False):
        super(HSV, self).__init__(num_parallel_calls, augment_label)
        self.hue = init_range(hue)
        self.saturation = init_scale_range(saturation)
        self.value = init_range(value)

    def transform_fn(self, batch_level=False):
        """Returns Tensorflow function performing random adjustments of brightness / contrast

        Args:
            batch_level (bool): if True, returned function processes image batches, otherwise individual images

        """
        if self.augment_label:
            return tf_duplicated_transform_fn(tf_random_hsv, True, batch_level, None,
                                              self.hue, self.saturation, self.value)
        else:
            return tf_apply_transform_fn(tf_random_hsv, True, batch_level, None,
                                         self.hue, self.saturation, self.value)


class GaussianNoise(Augmentor):
    """
    Adds random gaussian noise to an image.

    Args:
        min_std (float) : range of random additive brightness (if single number, brightness is randomly
            sampled from the range (-brightness, brightness).
        max_std (float) : range of random multiplicative contrast (if single number, contrast is randomly
            sampled from the range (min(contrast, 1.0), max(contrast, 1.0))
        prob (float) : the number of parallel processes (see documentation of Augmentor class)
        augment_label (bool) : if True the same transformation is applied to label.

    Attributes:
        min_std (float) : see above
        max_std (float) : see above
        prob (float) : see above

    """
    def __init__(self, min_std=0.0, max_std=0.1, prob=0.5, num_parallel_calls=tf.data.experimental.AUTOTUNE,
                 augment_label=False):

        super(GaussianNoise, self).__init__(num_parallel_calls, augment_label)
        self.min_std = min_std
        self.max_std = max_std
        self.prob = prob
        assert self.min_std <= self.max_std

    def transform_fn(self, batch_level=False):
        """Returns Tensorflow function adding random gaussian noise to an image

        Args:
            batch_level (bool): if True, returns function processes image batches, otherwise individual images

        """
        @tf.function
        def tf_random_gaussian_noise(image, min_std, max_std, prob):
            if tf.random.uniform(shape=()) < prob:
                min_value = tf.reduce_min(image)
                max_value = tf.reduce_max(image)
                std = tf.random.uniform(shape=(), minval=min_std, maxval=max_std) * (max_value - min_value)
                output_image = image + tf.random.normal(tf.shape(image), stddev=std)
                return tf.clip_by_value(output_image, min_value, max_value)
            else:
                return image

        if self.augment_label:
            return tf_duplicated_transform_fn(tf_random_gaussian_noise, False, batch_level, None,
                                              self.min_std, self.max_std, self.prob)
        else:
            return tf_apply_transform_fn(tf_random_gaussian_noise, False, batch_level, None,
                                         self.min_std, self.max_std, self.prob)


class RescaleIntensities(Augmentor):
    """
    Rescales intensities. The resulting image is of type tf.float32.

    Args:
        scale_factor (float) : defines the maximum intensity value in the resulting image
        num_parallel_calls (int) : the number of parallel processes (see documentation of Augmentor class)
        augment_label (bool) : if True the same transformation is applied to label.

    Attributes:
        scale_factor (float) : tuple of two float values (minimum and maximum brightness factors)

    """

    def __init__(self, scale_factor=1.0, num_parallel_calls=tf.data.experimental.AUTOTUNE, augment_label=False):
        super(RescaleIntensities, self).__init__(num_parallel_calls, augment_label)
        self.scale_factor = scale_factor

    def transform_fn(self, batch_level=False):
        """Returns Tensorflow function which rescales image intensities

        Args:
            batch_level (bool): if True, returned function processes image batches, otherwise individual images

        """

        @tf.function
        def tf_rescale_intensities(images, factor):
            out_images = tf.cast(images, tf.float32)
            return factor * out_images / (tf.math.reduce_max(out_images) + 1.0e-7)

        if self.augment_label:
            return tf_duplicated_transform_fn(tf_rescale_intensities, True, batch_level, None, self.scale_factor)
        else:
            return tf_apply_transform_fn(tf_rescale_intensities, True, batch_level, None, self.scale_factor)


class BoxBlur(Augmentor):
    """
    Random box blurring. The resulting image is of type tf.float32, but keeps original intensity value range

    Args:
        max_box_size (int) : maximum size of the box filter
        prob (float) : probability of this transformation in the range [0, 1]

    Attributes:
        max_box_size (int) : maximum size of the box filter
        prob (float) : probability of this transformation in the range [0, 1]

    """

    def __init__(self, max_box_size, prob=0.5, num_parallel_calls=tf.data.experimental.AUTOTUNE, augment_label=False):
        super(BoxBlur, self).__init__(num_parallel_calls, augment_label)
        self.max_box_size = max_box_size
        self.prob = prob

    def transform_fn(self, batch_level=False):
        """Returns Tensorflow function which performs random box blur

        Args:
            batch_level (bool): if True, returned function processes image batches, otherwise individual images

        """
        if self.augment_label:
            return tf_duplicated_transform_fn(tf_box_blur, False, batch_level, tf.float32, self.max_box_size, self.prob)
        else:
            return tf_apply_transform_fn(tf_box_blur, False, batch_level, tf.float32, self.max_box_size, self.prob)


class GaussianBlur(Augmentor):
    """
    Random Gaussian blurring. The resulting image is of type tf.float32, but keeps original intensity value range

    Args:
        sigma (tuple) : range of sigma values
        prob (float) : probability of this transformation in the range [0, 1]

    Attributes:
        sigma (tuple) : range of sigma values
        prob (float) : probability of this transformation in the range [0, 1]

    """
    def __init__(self, sigma, prob=0.5, num_parallel_calls=tf.data.experimental.AUTOTUNE, augment_label=False):
        super(GaussianBlur, self).__init__(num_parallel_calls, augment_label)
        self.sigma = sigma
        self.prob = prob

    def transform_fn(self, batch_level=False):
        """Returns Tensorflow function which performs random gaussian blurring

        Args:
            batch_level (bool): if True, returned function processes image batches, otherwise individual images

        """
        if self.augment_label:
            return tf_duplicated_transform_fn(tf_gaussian_blur, False, batch_level, tf.float32, self.sigma, self.prob)
        else:
            return tf_apply_transform_fn(tf_gaussian_blur, False, batch_level, tf.float32, self.sigma, self.prob)


class DivisibleCrop(Augmentor):
    """
    Crops the largest central image region such that its sides are divisible by the given factors

    Args:
        div_factors (int or tuple) : divisors, either single integer or tuple (height, width)

    Attributes:
        div_factors (int or tuple) : divisors, either single integer or tuple (height, width)

    """
    def __init__(self, div_factors, num_parallel_calls=tf.data.experimental.AUTOTUNE, augment_label=False):
        super(DivisibleCrop, self).__init__(num_parallel_calls, augment_label)
        self.div_factors = init_size(div_factors)

    def transform_fn(self, batch_level=False):
        """Returns Tensorflow function which performs divisible cropping

        Args:
            batch_level (bool): if True, returned function processes image batches, otherwise individual images

        """
        if self.augment_label:
            return tf_duplicated_transform_fn(tf_divisible_crop, False, batch_level, None,
                                              self.div_factors[0], self.div_factors[1])
        else:
            return tf_apply_transform_fn(tf_divisible_crop, False, batch_level, None,
                                         self.div_factors[0], self.div_factors[1])


class DivisiblePad(Augmentor):
    """
    Pads the image (each side equally) such that the sides are divisible by the given factors

    Args:
        div_factors (int or tuple) : divisors, either single integer or tuple (height, width)
        mode (str): "REFLECT" (default), "CONSTANT" (0), "SYMMETRIC"

    Attributes:
        div_factors (int or tuple) : see above
        mode (str): see above

    """
    def __init__(self, div_factors, mode="REFLECT", num_parallel_calls=tf.data.experimental.AUTOTUNE,
                 augment_label=False):

        super(DivisiblePad, self).__init__(num_parallel_calls, augment_label)
        self.div_factors = init_size(div_factors)
        self.mode = mode

    def transform_fn(self, batch_level=False):
        """Returns Tensorflow function which performs divisible padding

        Args:
            batch_level (bool): if True, returned function processes image batches, otherwise individual images

        """
        if self.augment_label:
            return tf_duplicated_transform_fn(tf_divisible_pad, False, batch_level, None,
                                              self.div_factors[0], self.div_factors[1], self.mode)
        else:
            return tf_apply_transform_fn(tf_divisible_pad, False, batch_level, None,
                                         self.div_factors[0], self.div_factors[1], self.mode)


class CenterPad(Augmentor):
    """
    Applies center padding of the image (each side).
    Argument `size` can be single integer or a tuple: (`height`, `width`). If single integer, image is extended
    by `size` pixels to the left, right, top and bottom, respectively. If `size` is tuple, `height` values is used to
    pad image from the top and below, and `width` from left and right.

    Args:
        size (int or tuple) : either single integer or tuple: (`pad_height`, `pad_width`)
        mode (str): "REFLECT" (default), "CONSTANT" (0), "SYMMETRIC"

    Attributes:
        size (int or tuple) : see above
        mode (str): see above

    """
    def __init__(self, size, mode="REFLECT", num_parallel_calls=tf.data.experimental.AUTOTUNE,
                 augment_label=False):

        super(CenterPad, self).__init__(num_parallel_calls, augment_label)
        self.factors = init_size(size)
        self.factors = init_size(size)
        self.mode = mode

    def transform_fn(self, batch_level=False):
        """Returns Tensorflow function which performs center padding

        Args:
            batch_level (bool): if True, returned function processes image batches, otherwise individual images

        """
        if self.augment_label:
            return tf_duplicated_transform_fn(tf_center_pad, False, batch_level, None,
                                              self.factors[0], self.factors[1], self.mode)
        else:
            return tf_apply_transform_fn(tf_center_pad, False, batch_level, None,
                                         self.factors[0], self.factors[1], self.mode)


class Sequential(object):
    """
    Main class for building sequential image augmentation pipeline.

    Args:
        ops_sequence (list) : a list of augmentor objects (parent class Augmentor)
        batch_level (bool) : set True to augment batches, False for augmenting individual images

    Attributes:
        ops (list) : a list of augmentor objects
        batch_level (bool) : (see above)

    Examples:

        # Example for classification problem
        --------------------------------------------------------------------------------
        classifier_batch_augmentor = Sequential([
            Affine(rotation=15.0, scale=(0.8, 1.2)),  # random rotation -15 to 15 degrees and scaling
            RandomCrop(size=(45, 45)),  # crop random region of size (120, 100)
            CenterCrop(size=(40, 40)),  # crop central region of size (80, 60)
            BrightnessContrast(brightness=(-0.1, 0.1), contrast=(0.9, 1.1)),  # adjust brightness and contrast
            NormalizeMeanStd()
        ], batch_level=True)

        # ... Load / Create tf.Data.Dataset
        dataset = dataset.batch(32)
        dataset = classifier_batch_augmentor.update(dataset)  # applied to batch
        # ... Other operations

        --------------------------------------------------------------------------------

        # Example for detection problem. Note that attribute `augment_label`, which means that the same transformation
        # is applied for image and mask
        detector_batch_augmentor = ia.Sequential([
            ia.RandomCrop(size=(80, 150), augment_label=True),  # random crop region 80 x 150 pixels
            ia.Affine(rotation=10.0, scale=(0.8, 1.2), augment_label=True),  # perform affine random transformation
            ia.CenterCrop(size=(50, 120), augment_label=True),  # crop central region
            ia.RescaleIntensities(augment_label=True), # rescale intensities for image and mask to [0, 1]
            ia.BrightnessContrast(brightness=0.1, contrast=(0.9, 1.1)), # augment brightness and contrast (image only)
            ia.NormalizeMeanStd() # normalize image only
        ])

        # ... Load / Create tf.Data.Dataset
        dataset = detector_batch_augmentor.update(dataset)  # applied to individual images
        dataset = dataset.batch(32)  # prepare batches
        # ... Other operations

    """
    def __init__(self, ops_sequence, batch_level=False):
        self.ops = ops_sequence
        self.batch_level = batch_level

    def update(self, dataset):
        """Update tf.Data.Dataset with the defined augmentation pipeline

        Returns:
            tf.data.Dataset : pipeline with augmentation steps
        """
        for op in self.ops:
            dataset = op.augment(dataset, self.batch_level)
        return dataset

    @staticmethod
    def from_dict(operations):
        """
        Create augmentation pipeline from list of dictionaries.

        Format of input:
            [
                ["op1", {"param1": val1, "param2": val2, ...}],
                ["op2", {"param1": val1, "param2": val2, ...}]
                ...
                ["op"]  # Operation without parameters
            ]

        Example:
            [
                ["Affine", {"rotation": 3.0, "scale": [0.95, 1.05]}],
                ["CenterCrop", {"size":  [70, 440]}],
                ["NormalizeMeanStd"]
            ]

        Returns:
            result (Sequential) object

        """

        op_objects = []
        for operation in operations:
            op_name, op_args = operation[0], operation[1] if len(operation) > 1 else None

            op_class = getattr(sys.modules[__name__], op_name)
            op_objects.append(op_class(**op_args) if op_args else op_class())

        return Sequential(op_objects)


class SequentialGPU(object):
    def __init__(self, ops_sequence, batch_level=False):
        self.ops = ops_sequence
        self.fns = [op.transform_fn(batch_level) for op in self.ops]
        self.batch_level = batch_level

    def eval(self, *args):
        """Update tf.Data.Dataset with the defined augmentation pipeline

        Returns:
            tf.data.Dataset : pipeline with augmentation steps
        """
        data = args
        for fn in self.fns:
            data = fn(*data)
        return data
