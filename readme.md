Anyline Python ML Tools
----

Python helper scripts for ML tasks. 
This version contains a single subpackage:
 * anyline_mltools.augment
 
for building efficient data augmentation pipelines based on `tf.data` subpackage.

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the package.
```
pip install git+https://github.com/Anyline/anyline-ml-tools.git
```


## Requirements

* tensorflow_gpu >= 2.1
* tensorflow_addons 
* numpy


## Usage

Here we provide examples of `anyline_mltools.augment` for image augmentation.


###### Example 1. Augmenting classification data:


```python
import anyline_mltools.augment as ia

...

# Load dataset 
dataset = tf.data.TFRecordDataset("/path/to/tfrecord/dataset")
dataset = tf.data.map(decode_tfrecord)

# Example for classification problem
batch_augmentor = ia.Sequential([
    # random rotation -15 to 15 degrees and scaling
    ia.Affine(rotation=15.0, scale=(0.8, 1.2)),  
    
    # crop random region of size (45, 45)
    ia.RandomCrop(size=(45, 45)),  

    # crop central region of size (40, 40)
    ia.CenterCrop(size=(40, 40)),  

    # adjust brightness and contrast
    ia.BrightnessContrast(brightness=(-0.1, 0.1), contrast=(0.9, 1.1)),  
    
    # normalize image by subtracting mean and dividing by standard deviation    
    ia.NormalizeMeanStd()
], batch_level=True)  # apply for a batch - should be called after dataset.batch(...)


dataset = dataset.batch(32)  # create batches of size 32
dataset = batch_augmentor.update(dataset)  # applied to batch 

... 

```

###### Example 2. Augmenting detection data:

```python
import anyline_mltools.augment as ia

...

# Load dataset 
dataset = tf.data.TFRecordDataset("/path/to/tfrecord/dataset")
dataset = tf.data.map(decode_tfrecord)

# Example for detection problem. 
# Note that `augment_label=True` - same transformation is applied for image and mask
detector_batch_augmentor = ia.Sequential([
    # random crop region 80 x 150 pixels
    ia.RandomCrop(size=(80, 150), augment_label=True),  
   
    # perform affine random transformation
    ia.Affine(rotation=10.0, scale=(0.8, 1.2), augment_label=True),  

    # crop central region
    ia.CenterCrop(size=(50, 120), augment_label=True),  

    # rescale intensities for image and mask to [0, 1]
    ia.RescaleIntensities(augment_label=True), 

    # augment brightness and contrast (image only)
    ia.BrightnessContrast(brightness=0.1, contrast=(0.9, 1.1)), 

    # normalize image by subtracting mean and dividing by standard deviation  
    ia.NormalizeMeanStd() 
])  # in this case batch_level=False

dataset = detector_batch_augmentor.update(dataset)  # augment individual images
dataset = dataset.batch(32)  # prepare batches

...

```


The following augmentation operations are implemented:

* RandomCrop;
* CenterCrop;
* Affine;
* NormalizeMeanStd;
* BrightnessContrast;
* GaussianNoise;
* RescaleIntensities;
* BoxBlur;
* GaussianBlur;
* DivisibleCrop;
* DivisiblePad;
