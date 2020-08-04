import gzip
import os

import numpy as np
from mxnet.gluon.data.vision import transforms

from source.onnx_interface.model_scoring import write_numpy_array
from source.onnx_interface.onnx_if import input_pb_name, output_pb_name


def createImageNetValidationPB(model_dir, test_dir, imagenet_dir, ):
    test_path = os.path.join(model_dir, test_dir)

    image_path = os.path.join(imagenet_dir, 'val/val/n01514668/ILSVRC2012_val_00004550.JPEG')

    input_test_data = os.path.join(test_path, '%s' % input_pb_name)
    output_test_data = os.path.join(test_path, output_pb_name)

    # Load and reshape images
    from skimage.transform import resize
    import numpy

    from matplotlib import pyplot as plt

    ximg = plt.imread(image_path)
    plt.imshow(ximg)
    #    plt.show()
    print(type(ximg), ximg.shape, ximg.dtype)
    print(ximg.shape)
    ximg224 = resize(ximg / 255, (224, 224, 3), anti_aliasing=True)
    ximg = ximg224[numpy.newaxis, :, :, :]
    ximg = ximg.astype(numpy.float32)
    print(ximg.shape)
    # Create protobuf
    if not os.path.exists(os.path.join(test_path)):
        os.mkdir(os.path.join(test_path))

    write_numpy_array(input_test_data, ximg)


def createMNISTValidationPB(test_dir, mnist_dir, ):
    test_path = test_dir

    val_set = 't10k-images-idx3-ubyte.gz'
    val_label = 't10k-labels-idx1-ubyte.gz'

    input_test_data = os.path.join(test_path, '%s' % input_pb_name)
    output_test_data = os.path.join(test_path, output_pb_name)

    # Extract and format mnist validation data as numpy
    images = gzip.open(os.path.join(mnist_dir, val_set), 'r')
    labels = gzip.open(os.path.join(mnist_dir, val_label), 'r')

    image_size = 28
    num_images = 10000

    # Load images
    images.read(16)
    buf = images.read(image_size * image_size * num_images)
    images_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    images_data = images_data.reshape(num_images, 1, 1, image_size, image_size)
    assert (images_data.shape == (10000, 1, 1, 28, 28))
    assert (images_data.dtype == np.float32)
    # showInputMnist(images_data[0:15].reshape(15, 1, 28, 28))

    # Load labels
    labels.read(8)
    buf = labels.read(num_images)
    labels_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    labels_data = labels_data.reshape(num_images)
    assert (labels_data[0] == 7)
    assert (labels_data[1] == 2)
    assert (labels_data[2] == 1)

    # Create protobuf
    if not os.path.exists(os.path.join(test_path)):
        os.mkdir(os.path.join(test_path))

    write_numpy_array(input_test_data, images_data)
    write_numpy_array(output_test_data, labels_data)


# Pre-processing function for ImageNet models
def preprocess(img):
    '''
    Preprocessing required on the images for inference with mxnet gluon
    The function takes path to an image and returns processed tensor
    '''
    transform_fn = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform_fn(img)
    img = img.expand_dims(axis=0)  # batchify

    return img
