# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Validation script for Imagenet models
# 
# ## Overview
# Use this notebook to verify the accuracy of a trained ONNX model on the validation set of ImageNet dataset.
# 
# ## Models Support in This Demo
# 
# * SqueezeNet
# * VGG
# * ResNet
# * MobileNet
# 
# ## Prerequisites
# Dependencies:
# * Protobuf compiler - `sudo apt-get install protobuf-compiler libprotoc-dev` (required for ONNX. This will work for any linux system. For detailed installation guidelines head over to [ONNX documentation](https://github.com/onnx/onnx#installation))
# * ONNX - `pip install onnx`
# * MXNet - `pip install mxnet-cu90mkl --pre -U` (tested on this version GPU, can use other versions. `--pre` indicates a pre build of MXNet which is required here for ONNX version compatibility. `-U` uninstalls any existing MXNet version allowing for a clean install)
# * numpy - `pip install numpy`
# * matplotlib - `pip install matplotlib`
# * gluoncv - `pip install gluoncv` (for ImageNet data preparation)
# 
# In order to do validate accuracy with a python script: 
# * Generate the script : In Jupyter Notebook browser, go to File -> Download as -> Python (.py)
# * Run the script: `python imagenet_validation.py`
# 
# The ImageNet dataset must be downloaded and extracted in the required directory structure. Refer to the guidelines in [imagenet_prep](imagenet_prep.md).

import multiprocessing
import sys
from collections import namedtuple

# %%
import mxnet as mx
from gluoncv.data import imagenet
from mxnet import gluon
from mxnet.contrib.onnx.onnx2mx.import_model import import_model
from mxnet.gluon.data.vision import transforms

# %% [markdown]
# ### Set context, paths and parameters

# %%
# Determine and set context
if len(mx.test_utils.list_gpus()) == 0:
    ctx = [mx.cpu()]
else:
    ctx = [mx.gpu(0)]

# path to imagenet dataset folder
data_dir = '/datasets'

# batch size (set to 1 for cpu)
batch_size = 128

# number of preprocessing workers
num_docker = 2
num_workers = int(multiprocessing.cpu_count() / num_docker)

# path to ONNX model file
model_path = '/run/' + sys.argv[1]
print(sys.argv[2])
print(sys.argv[1])
print(model_path)
# %% [markdown]
# ### Import ONNX model
# Import a model from ONNX to MXNet symbols and params using `import_model`

# %%
sym, arg_params, aux_params = import_model(model_path)

# %% [markdown]
# ### Define evaluation metrics
# top1 and top 5 accuracy

# %%
# Define evaluation metrics
acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)

# %% [markdown]
# ### Preprocess images
# For each image-> resize to 256x256, take center crop of 224x224, normalize image

# %%
# Define image transforms
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

# Load and process input
val_data = gluon.data.DataLoader(
    imagenet.classification.ImageNet(data_dir, train=False).transform_first(transform_test),
    batch_size=batch_size, shuffle=False, num_workers=num_workers)

# %% [markdown]
# ### Load network for validation
# Use `mx.mod.Module` to define the network architecture and bind the parameter values using `mod.set_params`. `mod.bind` tells the network the shape of input and labels to expect.

# %%
# Load module
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))],
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)

# %% [markdown]
# ### Compute evaluations
# Perform forward pass over each batch and generate evaluations

# %%
# Compute evaluations
Batch = namedtuple('Batch', ['data'])
acc_top1.reset()
acc_top5.reset()
num_batches = int(50000 / batch_size)
print('[0 / %d] batches done' % (num_batches))
# Loop over batches
for i, batch in enumerate(val_data):
    # Load batch
    data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
    label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
    # Perform forward pass
    mod.forward(Batch([data[0]]))
    outputs = mod.get_outputs()
    # Update accuracy metrics
    acc_top1.update(label, outputs)
    acc_top5.update(label, outputs)
    if (i + 1) % 50 == 0:
        print('[%d / %d] batches done' % (i + 1, num_batches))

# %% [markdown]
# ### Print results
# top1 and top5 accuracy of the model on the validation set are shown in the output

# %%
# Print results
_, top1 = acc_top1.get()
_, top5 = acc_top5.get()
print("Top-1 accuracy: {}, Top-5 accuracy: {}".format(top1, top5))

import json
print(sys.argv[2])
with open('/run/' + sys.argv[2], 'w') as result_file:
    result_file.write(json.dumps({'top1': top1, 'top5': top5}, sort_keys=True, indent=4))
