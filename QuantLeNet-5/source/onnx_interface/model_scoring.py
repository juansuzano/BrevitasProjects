import json
import math
import os
import subprocess

import numpy as np
import onnx
import onnxruntime
import tabulate
from matplotlib import pyplot as plt
from onnx import numpy_helper

from source.onnx_interface.onnx_if import input_pb_name, output_pb_name, get_model, get_input, get_dim, get_layers, \
    get_layer_size, get_weights_by_layer_name
from source.utils import pool_map_bar, toc


def score_imagenet(model_path, approx_model_dir):
    if type(model_path) is not list:
        model_path = [model_path]

    # Run inference in docker
    result = pool_map_docker_inference(approx_model_dir, model_path)

    # Retrieve results
    return result


def pool_map_docker_inference(approx_model_dir, model_path):
    num_docker = 2
    args = [[approx_model_dir, model_path[i], 'res{}.json'.format(i)] for i in range(len(model_path))]
    return pool_map_bar(run_inference_docker, args, num_docker)


def run_inference_docker(args):
    assert (len(args) == 3)
    approx_model_dir = args[0]
    model_path = args[1]
    log_name = args[2]
    FNULL = open(os.devnull, 'w')

    #  Try inference
    tic = toc()
    model_name = os.path.split(model_path)[-1]
    try:
        subprocess.check_call(['./dockerrun.sh', model_name, log_name, approx_model_dir],
                              stdout=FNULL)
    except:
        print("error for network: ", model_name)
        return -1, toc(tic), model_name

    with open(os.path.join(approx_model_dir, log_name), 'r') as res_file:
        res_file = json.loads(res_file.read())
        return res_file['top1'], toc(tic), model_name


def scoreImageNet(model_dir, model_name, test_dir):
    # Load data
    input_test_data = os.path.join(model_dir, test_dir, input_pb_name)
    output_test_data = os.path.join(model_dir, test_dir, output_pb_name)

    input_val = read_pb(input_test_data)
    # output_val = read_pb(output_test_data)

    # Load model
    session = onnxruntime.InferenceSession(os.path.join(model_dir, model_name))

    # Inference
    output_name = session.get_outputs()[0].name
    input_name = session.get_inputs()[0].name

    # Reshape if needed
    if input_val.shape != session.get_inputs()[0].shape:
        input_val = np.transpose(input_val, (0, 3, 1, 2))
    assert (input_val.shape != session.get_inputs()[0].shape)

    # Run inference
    result = session.run([output_name], {input_name: input_val})
    print(np.array(result).shape)
    print(np.argmax(result), np.max(result))
    print(np.array(result).squeeze().argsort()[-5:][::-1])
    top5 = [a for a in np.array(result).squeeze().argsort()[-5:][::-1]]
    print([[a, result[0][0][a]] for a in top5])

    # Format
    with open('datasets/imagenet/imagenet_class_index.json') as f:
        CLASS_INDEX = json.load(f)
    pred = np.array(result).squeeze()
    top_indices = pred.argsort()[-5:][::-1]
    res = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
    res.sort(key=lambda x: x[2], reverse=True)
    print(tabulate.tabulate(res, headers=["class_id", "name", "P"]))

    # print(np.argmax(output_val), np.max(output_val))

    # Show img
    # print(input_val.shape)
    plt.imshow(np.array(np.transpose(input_val, (0, 2, 3, 1)).squeeze(), dtype=float), cmap=plt.cm.gray)
    plt.show()


def scoreMnist(model_path, test_dir="models/mnist/test_data_set_val"):
    # Load data
    input_test_data = os.path.join(test_dir, input_pb_name)
    output_test_data = os.path.join(test_dir, output_pb_name)

    input_val = read_pb(input_test_data)
    output_val = read_pb(output_test_data)

    # Load model
    session = onnxruntime.InferenceSession(model_path)

    # Inference
    output_name = session.get_outputs()[0].name
    input_name = session.get_inputs()[0].name

    result = []
    top1 = []
    i = 0
    for input_data in input_val:
        result.append(session.run([output_name], {input_name: input_data}))
        top1.append(int(np.argmax(np.array(result[i]).squeeze(), axis=0)))
        i += 1
    top1 = np.array(top1)
    assert (top1.shape == output_val.shape)
    accuracy = np.count_nonzero(top1 == output_val) / len(output_val)
    return accuracy


def read_pb(filename):
    read_tensor = onnx.TensorProto()
    with open(filename, 'rb') as f:
        read_tensor.ParseFromString(f.read())
    return numpy_helper.to_array(read_tensor)


def write_numpy_array(filename, numpy_array):
    write_tensor = numpy_helper.from_array(numpy_array)
    with open(filename, 'wb') as f:
        f.write(write_tensor.SerializeToString())


def showInputMnist(test_inputs):
    plt.figure(figsize=(16, 6))
    for test_image in np.arange(len(test_inputs)):
        plt.subplot(1, 15, test_image + 1)
        plt.axhline('')
        plt.axvline('')
        plt.imshow(test_inputs[test_image].reshape(28, 28), cmap=plt.cm.Greys)
    plt.show()


def test_inference():
    model_path_list = [os.path.join("models", model, "model.onnx") for model in model_list]
    temp_approx = os.path.join("temp_approx")

    import shutil
    for path, model_name in zip(model_path_list, model_list):
        print(model_name)
        shutil.copyfile(path, os.path.join(temp_approx, model_name))

    res = score_imagenet(model_list, temp_approx)
    print(list(res))
    for item in zip(res):
        print(item)


model_list = [
    "inception_v1",
    "inception_v2",
    "bvlc_googlenet",
    "mobilenetv2-1.0",
    "densenet121",
    "squeezenet1.1",
    "shufflenet",
    "shufflenet_v2",
    "zfnet512",
    "bvlc_reference_caffenet",
    "bvlc_alexnet",
    "vgg16",
    "vgg19",
    "resnet18v2",
    "resnet34v2",
    "resnet50v2",
    "resnet101v2",
    "resnet152v2",
]


def test_shape():
    model_shape_list = []
    for model_name in model_list:
        print(model_name)
        model = get_model("models/{}/model.onnx".format(model_name))[0]
        for input in [
            get_input(model, model.graph.node[0].input[-1]),
            get_input(model, model.graph.node[0].input[0]),
            model.graph.input[0],
            get_input(model, model.graph.node[-1].input[-1]),
            get_input(model, model.graph.node[-1].input[0]),
            get_input(model, model.graph.node[-1].output[0])
        ]:
            try:
                print(model_name, get_dim(input))
            except:
                print("error", input)

        assert get_dim(get_input(model, model.graph.node[0].input[0])) == (1, 3, 224, 224)


def get_memory(onnx_model, compressed_layers=None):
    layers = get_layers(onnx_model)
    memory_by_layer = {}

    if compressed_layers:
        layers = list(set(layers) - set(compressed_layers))

    # Calculate memory for non-compressed layers
    sum_memory_uncompressed = 0
    sum_weights_count_uncompressed = 0
    for layer in layers:
        layer_size = get_layer_size(onnx_model, layer)
        memory_by_layer[layer] = {
            "total": layer_size * 32,
            "weight_count": layer_size,
        }
        sum_memory_uncompressed += layer_size * 32
        sum_weights_count_uncompressed += layer_size

    # Calculate memory for compressed layers
    sum_memory_compressed = 0
    sum_weights_count_compressed = 0
    sum_value_matrix = 0
    sum_index_matrix = 0
    if compressed_layers:
        for layer in compressed_layers:
            layer_weights = get_weights_by_layer_name(onnx_model, layer)
            cluster_count = len(np.unique(layer_weights))
            index_bit_count = math.ceil(math.log2(float(cluster_count)))
            value_matrix = cluster_count * 32
            layer_size = layer_weights.size
            index_matrix = index_bit_count * layer_size
            memory_by_layer[layer] = {
                "value_matrix": value_matrix,
                "index_matrix": index_matrix,
                "total": value_matrix + index_matrix,
                "weight_count": layer_size,
                "cluster_count": cluster_count
            }
            sum_memory_compressed += value_matrix + index_matrix
            sum_weights_count_compressed += layer_size
            sum_value_matrix += value_matrix
            sum_index_matrix += index_matrix

    # Save data
    memory_by_layer["total memory original"] = sum_memory_uncompressed
    memory_by_layer["total memory compressed"] = sum_memory_compressed
    memory_by_layer["total"] = sum_memory_uncompressed + sum_memory_compressed

    memory_by_layer["weight count original"] = sum_weights_count_uncompressed
    memory_by_layer["weight count compressed"] = sum_weights_count_compressed
    memory_by_layer["weight count original"] = sum_weights_count_uncompressed + sum_weights_count_compressed

    memory_by_layer["total memory values"] = sum_value_matrix
    memory_by_layer["total memory indexes"] = sum_index_matrix

    return memory_by_layer
