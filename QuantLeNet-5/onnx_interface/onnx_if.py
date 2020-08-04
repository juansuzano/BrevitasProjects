import onnx
from onnx import numpy_helper

input_pb_name = 'input_0.pb'
output_pb_name = 'output_0.pb'


def get_tensor(node, onnx_model):
    tensor = [lay for lay in onnx_model.graph.initializer if lay.name == node.input[1]][0]
    return tensor


def get_node(layer_name, onnx_model):
    node = [node for node in onnx_model.graph.node if node.name == layer_name][0]
    return node


def get_weights_by_layer_name(onnx_model, layer_name):
    node = get_node(layer_name, onnx_model)
    tensor = get_tensor(node, onnx_model)
    return numpy_helper.to_array(tensor)


def get_dim_by_layer_name(onnx_model, layer_name):
    node = get_node(layer_name, onnx_model)
    tensor = get_tensor(node, onnx_model)
    return numpy_helper.to_array(tensor).shape


def set_weights_by_layer_name(onnx_model, layer_name, np_array):
    node = get_node(layer_name, onnx_model)
    tensor = get_tensor(node, onnx_model)
    assert(numpy_helper.to_array(tensor).shape == np_array.shape)
    tensor.raw_data = numpy_helper.from_array(np_array).raw_data


def set_model(path, onnx_model):
    onnx.save(onnx_model, path)


def get_model(model_path, force_rename=False, verbose=True):
    model = onnx.load(model_path)
    init = model.graph.initializer
    onnx.checker.check_model(model)

    # Model shape
    input_shape = get_input_shape(model)
    output_shape = get_output_shape(model)
    print(
        "model {} has {} layers, input_shape:{}, output_shape:{}".format(
            model_path,
            len(model.graph.node),
            input_shape,
            output_shape
        )
    )

    # Label layers if required
    if force_rename or '' in [lay.name for lay in model.graph.node]:
        i = 1
        for lay in model.graph.node:
            lay.name = '{}{}'.format(lay.op_type, i)
            i += 1

    if verbose:
        pretty_print_model_layers(model)

    # Get conv and fc layer list
    return model, get_layers(model)


def get_layers(model):
    conv_lay = [lay.name for lay in model.graph.node if lay.op_type == 'Conv' or lay.op_type == 'Gemm']
    return conv_lay


# Print
def pretty_print_model_layers(model):
    from tabulate import tabulate
    import pandas as pd

    # Extract data from model
    model_data = []
    Conv_weight_count = 0
    FC_weight_count = 0

    Conv_layer_count = 0
    FC_layer_count = 0
    for lay in model.graph.node:
        # Extract data from graph_node
        lay_name = lay.name
        lay_op = lay.op_type

        # infer layer shape and size
        input = get_input(model, lay.input[1]) if len(lay.input) > 1 else None
        dim = get_dim(input) if input else ""
        size = get_layer_size(model, lay_name) if lay_op == "Conv" or lay_op == "Gemm" else ""

        # Update internal var
        if lay_op == "Conv":
            Conv_layer_count += 1
            Conv_weight_count += size
        elif lay_op == "Gemm":
            FC_layer_count += 1
            FC_weight_count += size

        model_data.append([lay_name, lay_op, dim, size])

    total_weight_count = Conv_weight_count + FC_weight_count
    for a in model_data:
        a.append("{:.2f}%".format(a[-1] / total_weight_count * 100) if a[-1] else '')

    df = pd.DataFrame(model_data)
    print(tabulate(df, headers=[
        "ID",
        "layer_name",
        "operation",
        "shape",
        "weight_count",
        "weight_part"], tablefmt="psql"))
    print(
        "Total weight count: {0}\n"
        "CONV: {1} weights ({2:.2f}%) in {3} layers\n"
        "FC: {4} weights ({5:.2f}%) in {6} layers\n".format(
            total_weight_count,
            Conv_weight_count,
            Conv_weight_count / total_weight_count * 100 if Conv_layer_count else 0.,
            Conv_layer_count,
            FC_weight_count,
            FC_weight_count / total_weight_count * 100 if FC_layer_count else 0.,
            FC_layer_count,
        )
    )


def get_model_shape(model):
    model_shapes = [(input.name, get_dim(input.name)) for input in model.graph.input]
    return model_shapes


def get_dim(input):
    return tuple([i.dim_value for i in input.type.tensor_type.shape.dim if input.type.tensor_type.HasField("shape")])


def get_input(model, input_name):
    result = [i for i in model.graph.input if i.name == input_name]
    if len(result) == 1:
        return result[0]
    else:
        return None


def get_input_shape(model):
    input = get_input(model, model.graph.node[0].input[0])
    if input:
        return get_dim(input)
    else:
        return input


def get_output_shape(model):
    output = get_input(model, model.graph.node[-1].input[-1])
    if output:
        return get_dim(output)
    else:
        return output


def get_layer_size(onnx_model, layer):
    return get_weights_by_layer_name(onnx_model, layer).size


def get_largest_layers(onnx_model, layers, proportion):
    # Get size of layers
    layers_size = [get_layer_size(onnx_model, layer) for layer in layers]
    threshold = proportion * sum(layers_size)

    # Sort and select largest layers to have at least the proportion required
    layers_size, layers = zip(*sorted(zip(layers_size, layers), reverse=True))
    for i, _ in enumerate(layers_size):
        if sum(layers_size[:i + 1]) >= threshold:
            return layers[:i + 1], sum(layers_size[:i + 1]) / sum(layers_size)
