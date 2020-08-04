import os

import onnx

from source.onnx_interface.model_scoring import scoreMnist

model_path = "models/pytorch_LeNet/"
model_path = os.path.join(model_path, "model.onnx")


def check_model():
    print('7. check ONNX model')
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)


def score_model():
    print('8. test ONNX model')

    print(scoreMnist(model_path))


if __name__ == "__main__":
    check_model()
    score_model()
