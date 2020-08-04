from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):
	def __init__(self):
		super(LeNet, self).__init__()
		self.fc1 = nn.Linear(2, 2)
		self.fc2 = nn.Linear(2, 1)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features


net = LeNet()
print(net)
model_path = ''
x = torch.randn(1, 1, 1, 2, requires_grad=True)

torch.onnx.export(
    net,  # model being run
    x,  # model input (or a tuple for multiple inputs)
    os.path.join(model_path, "model.onnx"),
    # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=10,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
)