import os

import torch
from torch import nn
import torch.nn.functional as F

import brevitas
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
import brevitas.onnx as bo

class Net(nn.Module):   
	def __init__(self):
		super(Net, self).__init__()


		# Defining a 2D convolution layer
		self.conv1 = qnn.QuantConv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1, 
								 weight_quant_type=QuantType.INT, weight_bit_width=2, bias=False)

		self.relu1 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=2, max_val=6)

		# Defining another 2D convolution layer
		self.conv2 = qnn.QuantConv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1,
								 weight_quant_type=QuantType.INT, weight_bit_width=2,bias=False)

		self.relu2 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=2, max_val=6)


		self.fc1   = qnn.QuantLinear(in_features=4*7*7, out_features=10, bias=False, 
								weight_quant_type=QuantType.INT,  weight_bit_width=2)

	# Defining the forward pass    
	def forward(self, x):
		out = self.conv1(x)
		#out = nn.BatchNorm2d(out, num_features=4)
		out = self.relu1(out)
		out = F.max_pool2d(out, kernel_size=2, stride=2)

		out = self.conv2(out)
		#out = nn.BatchNorm2d(out, num_features=4)
		out = self.relu2(out)
		out = F.max_pool2d(out, kernel_size=2, stride=2)

		out = out.view(out.size(0), -1)
		out = self.fc1(out)

		return out


net = Net()
# print(net.relu2.get_exportable_quantization_type())







model_path = "models/"
if not os.path.exists(model_path):
	os.makedirs(model_path)
# submission = pd.DataFrame(final_prediction, dtype=int, columns=['ImageId', 'Label'])
# submission.to_csv(os.path.join(model_path, 'pytorch_LeNet.csv'), index=False, header=True)

print('6. Converting to ONNX')

net.eval()

# Input to the model
batch_size = 1
use_gpu = False
if use_gpu:
	x = torch.randn(batch_size, 1, 28, 28, requires_grad=True).cuda()
else:
	x = torch.randn(batch_size, 1, 28, 28, requires_grad=True)

torch_out = net(x)


bo.export_finn_onnx(net, (batch_size, 1, 28, 28), model_path+"/test.onnx")


# class Net(Module):   
# 	def __init__(self):
# 		super(Net, self).__init__()

# 		self.cnn_layers = Sequential(
# 			# Defining a 2D convolution layer
# 			Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
# 			BatchNorm2d(4),
# 			ReLU(inplace=True),
# 			MaxPool2d(kernel_size=2, stride=2),
# 			# Defining another 2D convolution layer
# 			Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
# 			BatchNorm2d(4),
# 			ReLU(inplace=True),
# 			MaxPool2d(kernel_size=2, stride=2),
# 		)

# 		self.linear_layers = Sequential(
# 			Linear(4 * 7 * 7, 10)
# 		)

# 	# Defining the forward pass    
# 	def forward(self, x):
# 		x = self.cnn_layers(x)
# 		x = x.view(x.size(0), -1)
# 		x = self.linear_layers(x)
# 		return x
