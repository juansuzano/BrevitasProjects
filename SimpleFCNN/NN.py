import os
import torch
from torch import nn

import brevitas.nn as qnn
from brevitas.core.quant import QuantType
#import torch.onnx

import brevitas.onnx as bo

class SimpleNN(nn.Module):
	def __init__(self):
		super(SimpleNN, self).__init__()

		self.fc1   = qnn.QuantLinear(2, 2, bias=False, 
									 weight_quant_type=QuantType.INT, 
									 weight_bit_width=2)
		self.relu1 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=2, max_val=6)
		self.fc2   = qnn.QuantLinear(2, 1, bias=False, 
									 weight_quant_type=QuantType.INT, 
									 weight_bit_width=2)
		self.relu2 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=2, max_val=6)


	def forward(self, x):
		out = self.relu1(self.fc1(x))
		out = self.relu2(self.fc2(out))
		return out



net = SimpleNN()
print(net)

use_gpu = torch.cuda.is_available()
if use_gpu:
	net = net.cuda()
	print('USE GPU')
else:
	print('USE CPU')

model_path = "models/"
if not os.path.exists(model_path):
	os.makedirs(model_path)


if use_gpu:
    x = torch.randn(1, 1, 1, 2, requires_grad=True).cuda()
else:
    x = torch.randn(1, 1, 1, 2, requires_grad=True)

print(x)
torch_out = net(x)


bo.export_finn_onnx(net, (1,1,1,2), os.path.join(model_path, "model.onnx"))
