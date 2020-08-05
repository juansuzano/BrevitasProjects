import os

#import numpy as np
import pandas as pd

import torch.nn.functional as F
import torch.onnx
# from sklearn.utils import shuffle
from torch import nn
from torch import optim
from torch.autograd import Variable

import brevitas.nn as qnn
from brevitas.core.quant import QuantType

#from random import shuffle

import brevitas.onnx as bo

#from test_lenet import check_model, score_model

class QuantLeNet(nn.Module):
	def __init__(self):
		super(QuantLeNet, self).__init__()
		self.conv1 = qnn.QuantConv2d(1, 6, 5, 
									 weight_quant_type=QuantType.INT, 
									 weight_bit_width=2 , padding=2, bias=False)
		self.relu1 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=2, max_val=6)
		self.conv2 = qnn.QuantConv2d(6, 16, 5, 
									 weight_quant_type=QuantType.INT, 
									 weight_bit_width=2, bias=False)
		self.relu2 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=2, max_val=6)
		self.fc1   = qnn.QuantLinear(16*5*5, 120, bias=True, 
									 weight_quant_type=QuantType.INT, 
									 weight_bit_width=2)
		self.relu3 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=2, max_val=6)
		self.fc2   = qnn.QuantLinear(120, 84, bias=True, 
									 weight_quant_type=QuantType.INT, 
									 weight_bit_width=2)
		self.relu4 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=2, max_val=6)
		self.fc3   = qnn.QuantLinear(84, 10, bias=False, 
									 weight_quant_type=QuantType.INT, 
									 weight_bit_width=2)

	def forward(self, x):
		out = self.relu1(self.conv1(x))
		out = F.max_pool2d(out, 2)
		out = self.relu2(self.conv2(out))
		out = F.max_pool2d(out, 2)
		out = out.view(out.size(0), -1)
		out = self.relu3(self.fc1(out))
		out = self.relu4(self.fc2(out))
		out = self.fc3(out)
		return out


net = QuantLeNet()
print(net)

use_gpu = torch.cuda.is_available()
if use_gpu:
	net = net.cuda()
	print('USE GPU')
else:
	print('USE CPU')

# #check if layer is exportable
# print(net.conv1.get_exportable_quantization_type())
# print(net.conv2.get_exportable_quantization_type())
# print(net.relu1.get_exportable_quantization_type())
# print(net.relu2.get_exportable_quantization_type())
# print(net.relu3.get_exportable_quantization_type())
# print(net.fc1.get_exportable_quantization_type())
# print(net.fc2.get_exportable_quantization_type())
# print(net.fc3.get_exportable_quantization_type())


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)


print("1. Loading data")
train = pd.read_csv("datasets/train.csv").values
#train = shuffle(train)
test = pd.read_csv("datasets/test.csv").values

print("2. Converting data")
X_data = train[:, 1:].reshape(train.shape[0], 1, 28, 28)
X_data = X_data.astype(float)
X_data /= 255.0
X_data = torch.from_numpy(X_data);
X_label = train[:, 0];
X_label = X_label.astype(int);
X_label = torch.from_numpy(X_label);
X_label = X_label.view(train.shape[0], -1);
print(X_data.size(), X_label.size())

print("3. Training phase")
nb_train = train.shape[0]
nb_epoch = 2000 #750000
nb_index = 0
nb_batch = 4

for epoch in range(nb_epoch):
	if nb_index + nb_batch >= nb_train:
		nb_index = 0
	else:
		nb_index = nb_index + nb_batch

	#print("1")
	mini_data = Variable(X_data[nb_index:(nb_index + nb_batch)].clone())
	mini_label = Variable(X_label[nb_index:(nb_index + nb_batch)].clone(), requires_grad=False)
	mini_data = mini_data.type(torch.FloatTensor)
	mini_label = mini_label.type(torch.LongTensor)
	#print("2")
	if use_gpu:
		mini_data = mini_data.cuda()
		mini_label = mini_label.cuda()
	#print("3")
	optimizer.zero_grad()
	#print("4")
	#print(mini_data)
	mini_out = net(mini_data)
	#print("5")
	mini_label = mini_label.view(nb_batch)
	#print("6")
	mini_loss = criterion(mini_out, mini_label)
	mini_loss.backward()
	optimizer.step()
	
	if (epoch + 1) % 2000 == 0:
		print("Epoch = %d, Loss = %f" % (epoch + 1, mini_loss.data))



print("4. Testing phase")

Y_data = test.reshape(test.shape[0], 1, 28, 28)
Y_data = Y_data.astype(float)
Y_data /= 255.0
Y_data = torch.from_numpy(Y_data);
print(Y_data.size())
nb_test = test.shape[0]

net.eval()

final_prediction = np.ndarray(shape=(nb_test, 2), dtype=int)
#for each_sample in range(nb_test):
for each_sample in range(100):
	sample_data = Variable(Y_data[each_sample:each_sample + 1].clone())
	sample_data = sample_data.type(torch.FloatTensor)
	if use_gpu:
		sample_data = sample_data.cuda()
	sample_out = net(sample_data)
	_, pred = torch.max(sample_out, 1)
	final_prediction[each_sample][0] = 1 + each_sample
	final_prediction[each_sample][1] = pred.data[0]
	if (each_sample + 1) % 2000 == 0:
		print("Total tested = %d" % (each_sample + 1))


print('5. Generating submission file')

model_path = "models/"
if not os.path.exists(model_path):
	os.makedirs(model_path)
# submission = pd.DataFrame(final_prediction, dtype=int, columns=['ImageId', 'Label'])
# submission.to_csv(os.path.join(model_path, 'pytorch_LeNet.csv'), index=False, header=True)

print('6. Converting to ONNX')

net.eval()

# Input to the model
batch_size = 1

if use_gpu:
	x = torch.randn(batch_size, 1, 28, 28, requires_grad=True).cuda()
else:
	x = torch.randn(batch_size, 1, 28, 28, requires_grad=True)

torch_out = net(x)


bo.export_finn_onnx(net, (1, 1, 28, 28), model_path+"/LeNet_w8_a8.onnx")


#check_model()
#score_model()




