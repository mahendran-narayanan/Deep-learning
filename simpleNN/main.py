import torch
import argparse
import os
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchsummary as summary
from torch.autograd import Variable

def dataset():
	mnist_train = torchvision.datasets.MNIST(root='../data/',download=True,train=True,transform=transforms.ToTensor())
	mnist_test = torchvision.datasets.MNIST(root='../data/',download=True,train=False,transform = transforms.ToTensor())
	return mnist_train,mnist_test

def NNmodel():
	linear1 = torch.nn.Linear(784,512,bias=True)
	linear2 = torch.nn.Linear(512,10,bias=True)
	relu = torch.nn.ReLU()
	return torch.nn.Sequential(linear1,relu,linear2)

def performance(mnist_test,model):
	right = 0
	total = 0
	for images,labels in mnist_test:
		images = Variable(images.view(-1,28*28))
		outputs = model(images)
		_,predicted = torch.max(outputs.data,1)
		total+=1
		right+=(predicted == labels).sum()
	print("Accuracy: {:>.4}".format(100*right/total))

def main():
	mnist_train,mnist_test = dataset()
	#hyperparameters
	epochs = 5
	learning_rate = 0.004
	batch_size=25
	data_loader = torch.utils.data.DataLoader(dataset = mnist_train,batch_size = batch_size,shuffle=True)
	#model
	model = NNmodel()
	loss = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
	# training phase
	for epoch in range(epochs):
		costavg = 0
		total_batch = len(mnist_train)//batch_size
		for i,(batch_xs,batch_ys) in enumerate(data_loader):
			X = Variable(batch_xs.view(-1,28*28))
			Y = Variable(batch_ys)
			optimizer.zero_grad()
			hypothesis = model(X)
			cost = loss(hypothesis,Y)
			cost.backward()
			optimizer.step()

			costavg += cost.data/total_batch
		print("Epoch {:>2} loss: {:>.5}".format(epoch+1,costavg))
	performance(mnist_test,model)


if __name__ == '__main__':
	main()