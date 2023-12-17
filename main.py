
# 24*24 input neurons.		[x]
# Hidden layers:			[x]
#	- Dotprod, RecLu
# Output layer:
#	- Dotprod, BinaryCrossEntropy
# Backprop/GradientDecent

import json

import numpy as np
np.random.seed(0)

# ----- Import DATA
_data_json = ""
with open("./Data.json", "r") as f:
	_data_json = json.loads(f.read())

trainingData = np.array(_data_json["training_data"])
dataLabels = np.array(_data_json["data_labels"])
# ----- ||~Import DATA~|| ----- #

imgPixelSize = (24*24)

class Layer:
	w = []
	b = []

	def __init__(self, inputs, neurons):
		self.w = 0.10 * np.random.randn(inputs, neurons)
		self.b = np.zeros((1, neurons))
	
	def forward(self, _input):
		self.output = np.dot(_input, self.w) + self.b

# ---- Activation ---- #
def ReLU(a):
	return np.maximum(0, a)

def Sigmoid(a):
	return 1 / (1 + np.exp(-a))
# ---- Activation ---- #

# ---- Cost ---- #
def BinaryCrossEntropy(_pred, _label):
	sampleClipped = np.clip(_pred, 1e-7, 1 - 1e-7)
	loss = -(_label * np.log(sampleClipped) + (1 - _label) * np.log(1 - sampleClipped))
	return np.mean(loss)

def calculate_accuracy(predictions, true_labels):
    binary_predictions = (predictions > 0.5).astype(int)
    correct_predictions = (binary_predictions == true_labels).astype(int)
    accuracy = np.mean(correct_predictions)
    return accuracy
# ---- Cost ---- #

Layer_1 = Layer(imgPixelSize, imgPixelSize * 2)
Layer_2 = Layer(imgPixelSize * 2, 2)

def Train(epochs):
	tw1 = Layer_1.w.copy()
	tb1 = Layer_1.b.copy()
	tw2 = Layer_2.w.copy()
	tb2 = Layer_2.b.copy()

	learning_rate = 0.00001
	loss_min = float('inf')

	for i in range(epochs):
		Layer_1.forward(trainingData)
		activ1 = ReLU(Layer_1.output)

		Layer_2.forward(activ1)
		activ2 = Sigmoid(Layer_2.output)

		loss = BinaryCrossEntropy(activ2, dataLabels)
		acc = calculate_accuracy(activ2, dataLabels)
		if(loss < loss_min):
			loss_min = loss
			print(f"New weights found at epoch: {i}, loss: {loss}, acc: {acc}")
			tw1 = Layer_1.w.copy()
			tb1 = Layer_1.b.copy()
			tw2 = Layer_2.w.copy()
			tb2 = Layer_2.b.copy()
		
		dL2 = activ2 - dataLabels
		dL1 = np.dot(dL2, tw2.T) * (activ1 > 0)

		Layer_2.w -= learning_rate * np.dot(activ1.T, dL2)
		Layer_2.b -= learning_rate * np.sum(dL2, axis=0, keepdims=True)

		Layer_1.w -= learning_rate * np.dot(trainingData.T, dL1)
		Layer_1.b -= learning_rate * np.sum(dL1, axis=0, keepdims=True)

Train(10000)

Results = {"L1_weights": Layer_1.w.tolist(), "L2_weights": Layer_2.w.tolist(), "L1_biases": Layer_1.b.tolist(), "L2_biases": Layer_2.b.tolist()}
with open("./Trained.json", "w") as f:
	f.write(json.dumps(Results))
