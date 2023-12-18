
import json

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image

import numpy as np
np.random.seed(0)

class _img:
	def __init__(self, _path) -> None:
		self.img = Image.open(_path).convert('L')

		wPercent = (24 / float(self.img.size[0]))
		hSize = int((float(self.img.size[1]) * float(wPercent)))
		self.img = self.img.resize((24, hSize), Image.Resampling.LANCZOS)

		self.pixelArr = np.array(self.img)
		self.flatArr = self.pixelArr.flatten()
	
	def __str__(self) -> str:
		return str(self.flatArr).replace(" ", ",").replace("\n","")

# ----- Import DATA
_data_json = ""
with open("./Trained.json", "r") as f:
	_data_json = json.loads(f.read())

Lw1 = np.array(_data_json["L1_weights"])
Lw2 = np.array(_data_json["L2_weights"])
Lb1 = np.array(_data_json["L1_biases"])
Lb2 = np.array(_data_json["L2_biases"])
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

Layer_1 = Layer(imgPixelSize, imgPixelSize * 2)
Layer_2 = Layer(imgPixelSize * 2, 2)

Layer_1.w = np.array(Lw1)
Layer_1.b = np.array(Lb1)
Layer_2.w = np.array(Lw2)
Layer_2.b = np.array(Lb2)

_IMG = _img("./4.png")

def SquareOrCircle(X):
	Layer_1.forward(X)
	a1 = ReLU(Layer_1.output)

	Layer_2.forward(a1)
	a2 = Sigmoid(Layer_2.output)

	print(a2)
	if(a2[0][1] == 1): print("CIRCLE")
	else: print("SQUARE")

SquareOrCircle(_IMG.flatArr)
