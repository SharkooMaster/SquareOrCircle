import json
import numpy as np

from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image

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

mainPath = "./UnformatedData/"
onlyfiles = [f for f in listdir(mainPath) if isfile(join(mainPath, f))]
UnformatedDataPaths = []
for i in range(len(onlyfiles)):
	UnformatedDataPaths.append(mainPath + onlyfiles[i])
print(UnformatedDataPaths)

toStore_data = []
toStore_label = []
for i in range(len(onlyfiles)):
	temp = _img(UnformatedDataPaths[i])
	toStore_data.append(temp.flatArr.tolist())
	if(onlyfiles[i][0] == "c"):
		toStore_label.append([0, 1])
	else:
		toStore_label.append([1, 0])

structure = {"training_data": toStore_data, "data_labels": toStore_label}

with open("./Data.json", "w") as f:
	f.write(json.dumps(structure))
