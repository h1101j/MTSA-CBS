

import torch
import torchvision
import torchvision.transforms.functional as F
import folders


class DataLoader_n(object):
	"""
	Dataset class for IQA databases
	"""

	def __init__(self, dataset, path, img_indx, patch_size, patch_num, batch_size, istrain=True):

		self.batch_size = batch_size
		self.istrain = istrain

		if dataset == 'cbs':
			if istrain:
				transforms = torchvision.transforms.Compose([
					# torchvision.transforms.RandomHorizontalFlip(),
					# torchvision.transforms.RandomVerticalFlip(),
					torchvision.transforms.RandomCrop(size=patch_size),
					# torchvision.transforms.Resize([224, 224]),
					torchvision.transforms.ToTensor(),
					# torchvision.transforms.Normalize(mean=(0.0, 0.0, 0.0),
					# 								 std=(1.0, 1.0, 1.0))
				])
			else:
				transforms = torchvision.transforms.Compose([
					torchvision.transforms.RandomCrop(size=patch_size),
					torchvision.transforms.ToTensor(),
					# torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
					# 								 std=(0.229, 0.224, 0.225))
				])

		if dataset == 'cbs':
			self.data = folders.cbs(
				root=path, index=img_indx, transform=transforms, patch_num=patch_num)

	def get_data(self):
		if self.istrain:
			dataloader = torch.utils.data.DataLoader(
				self.data, batch_size=self.batch_size, shuffle=False)
		else:
			dataloader = torch.utils.data.DataLoader(
				self.data, batch_size=1, shuffle=False)
		return dataloader