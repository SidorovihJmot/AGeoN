# dementia reminder: CS is y-down x-right, indexing in field[y][x]
from Vec import *
from math import *
import numpy as np


class Point:
	
	def __init__(self, x, y, maxlen):
		self.x = x  # coords are indexes in field matrix
		self.y = y
		size = 10 ** 6 
		self.vis_volume = np.zeros((size + 1, 2), dtype=np.int32)
		self.vis_vol_indx = 0
		self.target = None

	def get_xy(self):
		return self.x, self.y

	def get_vis_volume(self):
		return self.vis_volume, self.vis_vol_indx
	
	def find_s(self, point2):
		x0, y0 = self.x, self.y	
		x1, y1 = point2.get_xy()
		return int(sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))

	def vis_on_target(self, target_point):
		self.target = target_point
		self.target_visible = False

	def is_target_vis(self):
		return self.target_visible

	#	based on raycasting, point2 as x y
	def find_visibility(self, point2, field):
		line = brezenhem_line(self.get_xy(), point2)
		for x, y in line:
			if self.target is not None:
				xt, yt = self.target.get_xy()
				if xt == x and yt == y:
					self.target_visible = True
			if field[y][x] == 2:
				return False
			self.vis_volume[self.vis_vol_indx, 0] = x
			self.vis_volume[self.vis_vol_indx, 1] = y 
			self.vis_vol_indx += 1
		return True
							
