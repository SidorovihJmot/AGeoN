# dementia reminder: CS is y-down x-right, indexing in field[y][x]
from Vec import *


class Point:
	
	def __init__(self, x, y):
		self.x = x  # coords are indexes in field matrix
		self.y = y
		with open("cfg.txt") as cfg_file:
			for line in cfg_file.readlines():
				line = line.strip()
				if line.startswith("max_line_len"):
					self.maxlen = int(line.split("=")[1].strip())	
		self.vis_volume = set()

	def get_xy(self):
		return self.x, self.y

	def get_vis_volume(self):
		return self.vis_volume
	
	def add_connection(self, point):
		self.connections.append(point)

	#	based on raycasting, point2 as x y
	def find_visibility(self, point2, field):
		line = brezenhem_line(self.get_xy(), point2)
		indx = 0
		for x, y in line:
			try:
				if field[y][x] == 2:
					return False
				self.vis_volume.add((x, y))
				indx += 1
			except Exception:
				pass
		return True
							
