import csv
from Point import *


class Field:

	def __init__(self, file_name):
		self.field = Field.read_field(file_name)

	@staticmethod
	def read_field(file_name):
		with open(file_name, encoding="UTF-8", newline="") as file:
			freader = csv.reader(file, delimiter=",")
			field = [list(map(int, row)) for row in freader]
		return field
	
	def get_field(self):
		return self.field

	def get_point_info(self, point):
		x, y = point.get_xy()
		info = self.field[y][x]

		if info == 0:
			print("can be placed")
		elif info == 1:
			print("only visible")
		else:
			print("its bound")
