import csv


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

	def get_bounds(x, y, maxlen):
		# looks like shit 
		left = x - maxlen
		if left < 0:
			left = 0
		up = y - maxlen
		if up < 0:
			up = 0 
		right = x + maxlen
		if right > len(self.field[0]):
			right = len(self.field[0])
		down = y + maxlen
		if down > len(self.field):
			down =  len(self.field)
		return left, up, right, down
