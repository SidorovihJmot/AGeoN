class Course:

	def __init__(self, points=None):
		if points is None:
			points = []
		self.course = points.copy()
	
	def add_point(self, point):
		self.course.append(point)

	def get_last(self):
		return self.course[-1]

	def printed(self):
		st = ""
		for i in self.course:
			st += f"{str(i.get_xy())} "
		return st 

	def get_list(self):
		print(type(self.course))
		return self.course
