class Course:

	def __init__(self, points=None):
		if points is None:
			points = []
		self.course = points.copy()
	
	def update(self, point):
		self.course.append(point)

