# dementia reminder: CS is y-down x-right, indexing in field[y][x]
import csv


# input vectors are 4 floats output 2 floats)))
def normilezed_vec(vec):
	ln = ((vec[2] - vec[0]) ** 2 + (vec[3] - vec[1]) ** 2) ** 0.5
	return (vec[2] - vec[0]) / ln, (vec[3] - vec[1]) / ln

# very slow shit (python)
def cut_square_field(field, x, y, maxlen):
	size = maxlen 
	res = []
	# looks like shit	
	left = x - size
	if left < 0:
		left = 0
	up = y - size
	if up < 0:
		up = 0 
	right = x + size
	if right > len(field[0]):
		right = len(field[0])
	down = y + size
	if down > len(field):
		down =  len(field)

	for i in range(up, down):
		res.append(field[i][left:right])
	return res


class Point:
	
	def __init__(self, x, y, field):
		self.x = x  # coords are indexes in field matrix
		self.y = y
		with open("cfg.txt") as cfg_file:
			for line in cfg_file.readlines():
				line = line.strip()
				if line.startswith("max_line_len"):
					self.maxlen = int(line.split("=")[1].strip())	
		self.field = cut_square_field(field, x, y, self.maxlen) 

	def get_xy(self):
		return self.x, self.y

	def find_vision_volume(self):
		pass

	# based on raycasting
	def find_visibility(self, point2):
		ray = (self.x, self.y, *point2.get_xy())
		field_check_chunk = [self.x, self.y]
		norm_ray = normilezed_vec(ray)
		step_size_vec = (abs(1 / norm_ray[0]), abs(1 / norm_ray[1]))
		approx_ray = [0, 0] # there will be stored len to obstacle from point
		
		# len = 1 * stepsize because in original idk why x - x (= 0) * stepsize
		if norm_ray[0] < 0:
			step_x = -1
		else:
			step_x = 1
			approx_ray[0] = step_size_vec[0]
		if norm_ray[1] < 0:
			step_y = -1
		else:
			step_y = 1
			approx_ray[1] = step_size_vec[1]
		
		print(ray, field_check_chunk, norm_ray, step_size_vec, approx_ray)
		find_obstacle = False
		dist = 0
		while (not find_obstacle and dist <= self.maxlen):
			if approx_ray[0] < approx_ray[1]:
				field_check_chunk[0] += step_x
				dist = approx_ray[0]
				approx_ray[0] += step_size_vec[0]
			else:
				field_check_chunk[1] += step_y
				dist = approx_ray[1]
				approx_ray[1] += step_size_vec[1]

			if self.field[field_check_chunk[1]][field_check_chunk[0]] == 2:
				find_obstacle = True 
				print(f"bound at coords {field_check_chunk[0]} {field_check_chunk[1]}")
				print(f"len is {dist}")



with open("mvs.csv", encoding="UTF-8", newline="") as file:
	freader = csv.reader(file, delimiter=",")
	field = [list(map(int, row)) for row in freader]

test = Point(100, 100, field) 
test2 = Point(3, 15, field)
test.find_visibility(test2)
