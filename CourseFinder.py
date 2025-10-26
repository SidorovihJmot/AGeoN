import random
from math import *
from Point import *
from Course import *
from Field import *
from Vec import *


class CourseFinder:

	def __init__(self, origin_points, field):
		self.origin_points = origin_points.copy()
		self.course = Course(self.origin_points[0])
		self.field = field
		with open("cfg.txt") as cfg_file:
			for line in cfg_file.readlines():
				line = line.strip()
				if line.startswith("points_density_per_km"): 
					self.density = int(line.split("=")[1].strip()) 
				if line.startswith("max_line_len"):
					self.maxlen = int(line.split("=")[1].strip())	

	def init_dir_ang(point1, point2):
		pass

	def points_on_circle(self, point):
		field = self.field 
		res = []
		x0, y0 = point.get_xy()
		circle_res = 20 #change if you want
		points = []	
		rad = self.maxlen
		for i in range(circle_res + 1):
			ang = i * 2 * pi / circle_res	
			x = rad * cos(ang) + x0
			y = y0 + rad * sin(ang) 
			points.append((int(x), int(y)))

		for i in range(circle_res):
			pair = points[i:i+2]
			line = brezenhem_line(pair[0], pair[1])
			res.append(line.copy())
		return res.copy()
	
	def check_circle_points(self, point, circ_lines):
		visible = []
		for line in circ_lines:
			for pt in line:
				if pt[0] < 0 or pt[1] < 0 or pt[0] > len(self.field[0]) or pt[1] > len(self.field):
					continue
				vis = point.find_visibility(pt, self.field)
				if vis:
					visible.append(pt)
		return visible

	def pick_best_on_circ(self, point):
		circ_lines = CourseFinder.points_on_circle(self, point)
		visible_points = CourseFinder.check_circle_points(self, point, circ_lines)
		vis_vols = []
		for pt in visible_points:
			pt = Point(pt[0], pt[1])
			vol = len(CourseFinder.check_circle_points(self, pt, CourseFinder.points_on_circle(self, pt)))
			vis_vols.append((vol, pt.get_xy()))
		return max(vis_vols, key=lambda x: x[0])	
