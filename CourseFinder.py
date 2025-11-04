from math import *
from Point import *
from Course import *
from Field import *
from Vec import *
import numpy as np


class CourseFinder:

	def __init__(self, origin_points, field):
		self.origin_points = origin_points.copy()
		self.course = Course([self.origin_points[0]])
		self.last_point = self.origin_points[-1]
		self.field = field
		with open("cfg.txt") as cfg_file:
			for line in cfg_file.readlines():
				line = line.strip()
				if line.startswith("points_density_per_km"): 
					self.density = int(line.split("=")[1].strip()) 
				if line.startswith("max_line_len"):
					self.maxlen = int(line.split("=")[1].strip())	

	def points_on_circle(self, point):
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
		return res
	
	def check_circle_points(self, point, circ_lines):
		visible = []
		for line in circ_lines:
			for pt in line:
				if pt[0] < 0 or pt[1] < 0 or pt[0] > len(self.field[0]) - 1 or pt[1] > len(self.field) - 1:
					continue
				if self.field[pt[1]][pt[0]] == 1:
					continue
				vis = point.find_visibility(pt, self.field)
				if vis:
					visible.append(pt)
		return visible 

	def pick_best_on_circ(self, point):
		circ_lines = CourseFinder.points_on_circle(self, point)
		point.vis_on_target(self.last_point)
		visible_points = CourseFinder.check_circle_points(self, point, circ_lines)
		if point.is_target_vis():
			return self.last_point, True
		pick_factor = []
		checked_points = []
		for pt in visible_points:
			pt = Point(pt[0], pt[1], self.maxlen)
			circ_lines = CourseFinder.points_on_circle(self, pt)
			CourseFinder.check_circle_points(self, pt, circ_lines)
			pick_factor.append(pt.get_vis_volume()[1] / pt.find_s(self.last_point))
			checked_points.append(pt)
			del pt

		best_point = max(pick_factor)	
		best_point_coords = checked_points[pick_factor.index(best_point)]
		return best_point_coords, False	

	def create_course(self):
		find_last = False
		while(not find_last):
			current_point = self.course.get_last()
			print(current_point.get_xy())
			new_point, find_last = CourseFinder.pick_best_on_circ(self, current_point)
			self.course.add_point(new_point)
		return self.course
