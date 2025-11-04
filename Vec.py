from math import *
import numpy as np


def brezenhem_line(start, end):
	x0, y0 = start
	x1, y1 = end

	dx = abs(x1 - x0)
	dy = abs(y1 - y0)
	size = int(sqrt(dx ** 2 + dy ** 2))

	sign_x = -1 if x0 < x1 else 1
	sign_y = -1 if y0 < y1 else 1
	error = dx - dy
	res = np.zeros((size + 1, 2), dtype = np.int32)
	indx = 0
	while (x0 != x1 or y0 != y1):
		res[indx, 0] = x0
		res[indx, 1] = y0
		double_error = 2 * error
		if double_error > -1 * dy:
			error -= dy
			x0 -= sign_x
		if double_error < dx:
			error += dx
			y0 -= sign_y
		indx += 1
	res[indx, 0] = x1
	res[indx, 1] = y1
	return res

