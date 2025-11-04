from CourseFinder import *
from Point import *
from Field import *
from Course import *

field = Field("mvs.csv")
ish_points = [Point(94, 2, 250), Point(900, 900, 250)]
for i in ish_points:
	field.get_point_info(i)
cf = CourseFinder(ish_points, field.get_field())
res = cf.create_course()
print("succes!\n", f"course is {res.printed()}")


