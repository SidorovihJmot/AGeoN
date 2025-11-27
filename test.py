from CourseFinder import *
from Point import *
from Field import *
from Course import *

field = Field("mvs.csv")
ish_points = [Point(4, 2, 10), Point(90, 90, 10)]
for i in ish_points:
  field.get_point_info(i)
cf = CourseFinder(ish_points, field.get_field())
res = cf.create_course()
print("succes!\n", f"course is {res.printed()}")

with open("out.csv", "w", encoding="UTF-8", newline="") as out:
  wrt = csv.writer(out)
  for coords in res.get_list():
    wrt.writerow(coords.get_xy())
