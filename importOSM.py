import math as m
import time as t
import numpy as np
import pandas as pd
import osmnx as ox
from shapely.geometry import Polygon
from rasterio.features import geometry_mask
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy import ndimage
from rasterio.transform import from_bounds
from shapely.geometry import LineString
# геодезические координаты центра объекта работ
# 60.096685, 29.962880
# 59.999649, 30.249756
# 59.937813, 30.224510
# 59.985784, 29.774339
# 60.008004, 30.258804

# -----------

def expDataFrame(a, b):
    try:
        df = pd.DataFrame(a)
        df.to_csv(b, index=False)
    except:
        print('Ошибка сохранения ', b)
        pass

def ffp_with_ch_er(a, b):
    try: 
        c = ox.features_from_polygon(a, b)
        return (c)
    except:
        print('Ошибка взятия значения ', a)
        pass  

def crtgradientmatrix(polygon, vectors_dict, resolution=1):
    """
    Parameters:
    polygon: Polygon - полигон области
    vectors_dict: dict - словарь с GeoDataFrame объектов
    resolution: int - разрешение в метрах +-
    
    Returns:
    numpy.array - матрица градиентов
    """
    bounds = polygon.bounds
    minx, miny, maxx, maxy = bounds
    lat_center = (miny + maxy) / 2
    meter_per_degree_lat = 111300  
    meter_per_degree_lon = m.cos(lat_center * m.pi / 180) * 111300 
    width_degrees = maxx - minx
    height_degrees = maxy - miny
    width_meters = width_degrees * meter_per_degree_lon
    height_meters = height_degrees * meter_per_degree_lat
    cols = int(width_meters / resolution)
    rows = int(height_meters / resolution)
    print(f"Matrix size: {rows}x{cols} pixels")
    transform = from_bounds(minx, miny, maxx, maxy, cols, rows)
    gradient_matrix = np.zeros((cols, rows))
    if 'highway' in vectors_dict and vectors_dict['highway'] is not None:
        highway_gdf = vectors_dict['highway']
        if not highway_gdf.empty:
            highway_mask = geometry_mask(
                highway_gdf.geometry,
                out_shape=(cols, rows),
                transform=transform,
                invert=True 
            )
            distance_highway = ndimage.distance_transform_edt(~highway_mask)
            highway_weight = 20 * np.exp(-distance_highway / 15)  
            intersection_areas = ndimage.convolve(highway_mask.astype(float), 
                                                np.ones((3, 3))) > 2
            highway_weight[intersection_areas] = 35 * np.exp(-distance_highway[intersection_areas] / 15)
            gradient_matrix = np.maximum(gradient_matrix, highway_weight) 
    if 'building' in vectors_dict and vectors_dict['building'] is not None:
        building_gdf = vectors_dict['building']
        if not building_gdf.empty:
            building_mask = geometry_mask(
                building_gdf.geometry,
                out_shape=(cols, rows),
                transform=transform,
                invert=True
            )
            distance_building = ndimage.distance_transform_edt(~building_mask)
            building_effect = 1 - np.exp(-distance_building / 15) 
            gradient_matrix = gradient_matrix * building_effect
    if 'water' in vectors_dict and vectors_dict['water'] is not None:
        water_gdf = vectors_dict['water']
        if not water_gdf.empty:
            water_mask = geometry_mask(
                water_gdf.geometry,
                out_shape=(cols, rows),
                transform=transform,
                invert=True
            )
            distance_water = ndimage.distance_transform_edt(~water_mask)
            water_effect = np.exp(-distance_water / 15)  
            gradient_matrix = np.maximum(gradient_matrix, water_effect)
    return gradient_matrix, transform,water_mask,building_mask

def GNSSstandable(polygon, building, cutoff):
    buildingEPSG = building.to_crs('EPSG:3857')
    GNSScutoff = buildingEPSG.buffer(cutoff)
    GNSScutoff = GNSScutoff.to_crs('EPSG:4326')
    return GNSScutoff

# -----------   
 
lat = 59.985784
long = 29.774339 
# размер объекта работ в км * 2
width = 0.5
lenght = 0.5
resolution = 1 #м^2
wdeg = width/111.13 
ldeg = lenght/(m.cos(lat*m.pi/180)*111.13)
nlatb = lat + wdeg 
slatb = lat - wdeg 
wlongb = long + ldeg 
elongb =long - ldeg 
poly = Polygon([(wlongb, slatb), (wlongb, nlatb), (elongb, nlatb), (elongb, slatb), (wlongb, slatb)])
# виды объектов
tbuilding = {'building': True}
thighway = {'highway': True}
twater = {'water':True}
print('start timik')
stime = t.time()
water_vector = ffp_with_ch_er(poly, twater)
buildings_vector = ffp_with_ch_er(poly, tbuilding)
highway_vector = ffp_with_ch_er(poly, thighway)
print(t.time()-stime)
vectors_dict = {
    'highway': highway_vector,
    'building': buildings_vector,
    'water': water_vector
}
stime = t.time()
gradient_matrix, transform, wm, bm  = crtgradientmatrix(poly, vectors_dict, resolution)
GNSScutoff = GNSSstandable(poly, buildings_vector,30)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 15))
ax1.set_xlim(elongb, wlongb)
ax1.set_ylim(slatb, nlatb)
ax3.set_xlim(elongb, wlongb)
ax3.set_ylim(slatb, nlatb)
try:
    if water_vector is not None and not water_vector.empty:
        water_vector.plot(ax=ax1, color="cyan", alpha=0.7)
except:
    pass

try:
    if buildings_vector is not None and not buildings_vector.empty:
        buildings_vector.plot(ax=ax1, color="brown", alpha=0.7)
except:
    pass

try:
    if highway_vector is not None and not highway_vector.empty:
        highway_vector.plot(ax=ax1, color="grey", alpha=0.7)
except:
    pass
m = wm + bm * 2
extent = [poly.bounds[0], poly.bounds[2], poly.bounds[1], poly.bounds[3]]
ax2.imshow(gradient_matrix, extent=extent, alpha=0.8)
ax4.imshow(m, extent=extent, alpha=0.8)
try:
    if GNSScutoff is not None and not GNSScutoff.empty:
        GNSScutoff.plot(ax=ax3, color="cyan", alpha=0.7)
except:
    pass
try:
    if buildings_vector is not None and not buildings_vector.empty:
        buildings_vector.plot(ax=ax3, color="brown", alpha=0.7)
except:
    pass
print(t.time()-stime)
expDataFrame(gradient_matrix,'exp.csv')
expDataFrame(buildings_vector,'vb.csv')
expDataFrame(highway_vector,'vh.csv')
expDataFrame(water_vector,'vw.csv')
expDataFrame(m,'mvs.csv')
plt.show()
print(f"Размер матрицы: {gradient_matrix.shape}")
print(f"Диапазон значений: {gradient_matrix.min():.2f} - {gradient_matrix.max():.2f}")
print(f"Среднее значение: {gradient_matrix.mean():.2f}")

 