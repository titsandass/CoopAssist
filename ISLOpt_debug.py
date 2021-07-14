def read_czml_file(czml_filepath):
    import json
    with open(czml_filepath, 'r') as czml_json:
        czml_list = json.load(czml_json)

    return czml_list

def change_color_of_satellites(point_color, point_outcolor, point_size, czml_list):
    for element in czml_list:
        if 'point' in element:
            if 'color' in element['point']:
                element['point']['color']['rgba'] = point_color
            if 'outlineColor' in element['point']:
                element['point']['outlineColor']['rgba'] = point_outcolor
            if 'pixelSize' in element['point']:
                element['point']['pixelSize'] = point_size

def dijkstra_with_graph(graph, start_node, end_node):
    from dijkstra import DijkstraSPF
    dijkstra_graph = DijkstraSPF(graph, start_node)
    optimal_path = dijkstra_graph.get_path(end_node)
    distance = dijkstra_graph.get_distance(end_node)

    return dijkstra_graph, optimal_path, distance

def add_cities_to_czml(cities, pivot_time, czml_list, city_linecolor, city_outlinecolor, city_labelcolor, city_labelbackgroundcolor):
    import datetime as dt

    availability = pivot_time.strftime('%Y-%m-%dT%H:%M:%S.%f/') + (pivot_time + dt.timedelta(days=2)).strftime('%Y-%m-%dT%H:%M:%S.%f')
    
    for city_name in cities.keys():
        c = dict()
        c['id'] = city_name
        c['description'] = city_name
        c['availability'] = availability
        c['position'] = dict()
        c['position']['epoch'] = pivot_time.strftime('%Y-%m-%dT%H:%M:%S.%f')
        c['position']['cartesian'] = [
            float(cities[city_name][0]),
            float(cities[city_name][1]),
            float(cities[city_name][2]),
        ]
        # c['position']['interpolationAlgorithm'] = 'LAGRANGE'
        # c['position']['interpolationDegree'] = 5
        c['position']['referenceFrame'] = 'FIXED'
        c['point'] = dict()
        c['point']['color'] = dict()
        c['point']['color']['rgba'] = city_linecolor
        c['point']['outlineColor'] = dict()
        c['point']['outlineColor']['rgba'] = city_outlinecolor
        c['point']['outlineWidth'] = 2
        c['point']['pixelSize'] = 7
        c['label'] = dict()
        c['label']['fillColor'] = dict()
        c['label']['fillColor']['rgba'] = city_labelcolor
        c['label']['font'] = "12pt Lucida Console"
        c['label']['horizontalOrigin'] = "LEFT"
        c['label']['pixelOffset'] = dict()
        c['label']['pixelOffset']['cartesian2'] = [8, 0]
        c['label']['style'] = 'FILL'
        c['label']['text'] = city_name
        c['label']['showBackground'] = 'true'
        c['label']['backgroundColor'] = dict()
        c['label']['backgroundColor']['rgba'] = city_labelbackgroundcolor

        czml_list.append(c)

def add_path_to_czml(path_name, optimal_path, path_start_time, timestep, czml_list, path_linecolor, path_outlinecolor):
    import datetime as dt

    path = dict()
    path['id'] = path_name
    path['name'] = path_name
    availability = path_start_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ/") + (path_start_time + dt.timedelta(seconds=timestep)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    path['availability'] = availability,
    path['polyline'] = dict()
    path['polyline']['show'] = 'true'
    path['polyline']['width'] = 6
    path['polyline']['material'] = dict()
    path['polyline']['material']['polylineOutline'] = dict()
    path['polyline']['material']['polylineOutline']['color'] = dict()
    path['polyline']['material']['polylineOutline']['color']['rgba'] = path_linecolor
    path['polyline']['material']['polylineOutline']['outlineColor'] = dict()
    path['polyline']['material']['polylineOutline']['outlineColor']['rgba'] = path_outlinecolor
    path['polyline']['material']['polylineOutline']['outlineWidth'] = 3
    path['polyline']['arcType'] = 'none'
    path['polyline']['positions'] = dict()
    path['polyline']['positions']['references'] = [str(satID)+'#position' for satID in optimal_path]

    czml_list.append(path)

def save_czml(czml_result_filepath, czml_data):
    import json
    with open(czml_result_filepath, 'w') as json_czml:
        json.dump(czml_data, json_czml, indent=4)

def latlong2eci_cities(latlon_cities, specific_time, source_city, destination_city):
    """
    convert geodetic coordinates to Earth Centered Internal ECI
    J2000 frame
    Parameters
    ----------
    lat : float
        geodetic latitude
    lon : float
        geodetic longitude
    alt : float
        altitude above ellipsoid  (meters)
    t : datetime.datetime, float
        UTC time
    ell : Ellipsoid, optional
        planet ellipsoid model
    deg : bool, optional
        if True, degrees. if False, radians
    use_astropy: bool, optional
        use AstroPy (recommended)

    Results
    -------
    x : float
        ECI x-location [meters]
    y : float
        ECI y-location [meters]
    z : float
        ECI z-location [meters]
    """

    from pymap3d import geodetic2eci, ellipsoid
    import datetime as dt

    wgs72 = ellipsoid.Ellipsoid(model='wgs72')
    
    eci_cities = dict()
    for city_name in latlon_cities.keys():
        if(city_name == source_city or city_name == destination_city):
            lat, lon = latlon_cities[city_name]
            eci_coord = geodetic2eci(lat=lat, lon=lon, alt=0, t=specific_time, ell=wgs72)

        eci_cities[city_name] = eci_coord

    return eci_cities

def latlong2ecef_cities(latlon_cities):
    from pymap3d import geodetic2ecef, ellipsoid

    wgs72 = ellipsoid.Ellipsoid(model='wgs72')
    ecef_cities = dict()
    for city_name in latlon_cities.keys():
        lat, lon = latlon_cities[city_name]
        ecef_coord = geodetic2ecef(lat=lat, lon=lon, alt=0, ell=wgs72)

        ecef_cities[city_name] = ecef_coord

    return ecef_cities

def parse_TLE(TLE_filepath):
    import os
    if not os.path.exists(TLE_filepath):
        raise FileExistsError(TLE_filepath+" Not Exists")

    import pickle

    if os.path.exists(TLE_filepath[:-3]+'pickle'):
        print('Found : '+TLE_filepath[:-3]+'pickle')
        with open(TLE_filepath[:-3]+'pickle', 'rb') as f:
            SATs = pickle.load(f)
            print(TLE_filepath[:-3]+'pickle Loaded')

    else:
        from sgp4.earth_gravity import wgs72
        from sgp4.io import twoline2rv

        SATs = dict()
        with open(TLE_filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip('\n')
                if line.startswith('0'):
                    SAT_Name = line.replace('\t',' ')[2:]
                elif line.startswith('1'):
                    l1 = line
                elif line.startswith('2'):
                    l2 = line
                    SAT_ID = line.split(' ')[1].zfill(5)
                    SATs[SAT_ID] = [SAT_Name, twoline2rv(l1, l2, wgs72)]

        with open(TLE_filepath[:-3]+'pickle', 'wb') as f:
            pickle.dump(SATs, f)
            print(TLE_filepath[:-3]+'pickle Saved')
    return SATs

def read_KNN_file(KNN_filepath):
    import pickle
    import os

    if os.path.exists(KNN_filepath[:-3]+'pickle'):
        print('Found : '+KNN_filepath[:-3]+'pickle')
        with open(KNN_filepath[:-3]+'pickle', 'rb') as f:
            pivot_date, KNN_Results = pickle.load(f)
            print(KNN_filepath[:-3]+'pickle Loaded')
            return pivot_date, KNN_Results

    with open(KNN_filepath, 'r') as f:
        print('generating KNN pickle')
        lines = f.readlines()
        KNN_Results = list()

        pivot_date = None
        for line in lines:
            if line.startswith('%'):
                continue
            leg_info = line.strip('\n').split('\t')
            
            if pivot_date is None:
                pivot_date = leg_info[6]+'-'+leg_info[7].zfill(2)+'-'+leg_info[8].zfill(2)+'T'+leg_info[9].zfill(2)+':'+leg_info[10].zfill(2)+':'+leg_info[11]

            knn = dict()
            knn['Pair']     = leg_info[0], leg_info[1]
            knn['Interval'] = float(leg_info[4]), float(leg_info[5])

            KNN_Results.append(knn)

    with open(KNN_filepath[:-3]+'pickle', 'wb') as f:
        pickle.dump((pivot_date, KNN_Results), f)

    return pivot_date, KNN_Results

def generate_timestep_map(KNN_Results, start_time, end_time, timestep):
    import os
    import pickle

    # timestep_map_name = '/home/shchoi/new_coop/DB/isl_opt_timestamp_map.pickle'
    # if os.path.exists(timestep_map_name):
    #     print('Found : '+timestep_map_name)
    #     with open(timestep_map_name, 'rb') as f:
    #         timestep_map = pickle.load(f)
    #         return timestep_map
    
    print('Generate_timestep_map')
    timestep_map = dict()
    for knn in KNN_Results:
        curr_time = start_time
        while curr_time <= end_time and curr_time >= knn['Interval'][0] and curr_time <= knn['Interval'][1]:
            if curr_time not in timestep_map:
                timestep_map[curr_time] = list()
            timestep_map[curr_time].append(knn['Pair'])
            curr_time += timestep

    # with open(timestep_map_name, 'wb') as f:
    #     pickle.dump(timestep_map, f)
    
    print('Done')
    return timestep_map

def propagate_SATs(SATs, time):
    for SAT_Name in SATs.keys():
        pos, vel = SATs[SAT_Name][1].propagate(time.year, time.month, time.day, time.hour, time.minute, time.second + time.microsecond/1000000)
        if len(SATs[SAT_Name]) == 2:
            SATs[SAT_Name].append(pos)
        else:
            SATs[SAT_Name][2] = pos
    return SATs

def generate_graph_with_pairs(SAT_Pairs):
    from dijkstra import Graph

    graph = Graph()
    for pair in SAT_Pairs:
        graph.add_edge(pair[0], pair[1], 1)
        graph.add_edge(pair[1], pair[0], 1)
    return graph

def find_closest_SAT_with_city(SATs, eci_cities, source_city, destination_city):
    import numpy as np

    pairs = list()
    for city in eci_cities.keys():
        if(city == source_city or city == destination_city):
            dist = None
            closest = None
            for SAT_Name in SATs.keys():
                a = np.array(eci_cities[city]).flatten()/1000
                b = np.array(SATs[SAT_Name][2])
                new_dist = np.linalg.norm(a-b)
                if dist == None:
                    dist = new_dist
                    closest = SAT_Name
                elif new_dist < dist:
                    dist = new_dist
                    closest = SAT_Name
            pairs.append((city, closest))
    return pairs

# def connect_cities_with_satellites(start_city, start_SATID, end_city, end_SATID, pivot_time, czml_list, city_linecolor, city_outlinecolor):
#     import datetime as dt
#     availability = pivot_time.strftime('%Y-%m-%dT%H:%M:%S.%f/') + (pivot_time + dt.timedelta(days=2)).strftime('%Y-%m-%dT%H:%M:%S.%f')
    
#     start_path = dict()
#     start_path['id'] = start_city + ' to ' + str(start_SATID)
#     start_path['name'] = start_city + ' to ' + str(start_SATID)
#     start_path['availability'] = availability,
#     start_path['polyline'] = dict()
#     start_path['polyline']['show'] = 'true'
#     start_path['polyline']['width'] = 5
#     start_path['polyline']['material'] = dict()
#     start_path['polyline']['material']['polylineOutline'] = dict()
#     start_path['polyline']['material']['polylineOutline']['color'] = dict()
#     start_path['polyline']['material']['polylineOutline']['color']['rgba'] = city_linecolor
#     start_path['polyline']['material']['polylineOutline']['outlineColor'] = dict()
#     start_path['polyline']['material']['polylineOutline']['outlineColor']['rgba'] = city_outlinecolor
#     start_path['polyline']['material']['polylineOutline']['outlineWidth'] = 3
#     start_path['polyline']['arcType'] = 'none'
#     start_path['polyline']['positions'] = dict()
#     start_path['polyline']['positions']['references'] = [start_city+'#position', str(start_SATID)+'#position']

#     end_path = dict()
#     end_path['id'] = str(end_SATID) + ' to ' + end_city
#     end_path['name'] = str(end_SATID) + ' to ' + end_city
#     end_path['availability'] = "2021-06-10T00:00:00.000Z/2021-06-10T01:00:00.000Z",
#     end_path['polyline'] = dict()
#     end_path['polyline']['show'] = 'true'
#     end_path['polyline']['width'] = 5
#     end_path['polyline']['material'] = dict()
#     end_path['polyline']['material']['polylineOutline'] = dict()
#     end_path['polyline']['material']['polylineOutline']['color'] = dict()
#     end_path['polyline']['material']['polylineOutline']['color']['rgba'] = city_linecolor
#     end_path['polyline']['material']['polylineOutline']['outlineColor'] = dict()
#     end_path['polyline']['material']['polylineOutline']['outlineColor']['rgba'] = city_outlinecolor
#     end_path['polyline']['material']['polylineOutline']['outlineWidth'] = 2
#     end_path['polyline']['arcType'] = 'none'
#     end_path['polyline']['positions'] = dict()
#     end_path['polyline']['positions']['references'] = [end_city+'#position', str(end_SATID)+'#position']

#     czml_list.append(start_path)
#     czml_list.append(end_path)
if __name__ == "__main__":
    import sys
    import datetime as dt
    from color_constants import *
    from LatLong_cities import latlong_cities

    # _, KNN_filepath, TLE_filepath, czml_filepath, czml_result_filepath, start_time, end_time, timestep, source_city, destination_city = sys.argv
    KNN_filepath      = 'all_starlink_thetaNN.tnn'
    TLE_filepath        = 'latest_all_starlink.tle'

    czml_filepath       = 'pretty_orbit.czml'
    czml_result_filepath= '123.czml'

    start_time  = 0
    end_time    = 6000
    timestep    = 10

    source_city  = 'Seoul'
    destination_city    = 'NewYork'
    
    start_time = int(start_time)
    end_time = int(end_time)
    timestep = int(timestep)

    pivot_date, KNN_Results = read_KNN_file(KNN_filepath)
    pivot_time = dt.datetime.strptime(pivot_date, '%Y-%m-%dT%H:%M:%S.%f')
    
    
    import time
    start = time.time()
    timestep_map = generate_timestep_map(KNN_Results, start_time, end_time, timestep)
    print("generate_timestep_map time :", time.time() - start)
    


    SATs = parse_TLE(TLE_filepath)

    dijkstra_paths = dict()
    curr_time = start_time
    

    loop_start = time.time()
    while curr_time <= end_time:    
        currtime_pairs = timestep_map[curr_time]
        start = time.time()
        graph = generate_graph_with_pairs(currtime_pairs)
        
        print("generate_graph_with_pairs time :", time.time() - start)
        start = time.time()

        utc_time = pivot_time + dt.timedelta(seconds=curr_time)
        
        print("timedelta time :", time.time() - start)
        start = time.time()

        SATs = propagate_SATs(SATs, utc_time)

        print("propagate_SATs time :", time.time() - start)
        start = time.time()

        eci_cities  = latlong2eci_cities(latlong_cities, utc_time, source_city, destination_city)
        
        print("latlong2eci_cities time :", time.time() - start)
        start = time.time()

        city_SAT = find_closest_SAT_with_city(SATs, eci_cities, source_city, destination_city)
        
        print("find_closest_SAT_with_city time :", time.time() - start)
        start = time.time()

        for (city, SAT) in city_SAT:
            graph.add_edge(city, SAT, 1)
            graph.add_edge(SAT, city, 1)
        # source_city = '46338'
        # destination_city = '46547'
        dijkstra_graph, optimal_path, distance = dijkstra_with_graph(graph, source_city, destination_city)
        dijkstra_paths[curr_time] = optimal_path
        
        print("time :", time.time() - start)
        start = time.time()

        curr_time += timestep
    
    print("total loop time :", time.time() - loop_start)

    czml_list = read_czml_file(czml_filepath)
    change_color_of_satellites(satellite_color, satellite_outcolor, satellite_size, czml_list)
    ecef_cities  = latlong2ecef_cities(latlong_cities)
    add_cities_to_czml(ecef_cities, pivot_time, czml_list, city_linecolor, city_outlinecolor, city_labelcolor, city_labelbackgroundcolor)

    path_start_time = pivot_time + dt.timedelta(seconds=start_time)
    for i, path in enumerate(dijkstra_paths):
        add_path_to_czml(source_city+' to '+destination_city+' '+str(i), dijkstra_paths[i*timestep+start_time], path_start_time, timestep, czml_list, path_linecolor, path_outlinecolor)
        path_start_time += dt.timedelta(seconds=timestep)
    save_czml(czml_result_filepath, czml_list)

    pass
