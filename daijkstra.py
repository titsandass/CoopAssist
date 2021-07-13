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

def generate_topological_graph_with_legs(legs):
    from dijkstra import Graph

    graph = Graph()
    for leg in legs:
        graph.add_edge(leg['PrimaryID'], leg['SecondaryID'], 1)
    return graph

def dijkstra_with_graph(graph, start_node, end_node):
    from dijkstra import DijkstraSPF

    dijkstra_graph = DijkstraSPF(graph, start_node)
    optimal_path = dijkstra_graph.get_path(end_node)
    distance = dijkstra_graph.get_distance(end_node)

    return dijkstra_graph, optimal_path, distance

def add_cities_to_czml(cities, czml_list, city_linecolor, city_outlinecolor, city_labelcolor, city_labelbackgroundcolor):
    for city_name in cities.keys():
        c = dict()
        c['id'] = city_name
        c['description'] = city_name
        c['availability'] = "2021-06-10T00:00:00+00:00/2021-06-10T01:00:00+00:00"
        c['position'] = dict()
        c['position']['epoch'] = "2021-06-10T00:00:00+00:00"
        c['position']['cartesian'] = [
            0.0,
            cities[city_name][0],
            cities[city_name][1],
            cities[city_name][2],
            300.0,
            cities[city_name][0]+1,
            cities[city_name][1]+1,
            cities[city_name][2]+1
        ]
        c['position']['interpolationAlgorithm'] = 'LAGRANGE'
        c['position']['interpolationDegree'] = 5
        c['position']['referenceFrame'] = 'INERTIAL'
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

def add_path_to_czml(path_name, optimal_path, czml_list, path_linecolor, path_outlinecolor):
    path = dict()
    path['id'] = path_name
    path['name'] = path_name
    path['availability'] = "2021-06-10T00:00:00.000Z/2021-06-10T01:00:00.000Z",
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

def connect_cities_with_satellites(start_city, start_SATID, end_city, end_SATID, czml_list, city_linecolor, city_outlinecolor):
    start_path = dict()
    start_path['id'] = start_city + ' to ' + str(start_SATID)
    start_path['name'] = start_city + ' to ' + str(start_SATID)
    start_path['availability'] = "2021-06-10T00:00:00.000Z/2021-06-10T01:00:00.000Z",
    start_path['polyline'] = dict()
    start_path['polyline']['show'] = 'true'
    start_path['polyline']['width'] = 5
    start_path['polyline']['material'] = dict()
    start_path['polyline']['material']['polylineOutline'] = dict()
    start_path['polyline']['material']['polylineOutline']['color'] = dict()
    start_path['polyline']['material']['polylineOutline']['color']['rgba'] = city_linecolor
    start_path['polyline']['material']['polylineOutline']['outlineColor'] = dict()
    start_path['polyline']['material']['polylineOutline']['outlineColor']['rgba'] = city_outlinecolor
    start_path['polyline']['material']['polylineOutline']['outlineWidth'] = 3
    start_path['polyline']['arcType'] = 'none'
    start_path['polyline']['positions'] = dict()
    start_path['polyline']['positions']['references'] = [start_city+'#position', str(start_SATID)+'#position']

    end_path = dict()
    end_path['id'] = str(end_SATID) + ' to ' + end_city
    end_path['name'] = str(end_SATID) + ' to ' + end_city
    end_path['availability'] = "2021-06-10T00:00:00.000Z/2021-06-10T01:00:00.000Z",
    end_path['polyline'] = dict()
    end_path['polyline']['show'] = 'true'
    end_path['polyline']['width'] = 5
    end_path['polyline']['material'] = dict()
    end_path['polyline']['material']['polylineOutline'] = dict()
    end_path['polyline']['material']['polylineOutline']['color'] = dict()
    end_path['polyline']['material']['polylineOutline']['color']['rgba'] = city_linecolor
    end_path['polyline']['material']['polylineOutline']['outlineColor'] = dict()
    end_path['polyline']['material']['polylineOutline']['outlineColor']['rgba'] = city_outlinecolor
    end_path['polyline']['material']['polylineOutline']['outlineWidth'] = 2
    end_path['polyline']['arcType'] = 'none'
    end_path['polyline']['positions'] = dict()
    end_path['polyline']['positions']['references'] = [end_city+'#position', str(end_SATID)+'#position']

    czml_list.append(start_path)
    czml_list.append(end_path)

def save_czml(czml_result_filepath, czml_data):
    import json
    with open(czml_result_filepath, 'w') as json_czml:
        json.dump(czml_data, json_czml, indent=4)

def latlong2eci_cities(latlon_cities, specific_time):
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

    wgs72 = ellipsoid.Ellipsoid(model='wgs72')
    
    eci_cities = dict()
    for city_name in latlon_cities.keys():
        lat, lon = latlon_cities[city_name]
        eci_coord = geodetic2eci(lat=lat, lon=lon, alt=0, t=specific_time, ell=wgs72)

        eci_cities[city_name] = eci_coord

    return eci_cities

def generate_paths_in_time(start_time, end_time, time_interval):
    pass

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
                line = lines.rstrip('\n')
                if line.startswith('0'):
                    SAT_Name = line.replace('\t',' ')[2:]
                elif line.startswith('1'):
                    l1 = line
                elif line.startswith('2'):
                    l2 = line
                    SAT_ID = line.split('\t')[1].zfill(5)
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
            KNN_Results = pickle.load(f)
            print(KNN_filepath[:-3]+'pickle Loaded')
            return KNN_Results

    with open(KNN_filepath, 'r') as f:
        lines = f.readlines()
        KNN_Results = list()

        for line in lines:
            if line.startswith('%'):
                continue
            leg_info = line.strip('\n').split('\t')

            knn = dict()
            knn['Pair']     = leg_info[0], leg_info[1]
            knn['Interval'] = float(leg_info[4]), float(leg_info[5])

            KNN_Results.append(knn)
    return KNN_Results

def generate_timestep_map(KNN_Results, start_time, end_time, timestep):
    curr_time = start_time
    timestep_map = dict()
    for knn in KNN_Results:
        while curr_time < end_time:
            if curr_time not in timestep_map:
                timestep_map[curr_time] = list()
            timestep_map[curr_time].append(knn['Pair'])
            curr_time += timestep
    return timestep_map

def propagate_SATs(SATs, time):
    for SAT_Name in SATs.keys():
        pos, vel = SATs[SAT_Name][1].propagate(time)
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
    return graph

def find_closest_SAT_with_city(SATs, eci_cities):
    import numpy as np

    for city in eci_cities.keys():
        dist = 0
        for SAT in SATs:
            eci_cities[city]
    pass

if __name__ == "__main__":
    import datetime as dt
    from color_constants import *
    from LatLong_cities import latlong_cities

    KNN_filepath      = './kNN_data/spaceWideKNN_results.txt'
    TLE_filepath        = './kNN_data/latest_starlink_leo.tle'

    czml_filepath       = './kNN_data/pretty_orbit.czml'
    czml_result_filepath= './kNN_data/path_added_orbit_euc.czml'

    start_time  = None
    end_time    = None
    timestep    = None

    KNN_Results = read_KNN_file(KNN_filepath)
    timestep_map = generate_timestep_map(KNN_Results, start_time, end_time, timestep)
    SATs = parse_TLE(TLE_filepath)

    curr_time = start_time
    while curr_time <= end_time:    
        currtime_pairs = timestep_map[curr_time]
        graph = generate_graph_with_pairs(currtime_pairs)
        
        SATs = propagate_SATs(SATs, curr_time)
        eci_cities  = latlong2eci_cities(latlong_cities, curr_time)

        city_SAT = find_closest_SAT_with_city(SATs, eci_cities)

    # legs = read_graph_file(KNN_filepath)
    czml_list = read_czml_file(czml_filepath)
    change_color_of_satellites(satellite_color, satellite_outcolor, satellite_size, czml_list)
    add_cities_to_czml(latlong_cities, czml_list, city_linecolor, city_outlinecolor, city_labelcolor, city_labelbackgroundcolor)

    ##############################################################################
    ############################### weighted graph ###############################
    ##############################################################################

    # graph = generate_graph_with_legs(legs)

    calc_start = dt.datetime.now()
    dijkstra_graph, optimal_path, distance = dijkstra_with_graph(graph, Seoul_SATID, SanFrancisco_SATID)
    elapsed = dt.datetime.now() - calc_start
    add_path_to_czml(str(Seoul_SATID)+' to '+str(SanFrancisco_SATID), optimal_path, czml_list, path_linecolor, path_outlinecolor)
    connect_cities_with_satellites(list(cities.keys())[0], Seoul_SATID, list(cities.keys())[1], SanFrancisco_SATID, czml_list, city_linecolor, city_outlinecolor)
    
    calc_start = dt.datetime.now()
    dijkstra_graph, optimal_path, distance = dijkstra_with_graph(graph, SanFrancisco_SATID, NewYork_SATID)
    elapsed += dt.datetime.now() - calc_start
    add_path_to_czml(str(SanFrancisco_SATID)+' to '+str(NewYork_SATID), optimal_path, czml_list, path_linecolor, path_outlinecolor)
    connect_cities_with_satellites(list(cities.keys())[1], SanFrancisco_SATID, list(cities.keys())[2], NewYork_SATID, czml_list, city_linecolor, city_outlinecolor)

    calc_start = dt.datetime.now()
    dijkstra_graph, optimal_path, distance = dijkstra_with_graph(graph, NewYork_SATID, London_SATID)
    elapsed += dt.datetime.now() - calc_start
    add_path_to_czml(str(NewYork_SATID)+' to '+str(London_SATID), optimal_path, czml_list, path_linecolor, path_outlinecolor)
    connect_cities_with_satellites(list(cities.keys())[2], NewYork_SATID, list(cities.keys())[3], London_SATID, czml_list, city_linecolor, city_outlinecolor)

    calc_start = dt.datetime.now()
    dijkstra_graph, optimal_path, distance = dijkstra_with_graph(graph, London_SATID, Seoul_SATID)
    elapsed += dt.datetime.now() - calc_start
    add_path_to_czml(str(London_SATID)+' to '+str(Seoul_SATID), optimal_path, czml_list, path_linecolor, path_outlinecolor)
    connect_cities_with_satellites(list(cities.keys())[3], London_SATID, list(cities.keys())[0], Seoul_SATID, czml_list, city_linecolor, city_outlinecolor)

    print('Euclidian Distance Dijkstra : '+str(elapsed.total_seconds()))

    ##############################################################################
    ############################# topological graph ##############################
    ##############################################################################

    path_linecolor      = [255, 255, 0, 255]
    city_linecolor      = [102, 0, 51, 255]

    graph = generate_topological_graph_with_legs(legs)

    calc_start = dt.datetime.now()
    dijkstra_graph, optimal_path, distance = dijkstra_with_graph(graph, Seoul_SATID, SanFrancisco_SATID)
    elapsed = dt.datetime.now() - calc_start
    add_path_to_czml(str(Seoul_SATID)+' to '+str(SanFrancisco_SATID)+'_top', optimal_path, czml_list, path_linecolor, path_outlinecolor)
    # connect_cities_with_satellites(list(cities.keys())[0], Seoul_SATID, list(cities.keys())[1], SanFrancisco_SATID, czml_list, city_linecolor, city_outlinecolor)
    
    calc_start = dt.datetime.now()
    dijkstra_graph, optimal_path, distance = dijkstra_with_graph(graph, SanFrancisco_SATID, NewYork_SATID)
    elapsed += dt.datetime.now() - calc_start
    add_path_to_czml(str(SanFrancisco_SATID)+' to '+str(NewYork_SATID)+'_top', optimal_path, czml_list, path_linecolor, path_outlinecolor)
    # connect_cities_with_satellites(list(cities.keys())[1], SanFrancisco_SATID, list(cities.keys())[2], NewYork_SATID, czml_list, city_linecolor, city_outlinecolor)

    calc_start = dt.datetime.now()
    dijkstra_graph, optimal_path, distance = dijkstra_with_graph(graph, NewYork_SATID, London_SATID)
    elapsed += dt.datetime.now() - calc_start
    add_path_to_czml(str(NewYork_SATID)+' to '+str(London_SATID)+'_top', optimal_path, czml_list, path_linecolor, path_outlinecolor)
    # connect_cities_with_satellites(list(cities.keys())[2], NewYork_SATID, list(cities.keys())[3], London_SATID, czml_list, city_linecolor, city_outlinecolor)

    calc_start = dt.datetime.now()
    dijkstra_graph, optimal_path, distance = dijkstra_with_graph(graph, London_SATID, Seoul_SATID)
    elapsed += dt.datetime.now() - calc_start
    add_path_to_czml(str(London_SATID)+' to '+str(Seoul_SATID)+'_top', optimal_path, czml_list, path_linecolor, path_outlinecolor)
    # connect_cities_with_satellites(list(cities.keys())[3], London_SATID, list(cities.keys())[0], Seoul_SATID, czml_list, city_linecolor, city_outlinecolor)

    print('Topological Distance Dijkstra : '+str(elapsed.total_seconds()))

    save_czml(czml_result_filepath, czml_list)

    pass
