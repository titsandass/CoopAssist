def parse_time_interval(start_time, end_time, time_step, verbose=False):
    import math

    time_step = int(time_step)

    start_time = int(start_time)
    start_time = start_time / time_step
    start_time = math.floor(start_time) * time_step
    
    end_time = int(end_time)
    end_time = end_time / time_step
    end_time = math.ceil(end_time) * time_step

    if verbose:
        print('Start Time : {}'.format(start_time))
        print('End Time : {}'.format(end_time))
        print('Time Step : {}'.format(time_step))
    
    return start_time, end_time, time_step

def check_file_existance(verbose=False):
    import consts_Filepaths
    from os import path

    TNN_pickle_exists = False
    TLE_pickle_exists = False
    SAT_position_map_pickle_exists = False
    TimestepMap_pickle_exists = False
    LVLHSphere_pickle_exists = False

    if path.exists(consts_Filepaths.TNN_pickle_filepath):
        TNN_pickle_exists = True
    if path.exists(consts_Filepaths.TLE_pickle_filepath):
        TLE_pickle_exists = True
    if path.exists(consts_Filepaths.SAT_position_map_pickle_filepath):
        SAT_position_map_pickle_exists = True
    if path.exists(consts_Filepaths.TimestepMap_pickle_filepath):
        TimestepMap_pickle_exists = True
    if path.exists(consts_Filepaths.LVLHSphere_pickle_filepath):
        LVLHSphere_pickle_exists = True
    
    if verbose:
        if TNN_pickle_exists:
            print('TNN pickle Exists')
        if TLE_pickle_exists:
            print('TLE pickle Exists')
        if TimestepMap_pickle_exists:
            print('Timestep Map pickle Exists')
        if LVLHSphere_pickle_exists:
            print('LVLH Sphere pickle Exists')
        if SAT_position_map_pickle_exists:
            print('SAT Position Map pickle Exists')

    return TNN_pickle_exists, TLE_pickle_exists, SAT_position_map_pickle_exists, TimestepMap_pickle_exists, LVLHSphere_pickle_exists

def read_TNN_file(TNN_filepath, verbose=False):
    import pickle

    with open(TNN_filepath, 'r') as f:
        if verbose:
            print('Generating TNN pickle')
        lines = f.readlines()
        TNN_Results = list()

        pivot_date = None
        for line in lines:
            if line.startswith('%'):
                continue
            leg_info = line.strip('\n').split('\t')
            
            if pivot_date is None:
                pivot_date = leg_info[6]+'-'+leg_info[7].zfill(2)+'-'+leg_info[8].zfill(2)+'T'+leg_info[9].zfill(2)+':'+leg_info[10].zfill(2)+':'+leg_info[11]
                
            tnn = dict()
            tnn['Pair']     = leg_info[0], leg_info[1]
            tnn['Interval'] = float(leg_info[4]), float(leg_info[5])

            TNN_Results.append(tnn)

    with open(TNN_filepath[:-3]+'pickle', 'wb') as f:
        pickle.dump((pivot_date, TNN_Results), f)
        if verbose:
            print('TNN Result Saved : {}'.format(TNN_filepath[:-3]+'pickle'))

    return pivot_date, TNN_Results

def generate_timestep_map(TNN_pickle_exists, timestep, verbose=False):
    import os
    import pickle
    import consts_Filepaths

    if TNN_pickle_exists:
        if verbose:
            print('Loading Timestep Map')
        with open(consts_Filepaths.TimestepMap_pickle_filepath, 'rb') as f:
            timestep_map = pickle.load(f)

    else:
        _, TNN_Results = read_TNN_file(consts_Filepaths.TNN_filepath, verbose)

        if verbose:
            print('Generating Timestep Map')

        timestep_map = dict()
        for i, knn in enumerate(TNN_Results):
            curr_time = knn['Interval'][0]
            while curr_time <= min(end_time, knn['Interval'][1]):
                if curr_time not in timestep_map:
                    timestep_map[curr_time] = list()
                timestep_map[curr_time].append(knn['Pair'])
                curr_time += timestep

            if verbose:
                print (str(i)+'/'+str(len(TNN_Results)))

        with open(consts_Filepaths.TimestepMap_pickle_filepath, 'wb') as f:
            pickle.dump(timestep_map, f)

    if verbose:
        print('Done')

    return timestep_map

def generate_SATs_from_TLE(TLE_pickle_exists, verbose=False):
    import consts_Filepaths
    import pickle

    if TLE_pickle_exists:
        if verbose:
            print('Loading TLE pickle')
        with open(consts_Filepaths.TLE_pickle_filepath, 'rb') as f:
            SATs = pickle.load(f)

    else:
        from sgp4.earth_gravity import wgs72
        from sgp4.io import twoline2rv

        if verbose:
            print('Generating TLE pickle')

        SATs = dict()
        with open(consts_Filepaths.TLE_filepath, 'r') as f:
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

        with open(consts_Filepaths.TLE_pickle_filepath, 'wb') as f:
            pickle.dump(SATs, f)

    if verbose:
        print('Done')

    return SATs

def generate_SAT_position_map(SAT_position_map_pickle_exists, time_step=None, SATs=None, pivot_time=None, verbose=False):
    import pickle
    import consts_Filepaths

    if SAT_position_map_pickle_exists:
        if verbose:
            print('Loading SAT Position Map pickle')
            
        with open(consts_Filepaths.SAT_position_map_pickle_filepath, 'rb') as f:
            SAT_position_map = pickle.load(f)
    else:
        import datetime

        if verbose:
            print('Generating SAT Position Map pickle')

        start_time  = 0
        end_time    = 86390

        SAT_position_map = dict()
        for i, SAT_Name in enumerate(SATs.keys()):
            if verbose:
                print('{} / {}'.format(i+1, len(SATs.keys())))

            SAT_position_map[SAT_Name] = dict()

            curr_time = start_time
            while curr_time <= end_time:
                time = pivot_time + datetime.timedelta(seconds=curr_time)
                pos, vel = SATs[SAT_Name][1].propagate(time.year, time.month, time.day, time.hour, time.minute, time.second + time.microsecond/1000000)
                SAT_position_map[SAT_Name][curr_time] = (pos, vel)
                curr_time += time_step

        with open(consts_Filepaths.SAT_position_map_pickle_filepath, 'wb') as f:
            pickle.dump(SAT_position_map, f)
            
        if verbose:
            print('Done')
        return SAT_position_map

def generate_graph_with_pairs(SAT_Pairs, verbose=False):
    from dijkstra import Graph

    if verbose:
        print('Generating Graph Structure')

    graph = Graph()
    for pair in SAT_Pairs:
        graph.add_edge(pair[0], pair[1], 1)
        graph.add_edge(pair[1], pair[0], 1)

    if verbose:
        print('Done')

    return graph

def latlong2eci_cities(latlon_cities, specific_time, origin_city, dest_city):
    from pymap3d import geodetic2eci, ellipsoid
    import datetime as dt

    wgs72 = ellipsoid.Ellipsoid(model='wgs72')
    
    eci_cities = dict()
    for city_name in latlon_cities.keys():
        if(city_name == origin_city or city_name == dest_city):
            lat, lon = latlon_cities[city_name]
            eci_coord = geodetic2eci(lat=lat, lon=lon, alt=0, t=specific_time, ell=wgs72)
            eci_cities[city_name] = eci_coord

    return eci_cities

def find_closest_SAT_with_city(SAT_position_map, curr_time, eci_cities, origin_city, dest_city):
    import numpy as np

    pairs = list()
    for city in eci_cities.keys():
        if(city == origin_city or city == dest_city):
            dist = None
            closest = None
            for SAT_Name in SAT_position_map.keys():
                a = np.array(eci_cities[city]).flatten()/1000
                b = np.array(SAT_position_map[SAT_Name][curr_time][0])
                new_dist = np.linalg.norm(a-b)
                if dist == None:
                    dist = new_dist
                    closest = SAT_Name
                elif new_dist < dist:
                    dist = new_dist
                    closest = SAT_Name
            pairs.append((city, closest))
    return pairs

def dijkstra_with_graph(graph, start_node, end_node):
    from dijkstra import DijkstraSPF

    dijkstra_graph = DijkstraSPF(graph, start_node)
    optimal_path = dijkstra_graph.get_path(end_node)
    distance = dijkstra_graph.get_distance(end_node)

    return dijkstra_graph, optimal_path, distance

def eci_to_LVLH(pos, vel):
    import numpy as np
    
    X = pos/np.linalg.norm(pos)
    Z = np.cross(pos, vel)/np.linalg.norm(np.cross(pos, vel))
    Y = np.cross(Z, X)

    return X, Y, Z

def add_SAT_to_LVLH_sphere(SAT_position_map, curr_time, optimal_path, SAT_Sphere_dict, verbose=False):
    import numpy as np

    if verbose:
        print('Adding SATs to LVLH Sphere')

    for i, SATID in enumerate(optimal_path):
        if SATID == optimal_path[0] or SATID == optimal_path[-2] or SATID == optimal_path[-1]:
            continue
        SAT1ID = SATID
        SAT2ID = optimal_path[i+1]

        pos1 = np.array(SAT_position_map[SAT1ID][curr_time][0])
        vel1 = np.array(SAT_position_map[SAT1ID][curr_time][1])

        pos2 = np.array(SAT_position_map[SAT2ID][curr_time][0])
        vel2 = np.array(SAT_position_map[SAT2ID][curr_time][1])

        if SAT1ID not in SAT_Sphere_dict:
            SAT_Sphere_dict[SAT1ID] = list()
        if SAT2ID not in SAT_Sphere_dict:
            SAT_Sphere_dict[SAT2ID] = list()

        X1, Y1, Z1 = eci_to_LVLH(pos1, vel1)
        rel_pos21 = pos2 - pos1
        translation21 = np.array([X1, Y1, Z1]).T
        translated_rel_pos21 = np.linalg.inv(translation21)@rel_pos21
        SAT_Sphere_dict[SAT1ID].append(translated_rel_pos21/np.linalg.norm(translated_rel_pos21))

        X2, Y2, Z2 = eci_to_LVLH(pos2, vel2)
        rel_pos12 = pos1 - pos2
        translation12 = np.array([X2, Y2, Z2]).T
        translated_rel_pos12 = np.linalg.inv(translation12)@rel_pos12
        SAT_Sphere_dict[SAT2ID].append(translated_rel_pos12/np.linalg.norm(translated_rel_pos12))

    if verbose:
        print('Done')

def save_sat_sphere(sat_sphere_dict, verbose=False):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import consts_Filepaths

    if verbose:
        print('Generating LVLH Sphere Plot')

    plt_file = consts_Filepaths.LVLHSphere_pickle_filepath[:-6] + 'png'
    f = plt.figure()
    plt.rcParams["figure.figsize"] = (16, 17)
    ax = f.add_subplot(projection='3d')

    ax.set_box_aspect((1,1,1))

    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)

    ax.set_xticks([-1, 1])
    ax.set_yticks([-1, 1])
    ax.set_zticks([-1, 1])

    ax.set_xlabel("X", labelpad=0)
    ax.set_ylabel("Y", labelpad=0)
    ax.set_zlabel("Z", labelpad=0)

    ax.view_init(20, 45)
    
    r = 0.93
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:50j, 0.0:2.0*pi:50j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='white', alpha=0.5, linewidth=1)

    xs = list()
    ys = list()
    zs = list()
    for SATID in sat_sphere_dict.keys():
        for pos in sat_sphere_dict[SATID]:
            x, y, z = pos
            xs.append(x)
            ys.append(y)
            zs.append(z)

    ax.scatter(xs, ys, zs, color="blue", marker='o', s=1, depthshade=True, alpha=1)
        
    if os.path.isfile(plt_file):
        os.remove(plt_file)
    plt.title('Position Vectors of connected SATs in Link Minimization for 24h')
    plt.savefig(plt_file, dpi=200)
    plt.clf()

    if verbose:
        print('Saved {}'.format(plt_file))

if __name__=='__main__':
    import consts_Filepaths
    import consts_Cities
    import sys
    import datetime
    import time

    verbose = True

    start_time  = 0
    end_time    = 86390
    time_step   = 10

    pivot_date = '2021-08-01T00:00:00.000'
    pivot_time = datetime.datetime.strptime(pivot_date, '%Y-%m-%dT%H:%M:%S.%f')

    origin_city = 'Seoul'
    dest_city   = 'NewYork'

    start_time, end_time, time_step = parse_time_interval(start_time, end_time, time_step, verbose)

    TNN_pickle_exists, TLE_pickle_exists, SAT_position_map_pickle_exists, TimestepMap_pickle_exists, LVLHSphere_pickle_exists = check_file_existance(verbose)

    timestep_map    = generate_timestep_map(TNN_pickle_exists, time_step, verbose)
    SATs            = generate_SATs_from_TLE(TLE_pickle_exists, verbose)
    SAT_position_map= generate_SAT_position_map(SAT_position_map_pickle_exists, time_step, SATs, pivot_time, verbose=verbose)
    SAT_Sphere_dict = dict()

    loop_start = time.time()
    curr_time = start_time
    while curr_time <= end_time: 
        if verbose:
            print('{} / {}'.format(curr_time, end_time))

        utc_time        = pivot_time + datetime.timedelta(seconds=curr_time)
        currtime_pairs  = timestep_map[curr_time]
        graph           = generate_graph_with_pairs(currtime_pairs, verbose)
        eci_cities      = latlong2eci_cities(consts_Cities.latlong_cities, utc_time, origin_city, dest_city)
        city_SAT        = find_closest_SAT_with_city(SAT_position_map, curr_time, eci_cities, origin_city, dest_city)
        
        for (city, SAT) in city_SAT:
            graph.add_edge(city, SAT, 1)
            graph.add_edge(SAT, city, 1)
        
        _, optimal_path, _ = dijkstra_with_graph(graph, origin_city, dest_city)
        add_SAT_to_LVLH_sphere(SAT_position_map, curr_time, optimal_path, SAT_Sphere_dict, verbose)

        curr_time += time_step

    if verbose:
        print("Total Elapsed Time : {}".format(time.time() - loop_start))

    save_sat_sphere(SAT_Sphere_dict, verbose)