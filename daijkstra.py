def read_graph_file(graph_filepath):
    with open(graph_filepath, 'r') as f:
        lines = f.readlines()
        legs = list()

        for line in lines:
            if line.startswith('%'):
                continue
            leg_info = line.strip('\n').split('\t')

            leg = dict()
            leg['PrimaryID'] = int(leg_info[0])
            leg['SecondaryID'] = int(leg_info[1])
            leg['MinDistance'] = float(leg_info[2])

            legs.append(leg)
    return legs

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

def generate_graph_with_legs(legs):
    from dijkstra import Graph

    graph = Graph()
    for leg in legs:
        graph.add_edge(leg['PrimaryID'], leg['SecondaryID'], leg['MinDistance'])
    return graph

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
    for city in cities.keys():
        c = dict()
        c['id'] = city
        c['description'] = city
        c['availability'] = "2021-06-10T00:00:00+00:00/2021-06-10T01:00:00+00:00"
        c['position'] = dict()
        c['position']['epoch'] = "2021-06-10T00:00:00+00:00"
        c['position']['cartesian'] = [
            0.0,
            cities[city][0],
            cities[city][1],
            cities[city][2],
            300.0,
            cities[city][0]+1,
            cities[city][1]+1,
            cities[city][2]+1
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
        c['label']['text'] = city
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


if __name__ == "__main__":
    import datetime as dt

    graph_filepath = './graph_src_new_1500km.txt'
    czml_filepath = './pretty_orbit.czml'
    czml_result_filepath = './path_added_orbit_euc.czml'

    satellite_color     = [255, 255, 255, 255]
    satellite_outcolor  = [0, 255, 0, 255]
    satellite_size      = 3
    city_linecolor      = [102, 0, 51, 255]
    city_outlinecolor   = [255, 255, 255, 255]
    city_labelcolor     = [255, 255, 255, 255]
    city_labelbackgroundcolor = [0, 0, 0, 255]
    path_linecolor      = [255, 0, 0, 255]
    path_outlinecolor   = [255, 255, 255, 255]

    Seoul_SATID         = 47548
    SanFrancisco_SATID  = 46684
    NewYork_SATID       = 46547
    London_SATID        = 47895

    Seoul           = [4584534.119813906, 2163242.740802608, 3857817.357544671]
    London          = [-805368.136465597, -3893488.348167617, 4970564.680860108]
    SanFrancisco    = [-3614711.7669382123, 3517060.8296958776, 3891440.05084575]
    NewYork         = [-4817644.446847552, -368069.54799630004, 4149708.9933215496]
    Singapore       = [6372122.384338922, 240885.27902601045, 136788.46629895904]
    cities = {'Seoul':Seoul, 'SanFrancisco':SanFrancisco, 'NewYork':NewYork, 'London':London, 'Singapore':Singapore}

    legs = read_graph_file(graph_filepath)
    czml_list = read_czml_file(czml_filepath)
    change_color_of_satellites(satellite_color, satellite_outcolor, satellite_size, czml_list)
    add_cities_to_czml(cities, czml_list, city_linecolor, city_outlinecolor, city_labelcolor, city_labelbackgroundcolor)

    ##############################################################################
    ############################### weighted graph ###############################
    ##############################################################################

    graph = generate_graph_with_legs(legs)

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
