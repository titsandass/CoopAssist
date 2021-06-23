dpi = 200
file_type = ".png"
class CDMList():
    def __init__(self, cdms = []):
        self._CDMs = cdms

    def __contains__(self, item):
        if not self._CDMs:
            for CDM in self._CDMs:
                if (CDM["SAT_1_ID"], CDM["SAT_2_ID"]) == (item["SAT_1_ID"], item["SAT_2_ID"])  or (CDM["SAT_2_ID"], CDM["SAT_1_ID"]) == (item["SAT_1_ID"], item["SAT_2_ID"]):
                    return True, self._CDMs.index(CDM)
        
        return False, None


def read_tle(filename):
    with open(filename, "r") as tle:
        tle_lines = tle.readlines()

        json_tle = list()
        RSO = None

        for tle_line in tle_lines:
            if tle_line.startswith('0'):
                sat_name = tle_line.split()
                del(sat_name[0])

                if RSO is not None:
                    json_tle.append(RSO)

                RSO = dict()
                RSO_name = str()

                for name in sat_name:
                    RSO_name += name + " "

                RSO['SATNAME'] = RSO_name[:-1]

            elif tle_line.startswith('1'):
                RSO['s'] = tle_line

            elif tle_line.startswith('2'):
                RSO['t'] = tle_line

                RSO['SAT_ID'] = tle_line.split()[1].zfill(5)

    return json_tle


def read_cdm(filename):
    import json

    with open(filename, "r") as cdm:
        json_cdm = json.load(cdm)
        CDMs = CDMList(json_cdm)

    return CDMs
    

def read_ppdb(filename):
    import datetime as dt
    ppdbs = []

    with open(filename, "r") as ppdb_data:
        lines = ppdb_data.readlines()
        for line in lines:
            if line.startswith("%"):
                continue
            
            ppdb = dict()
            data = line.split()

            ppdb["SAT1ID"] = data[0]
            ppdb["SAT2ID"] = data[1]
            ppdb["MinDistance"] = data[2]

            if data[11] == "60.000":
                data[11] = "59.000"
                std_date = " ".join(data[6:])
                TCA = dt.datetime.strptime(std_date, "%Y %m %d %H %M %S.%f")
                TCA += dt.timedelta(seconds=1)
            else:
                std_date = " ".join(data[6:])
                TCA = dt.datetime.strptime(std_date, "%Y %m %d %H %M %S.%f")

            CAStart     = TCA + dt.timedelta(seconds=float(data[4]) - float(data[3]))
            CAEnd       = TCA + dt.timedelta(seconds=float(data[5]) - float(data[3]))

            ppdb["TCA"]     = TCA.strftime("%Y-%m-%dT%H:%M:%S.%f")
            ppdb["CAStart"] = CAStart.strftime("%Y-%m-%dT%H:%M:%S.%f")
            ppdb["CAEnd"]   = CAEnd.strftime("%Y-%m-%dT%H:%M:%S.%f")

            ppdbs.append(ppdb)
    return ppdbs


def generate_RSO(tle):
    from sgp4.earth_gravity import wgs72
    from sgp4.io import twoline2rv
    import pickle

    RSOs = dict()

    for RSO in tle:
        sat_id = RSO['SAT_ID']
        RSOs[sat_id] = (RSO['SATNAME'], twoline2rv(RSO["s"], RSO["t"], wgs72))

    with open("./COOP_data/RSO.pickle", "wb") as f:
        pickle.dump(RSOs, f)

    return RSOs


def compare_SGP4_N_PPDB(RSOs, PPDBs, RADIUS_OF_EARTH):
    import datetime as dt
    import numpy as np

    # keys = ["SAT1_ID", "SAT2_ID", "CDM_ID", "CDM_TCA", "CDM_DCA", "SGP4_DCA@CDM_TCA", "SGP4_SAT1_POS@CDM_TCA", "SGP4_SAT2_POS@CDM_TCA", "PPDB_TCA", "PPDB_DCA", "SGP4_DCA@PPDB_TCA", "SGP4_SAT1_POS@PPDB_TCA", "SGP4_SAT2_POS@PPDB_TCA", "PPDB_TCA - CDM_TCA", "PPDB_DCA - CDM_DCA"]
    # header = "\t".join(keys) + "\n"

    # filtered_CDM = _filter_cdm_to_latest(CDMLists)

    total_results = list()
    not_in_tle_IDs = set()

    for PPDB in PPDBs:
        if PPDB["SAT1ID"].zfill(5) not in RSOs.keys():
            not_in_tle_IDs.add(PPDB["SAT1ID"].zfill(5))
            print(PPDB["SAT1ID"].zfill(5) + "is not is TLE")
            continue
        elif PPDB["SAT2ID"].zfill(5) not in RSOs.keys():
            not_in_tle_IDs.add(PPDB["SAT2ID"].zfill(5))
            print(PPDB["SAT2ID"].zfill(5) + "is not is TLE")
            continue

        sat1 = RSOs[PPDB["SAT1ID"].zfill(5)][1]
        sat2 = RSOs[PPDB["SAT2ID"].zfill(5)][1]

        PPDB_TCA = dt.datetime.strptime(PPDB["TCA"], "%Y-%m-%dT%H:%M:%S.%f")

        PPDB_sat1_pos, _ = sat1.propagate(PPDB_TCA.year, PPDB_TCA.month, PPDB_TCA.day, PPDB_TCA.hour, PPDB_TCA.minute, PPDB_TCA.second + PPDB_TCA.microsecond/1000000)
        PPDB_sat2_pos, _ = sat2.propagate(PPDB_TCA.year, PPDB_TCA.month, PPDB_TCA.day, PPDB_TCA.hour, PPDB_TCA.minute, PPDB_TCA.second + PPDB_TCA.microsecond/1000000)

        PPDB_DCA = np.linalg.norm((np.asarray(PPDB_sat2_pos) - np.asarray(PPDB_sat1_pos)))

        result = dict()
        result["SAT1_ID"]               = PPDB["SAT1ID"].zfill(5)
        result["SAT2_ID"]               = PPDB["SAT2ID"].zfill(5)

        result["PPDB_TCA"]              = PPDB["TCA"]
        result["PPDB_DCA"]              = PPDB["MinDistance"]
        result["SGP4_DCA@PPDB_TCA"]     = PPDB_DCA
        result["SGP4_SAT1_POS@PPDB_TCA"]= PPDB_sat1_pos
        result["SGP4_SAT1_ALT@PPDB_TCA"]= np.linalg.norm(np.asarray(PPDB_sat1_pos)) - RADIUS_OF_EARTH
        result["SGP4_SAT2_POS@PPDB_TCA"]= PPDB_sat2_pos
        result["SGP4_SAT2_ALT@PPDB_TCA"]= np.linalg.norm(np.asarray(PPDB_sat2_pos)) - RADIUS_OF_EARTH

        total_results.append(result)
    return total_results, not_in_tle_IDs


def _filter_cdm_to_latest(CDMLists):
    filtered_CDM = CDMList()
    for CDM in CDMLists._CDMs:
        exist, index = filtered_CDM.__contains__(CDM)
        if exist:
            import datetime as dt
           
            prev = dt.datetime.strptime(filtered_CDM._CDMs[index]["CREATED"], "%Y-%m-%d %H:%M:%S.%f")
            curr = dt.datetime.strptime(CDM["CREATED"], "%Y-%m-%d %H:%M:%S.%f")

            if prev < curr:
                del(filtered_CDM._CDMs[index])
                filtered_CDM._CDMs.append(CDM)

        else:
            filtered_CDM._CDMs.append(CDM)
    
    return filtered_CDM


def save_result(result_filename, total_results, not_in_tle_IDs):
    import pickle
    with open(result_filename, "w") as f:
        f.write("\t".join(not_in_tle_IDs) + "\n")

        keys = ["SAT1_ID", "SAT2_ID", "PPDB_TCA", "PPDB_DCA", "SGP4_DCA@PPDB_TCA", "SGP4_SAT1_POS@PPDB_TCA", "SGP4_SAT1_ALT@PPDB_TCA", "SGP4_SAT2_POS@PPDB_TCA", "SGP4_SAT2_ALT@PPDB_TCA"]
        header = "\t".join(keys) + "\n"

        f.write(header)
        for result in total_results:
            for key in result.keys():
                f.write(str(result[key]))
                f.write("\t")
            f.write("\n")
    
    with open(result_filename[:-4]+".pickle", "wb") as f:
        pickle.dump(total_results, f)


def calculate_and_save_conjunction_histogram(total_results, threshold_distance, distance_interval, bin_interval, save_individual, height_interval=None):
    import numpy as np
    import matplotlib.pyplot as plt

    distances = np.arange(distance_interval, threshold_distance + distance_interval, distance_interval)
    histogram = {distance : list() for distance in distances}

    bins = [i*bin_interval for i in range(int(2000/bin_interval))]

    for result in total_results:
        for distance in distances:
            if float(result["PPDB_DCA"]) <= distance:
                histogram[distance].append(float(result["SGP4_SAT1_ALT@PPDB_TCA"]))

    colors=["#4f8f1a",
            "#75a333",
            "#99b84c",
            "#dee381",
            "#fff89d",
            "#f9d76d",
            "#f6b442",
            "#f48d1d",
            "#f15f01",
            "#eb0909"]

    plt.rcParams["figure.figsize"] = (8, 8)
    plt.figure()

    for i, distance in enumerate(reversed(distances)):
        plt.title("Conjunction Distribution at Threshold")
        plt.xlabel("# Conjunctions")
        plt.ylabel("Altitude (km)")
        (n, bins, patches) = plt.hist(x=histogram[distance], bins=bins, orientation="horizontal", color=colors[i], label=str(distance)+" km")

        if i==0:
            xs = [patch._width for patch in patches]

        if height_interval:
            plt.ylim(height_interval[0], height_interval[1])
            maxX = 0.0
            for patch in patches:
                if patch.xy[1] < height_interval[0] or patch.xy[1] >= height_interval[1]:
                    del(patch)
                    # patch.set_visible(False)
                else:
                    if patch._width > maxX:
                        maxX = patch._width
            plt.xlim(0, maxX+10)
        else:
            plt.ylim(0,2000)

        patches.label = str(distance)+" km"
        plt.legend(title="Threshold", loc='lower right')      

    # Label Location
    # 'best'            0
    # 'upper right'	    1
    # 'upper left'	    2
    # 'lower left'	    3
    # 'lower right'	    4
    # 'right'	        5
    # 'center left'	    6
    # 'center right'    7
    # 'lower center'    8
    # 'upper center'    9
    # 'center'	        10

        # # plt.tight_layout()

        if save_individual:
            plt_file = "./COOP_data/output/"+str(distance)+"_histogram"+file_type
            if os.path.isfile(plt_file):
                os.remove(plt_file)
            plt.savefig(plt_file, dpi=dpi)
            plt.clf()

    # plt.hlines(860, 0, 8000, colors="black", linestyles="dashed")
    if not save_individual:
        plt.xlim(0, max(xs)+50)
        plt_file = "./COOP_data/output/total_histogram"+file_type
        if os.path.isfile(plt_file):
            os.remove(plt_file)
        plt.savefig(plt_file, dpi=dpi)
        plt.clf()
    
    return histogram


def draw_conjunction_polarplane_2D(total_results, RADIUS_OF_EARTH, show_full_quadrant):
    import numpy as np
    import math
    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = (15, 15)
    
    Rs = list()
    Thetas = list()

    for result in total_results:
        sat1_pos    = np.asarray(result["SGP4_SAT1_POS@PPDB_TCA"])
        x, y, z     = sat1_pos
        radius      = np.linalg.norm(sat1_pos) - RADIUS_OF_EARTH
        theta       = math.atan(z/np.linalg.norm(sat1_pos[:2]))

        if show_full_quadrant:
            theta = _reflect_for_full_quadrant(x, y, z, theta)

        Rs.append(radius)
        Thetas.append(theta)

        sat2_pos    = np.asarray(result["SGP4_SAT2_POS@PPDB_TCA"])
        x, y, z     = sat2_pos
        radius      = np.linalg.norm(sat2_pos) - RADIUS_OF_EARTH
        theta       = math.atan(z/np.linalg.norm(sat2_pos[:2]))

        if show_full_quadrant:
            theta = _reflect_for_full_quadrant(x, y, z, theta)

        Rs.append(radius)
        Thetas.append(theta)


    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    ax.set_ylim(0,2000)
    ax.set_yticks(np.arange(0,2100,200))
    ax.set_yticks(np.arange(0,2100,100), minor = True)
    ax.tick_params(axis='y', which = 'minor' , width = 1)
    ax.scatter(Thetas, Rs, color="#33AA33", s=0.1, alpha=0.5)

    if not show_full_quadrant:
        ax.set_thetamin(0)
        ax.set_thetamax(90)

    plt_file = "./COOP_data/output/polar_conjunction"+file_type
    if os.path.isfile(plt_file):
        os.remove(plt_file)
    plt.tight_layout()
    plt.savefig(plt_file, dpi=dpi)
    plt.clf()

    return Rs, Thetas


def _reflect_for_full_quadrant(x, y, z, theta):
    import math

    if y < 0:
        return (math.radians(180) - theta)
    else:
        return theta


def draw_RSO_histogram_at_time(RSOs, str_time, RADIUS_OF_EARTH, bin_interval):
    import datetime as dt
    import numpy as np
    import math
    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = (8, 8)
    plt.figure()
    
    colors=["#4f8f1a",
        "#75a333",
        "#99b84c",
        "#dee381",
        "#fff89d",
        "#f9d76d",
        "#f6b442",
        "#f48d1d",
        "#f15f01",
        "#eb0909"]

    # distances = np.arange(distance_interval, threshold_distance + distance_interval, distance_interval)
    # histogram = {distance : list() for distance in distances}
    bins = [i*bin_interval for i in range(int(2000/bin_interval))]

    time = dt.datetime.strptime(str_time, "%Y-%m-%dT%H:%M:%S.%f")

    altitudes = list()
    for SATID in RSOs.keys():
        sat_pos, _  = RSOs[SATID][1].propagate(time.year, time.month, time.day, time.hour, time.minute, time.second + time.microsecond/1000000)
        altitude    = np.linalg.norm(np.asarray(sat_pos)) - RADIUS_OF_EARTH

        altitudes.append(altitude)
    
    plt.hist(x=altitudes, bins=bins, orientation="horizontal", color="blue")
    plt.title("RSO Altitude Distribution \n from TLE @" + str(time))
    plt.ylim(0,2000)
    plt.xlabel("# RSOs")
    plt.ylabel("Altitude (km)")

    # plt.show()
    str_time = time.strftime("%Y-%m-%dT%H%M%S.%f")

    plt_file = "./COOP_data/output/RSO_altitude_distribution_" + str_time + file_type
    if os.path.isfile(plt_file):
        os.remove(plt_file)
    plt.tight_layout()
    plt.savefig(plt_file, dpi=dpi)
    plt.clf()




def draw_conjunctions_in_3D(total_results):
    sat1_x_list = []
    sat1_y_list = []
    sat1_z_list = []
    sat2_x_list = []
    sat2_y_list = []
    sat2_z_list = []

    for result in total_results:

        sat1_x, sat1_y, sat1_z = result["SGP4_SAT1_POS@PPDB_TCA"][0], result["SGP4_SAT1_POS@PPDB_TCA"][1], result["SGP4_SAT1_POS@PPDB_TCA"][2]
        sat2_x, sat2_y, sat2_z = result["SGP4_SAT2_POS@PPDB_TCA"][0], result["SGP4_SAT2_POS@PPDB_TCA"][1], result["SGP4_SAT2_POS@PPDB_TCA"][2]

        sat1_x_list.append(sat1_x)
        sat1_y_list.append(sat1_y)
        sat1_z_list.append(sat1_z)
        sat2_x_list.append(sat2_x)
        sat2_y_list.append(sat2_y)
        sat2_z_list.append(sat2_z)

    plt.rcParams["figure.figsize"] = (16, 17)
    fig = plt.figure()

    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect((1,1,1))
    ax.set_xlim(-6500, 6500)
    ax.set_ylim(-6500, 6500)
    # ax.set_yticks([])
    ax.set_zlim(-6500, 6500)
    ax.set_xlabel("X (km)", labelpad=30.0)
    ax.set_ylabel("Y (km)", labelpad=30.0)
    ax.set_zlabel("Z (km)", labelpad=30.0)
    # for spine in ['top', 'right', 'left', 'bottom']:
    #     ax.spines[spine].set_visible(True)

    ax.view_init(45, 45)

    import numpy as np
    r = 6378.14
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:50j, 0.0:2.0*pi:50j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='blue', alpha=0.5, linewidth=1)
    
    ax.scatter(sat1_x_list, sat1_y_list, sat1_z_list, 
                color="#33AA33", marker='o', s=0.10, depthshade=True, alpha=1)
    ax.scatter(sat2_x_list, sat2_y_list, sat2_z_list,
                color="#33AA33", marker='o', s=0.10, depthshade=True, alpha=1)

    plt.title("Conjunction Locations")
    # plt.show()
    plt_file = "./COOP_data/output/RSO_conjunction_in_3D" + file_type
    if os.path.isfile(plt_file):
        os.remove(plt_file)
    plt.tight_layout()
    plt.savefig(plt_file, dpi=dpi)
    plt.clf()


if __name__ == "__main__":
    import os
    import pickle

    import matplotlib.pyplot as plt 
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 16
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rc('font', size=SMALL_SIZE) # controls default text sizes 
    plt.rc('axes', titlesize=BIGGER_SIZE) # fontsize of the axes title 
    plt.rc('axes', labelsize=MEDIUM_SIZE) # fontsize of the x and y labels 
    plt.rc('xtick', labelsize=SMALL_SIZE) # fontsize of the tick labels 
    plt.rc('ytick', labelsize=SMALL_SIZE) # fontsize of the tick labels 
    plt.rc('legend', fontsize=SMALL_SIZE) # legend fontsize 
    plt.rc('figure', titlesize=BIGGER_SIZE) # fontsize of the figure title


    tle_filename    = "./COOP_data/LEO_full_16709_210321_0800UTC.tle"
    # cdm_filename    = "./COOP_data/CDM-20210403.json"
    ppdb_filename   = "./COOP_data/PPDB2.txt"
    result_filename = "./COOP_data/result.txt"

    RSO_filename = "./COOP_data/RSO.pickle"

    RADIUS_OF_EARTH             = 6378.1

    calculate_histogram         = True
    histogram_distance_interval = None #(300, 550)
    calculate_conjunction_plane = False
    show_full_quadrant          = False
    show_sat_alt_dist_histogram = True
    show_conjunctions_in_3D     = False

    threshold_distance, distance_interval, bin_interval = 10.0, 1.0, 25.0

    if os.path.exists(result_filename) and os.path.exists(result_filename[:-4]+".pickle") and os.path.exists(RSO_filename):
        print("Found " + result_filename)
        print("Found " + result_filename[:-4]+".pickle")
        print("Found " + RSO_filename)

        with open(result_filename[:-4]+".pickle", "rb") as f:
            total_results = pickle.load(f)

        with open(RSO_filename, "rb") as f:
            RSOs = pickle.load(f)
    else:
        print("Read TLE " + tle_filename)
        TLEs = read_tle(tle_filename)
        # print("Read CDM " + cdm_filename)
        # CDMLists = read_cdm(cdm_filename)
        print("Read PPDB " + ppdb_filename)
        PPDBs = read_ppdb(ppdb_filename)

        print("Generating RSOs")
        RSOs = generate_RSO(TLEs)

        print("Comparing SGP4 Propagation with PPDB")
        total_results, not_in_tle_IDs = compare_SGP4_N_PPDB(RSOs, PPDBs, RADIUS_OF_EARTH)

        print("Saving Result " + result_filename)
        save_result(result_filename, total_results, not_in_tle_IDs)
        print("Done")

    if calculate_histogram:
        print("Cutoff Distance : {0}\nDistance Interval : {1}\nBin Interval : {2}\n".format(threshold_distance, distance_interval, bin_interval))
        print("Calculating Conjunction Histogram")

        save_individual_histogram = True
        histogram = calculate_and_save_conjunction_histogram(total_results ,threshold_distance, distance_interval, bin_interval, save_individual_histogram, histogram_distance_interval)
        save_individual_histogram = False
        histogram = calculate_and_save_conjunction_histogram(total_results ,threshold_distance, distance_interval, bin_interval, save_individual_histogram, histogram_distance_interval)

        print("Done")

    if calculate_conjunction_plane:
        Rs, Thetas = draw_conjunction_polarplane_2D(total_results, RADIUS_OF_EARTH, show_full_quadrant)

    if show_sat_alt_dist_histogram:
        str_time = "2021-3-26T0:0:0.000"
        draw_RSO_histogram_at_time(RSOs, str_time, RADIUS_OF_EARTH, bin_interval)

    if show_conjunctions_in_3D:
        draw_conjunctions_in_3D(total_results)