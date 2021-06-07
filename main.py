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
        satellite = None

        for tle_line in tle_lines:
            if tle_line.startswith('0'):
                sat_name = tle_line.split()
                del(sat_name[0])

                if satellite is not None:
                    json_tle.append(satellite)

                satellite = dict()
                satellite_name = str()

                for name in sat_name:
                    satellite_name += name + " "

                satellite['SATNAME'] = satellite_name[:-1]

            elif tle_line.startswith('1'):
                satellite['s'] = tle_line

            elif tle_line.startswith('2'):
                satellite['t'] = tle_line

                satellite['SAT_ID'] = tle_line.split()[1].zfill(5)

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


def generate_satellite(tle):
    from sgp4.earth_gravity import wgs72
    from sgp4.io import twoline2rv
    import pickle

    satellites = dict()

    for satellite in tle:
        sat_id = satellite['SAT_ID']
        satellites[sat_id] = (satellite['SATNAME'], twoline2rv(satellite["s"], satellite["t"], wgs72))

    with open("./CoopAssist/satellite.pickle", "wb") as f:
        pickle.dump(satellites, f)

    return satellites


def compare_SGP4_N_PPDB(satellites, PPDBs, Radius_of_Earth):
    import datetime as dt
    import numpy as np

    # keys = ["SAT1_ID", "SAT2_ID", "CDM_ID", "CDM_TCA", "CDM_DCA", "SGP4_DCA@CDM_TCA", "SGP4_SAT1_POS@CDM_TCA", "SGP4_SAT2_POS@CDM_TCA", "PPDB_TCA", "PPDB_DCA", "SGP4_DCA@PPDB_TCA", "SGP4_SAT1_POS@PPDB_TCA", "SGP4_SAT2_POS@PPDB_TCA", "PPDB_TCA - CDM_TCA", "PPDB_DCA - CDM_DCA"]
    # header = "\t".join(keys) + "\n"

    # filtered_CDM = _filter_cdm_to_latest(CDMLists)

    total_results = list()
    not_in_tle_IDs = set()

    for PPDB in PPDBs:
        if PPDB["SAT1ID"].zfill(5) not in satellites.keys():
            not_in_tle_IDs.add(PPDB["SAT1ID"].zfill(5))
            print(PPDB["SAT1ID"].zfill(5) + "is not is TLE")
            continue
        elif PPDB["SAT2ID"].zfill(5) not in satellites.keys():
            not_in_tle_IDs.add(PPDB["SAT2ID"].zfill(5))
            print(PPDB["SAT2ID"].zfill(5) + "is not is TLE")
            continue

        sat1 = satellites[PPDB["SAT1ID"].zfill(5)][1]
        sat2 = satellites[PPDB["SAT2ID"].zfill(5)][1]

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
        result["SGP4_SAT1_ALT@PPDB_TCA"]= np.linalg.norm(np.asarray(PPDB_sat1_pos)) - Radius_of_Earth
        result["SGP4_SAT2_POS@PPDB_TCA"]= PPDB_sat2_pos
        result["SGP4_SAT2_ALT@PPDB_TCA"]= np.linalg.norm(np.asarray(PPDB_sat2_pos)) - Radius_of_Earth

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


def calculate_and_save_conjunction_histogram(total_results, cutoff_distance, distance_interval, bin_interval, save_individual):
    import numpy as np
    import matplotlib.pyplot as plt

    distances = np.arange(distance_interval, cutoff_distance + distance_interval, distance_interval)
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

    plt.rcParams["figure.figsize"] = (5, 10)
    plt.tight_layout()
    plt.figure()

    for i, distance in enumerate(reversed(distances)):
        if save_individual:
            plt.figure()
            plt.hist(x=histogram[distance], bins=bins, orientation="horizontal", color=colors[i], label=str(distance)+" km")
            plt.legend()
            plt.savefig("./CoopAssist/"+str(distance)+"_histogram.png", dpi=200)
        else:
            plt.hist(x=histogram[distance], bins=bins, orientation="horizontal", color=colors[i], label=str(distance)+" km")

    # plt.hlines(860, 0, 8000, colors="black", linestyles="dashed")
    plt.legend(title="Threshold")
    plt.xlabel("#Conjunctions")
    plt.ylabel("Heights (km)")

    if not save_individual:
        plt.savefig("./CoopAssist/total_histogram.png", dpi=200)
    
    return histogram

def draw_conjunction_polarplane_2D(total_results, Radius_of_Earth, show_full_quadrant):
    import numpy as np
    import math
    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = (15, 15)
    
    Rs = list()
    Thetas = list()

    for result in total_results:
        sat1_pos    = np.asarray(result["SGP4_SAT1_POS@PPDB_TCA"])
        x, y, z     = sat1_pos
        radius      = np.linalg.norm(sat1_pos) - Radius_of_Earth
        theta       = math.atan(z/np.linalg.norm(sat1_pos[:2]))

        if show_full_quadrant:
            theta = _reflect_for_full_quadrant(x, y, z, theta)
        # if show_full_quadrant:
        #     if y > 0 and z > 0:
        #         pass
        #     elif y < 0 and z > 0:
        #         theta = math.radians(180) - theta
        #     elif y < 0 and z < 0:
        #         theta = math.radians(180) - theta
        #     elif y > 0 and z < 0:
        #         pass
        #     else:
        #         raise ValueError("?")

        Rs.append(radius)
        Thetas.append(theta)
    
    # colors = []

    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    ax.set_ylim(0,2000)
    ax.set_yticks(np.arange(0,2100,100))
    ax.scatter(Thetas, Rs, c=Rs, s=1, cmap='tab20b', alpha=1)

    if not show_full_quadrant:
        ax.set_thetamin(0)
        ax.set_thetamax(90)

    plt.savefig("./CoopAssist/polar_conjunction.png", dpi=200)

    return Rs, Thetas


def _reflect_for_full_quadrant(x, y, z, theta):
    import math

    if y < 0:
        return (math.radians(180) - theta)
    else:
        return theta


def draw_satellite_histogram_at_time(satellites, str_time, Radius_of_Earth, bin_interval):
    import datetime as dt
    import numpy as np
    import math
    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = (10, 10)
    plt.tight_layout()
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

    # distances = np.arange(distance_interval, cutoff_distance + distance_interval, distance_interval)
    # histogram = {distance : list() for distance in distances}
    bins = [i*bin_interval for i in range(int(2000/bin_interval))]

    time = dt.datetime.strptime(str_time, "%Y-%m-%dT%H:%M:%S.%f")

    altitudes = list()
    for SATID in satellites.keys():
        sat_pos, _  = satellites[SATID][1].propagate(time.year, time.month, time.day, time.hour, time.minute, time.second + time.microsecond/1000000)
        altitude    = np.linalg.norm(np.asarray(sat_pos)) - Radius_of_Earth

        altitudes.append(altitude)

    plt.hist(x=altitudes, bins=bins, orientation="horizontal", color=colors[0], ec="black")
    plt.title("Satellite Altitude Distribution from TLE @" + str(time))
    plt.xlabel("#Satellites")
    plt.ylabel("Altitude (km)")

    # plt.show()
    str_time = time.strftime("%Y-%m-%dT%H%M%S.%f")
    plt.savefig("./CoopAssist/satellite_altitude_distribution_" + str_time + ".png", dpi=200)


if __name__ == "__main__":
    import os
    import pickle

    tle_filename    = "./CoopAssist/LEO_full_16709_210321_0800UTC.tle"
    # cdm_filename    = "./CoopAssist/CDM-20210403.json"
    ppdb_filename   = "./CoopAssist/PPDB2.txt"
    result_filename = "./CoopAssist/result.txt"

    satellite_filename = "./CoopAssist/satellite.pickle"

    Radius_of_Earth             = 6378.1
    calculate_histogram         = True
    calculate_conjunction_plane = True
    show_full_quadrant          = True
    show_sat_alt_dist_histogram = True
    cutoff_distance, distance_interval, bin_interval = 10.0, 1.0, 10.0

    if os.path.exists(result_filename) and os.path.exists(result_filename[:-4]+".pickle") and os.path.exists(satellite_filename):
        print("Found " + result_filename)
        print("Found " + result_filename[:-4]+".pickle")
        print("Found " + satellite_filename)

        with open(result_filename[:-4]+".pickle", "rb") as f:
            total_results = pickle.load(f)

        with open(satellite_filename, "rb") as f:
            satellites = pickle.load(f)
    else:
        print("Read TLE " + tle_filename)
        TLEs = read_tle(tle_filename)
        # print("Read CDM " + cdm_filename)
        # CDMLists = read_cdm(cdm_filename)
        print("Read PPDB " + ppdb_filename)
        PPDBs = read_ppdb(ppdb_filename)

        print("Generating Satellites")
        satellites = generate_satellite(TLEs)

        print("Comparing SGP4 Propagation with PPDB")
        total_results, not_in_tle_IDs = compare_SGP4_N_PPDB(satellites, PPDBs, Radius_of_Earth)

        print("Saving Result " + result_filename)
        save_result(result_filename, total_results, not_in_tle_IDs)
        print("Done")

    if calculate_histogram:
        print("Cutoff Distance : {0}\nDistance Interval : {1}\nBin Interval : {2}\n".format(cutoff_distance, distance_interval, bin_interval))
        print("Calculating Conjunction Histogram")

        save_individual_histogram = False
        histogram = calculate_and_save_conjunction_histogram(total_results ,cutoff_distance, distance_interval, bin_interval, save_individual_histogram)
        print("Done")

    if calculate_conjunction_plane:
        Rs, Thetas = draw_conjunction_polarplane_2D(total_results, Radius_of_Earth, show_full_quadrant)

    if show_sat_alt_dist_histogram:
        str_time = "2021-3-26T0:0:0.000"
        draw_satellite_histogram_at_time(satellites, str_time, Radius_of_Earth, bin_interval)