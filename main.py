#READFILE
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
                    satellite_name += name

                satellite['name'] = satellite_name

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

    return json_cdm
    

def read_ppdb(filename):
    import datetime as dt
    ppdb = []

    with open(filename, "r") as ppdb_data:
        lines = ppdb_data.readlines()
        for line in lines:
            if line.startswith("%"):
                continue
            
            cdm = dict()
            data = line.split()

            cdm["SAT1ID"] = data[0]
            cdm["SAT2ID"] = data[1]
            cdm["MinDistance"] = data[2]

            del_castart_sec = float(data[4]) - float(data[3])

            std_date = " ".join(data[6:])
            std_time = dt.datetime.strptime(std_date, "%Y %m %d %H %M %S.%f")
            
            tca = std_time + dt.timedelta(seconds=float(data[3]))
            castart = tca + dt.timedelta(seconds=del_castart_sec)
            caend = castart + dt.timedelta(seconds=float(data[5]))

            cdm["TCA"] = tca.strftime("%Y-%m-%dT%H:%M:%S.%f")
            cdm["CAStart"] = castart.strftime("%Y-%m-%dT%H:%M:%S.%f")
            cdm["CAEnd"] = caend.strftime("%Y-%m-%dT%H:%M:%S.%f")

            ppdb.append(cdm)
    return ppdb


def generate_satellite(tle):
    from sgp4.earth_gravity import wgs72
    from sgp4.io import twoline2rv

    satellites = dict()

    for satellite in tle:
        sat_id = satellite['SAT_ID']
        satellites[sat_id] = twoline2rv(satellite["s"], satellite["t"], wgs72)

    return satellites




def compare_CDM_N_PPDB(satellites, CDMs, PPDBs):
    import datetime as dt

    keys = ["SAT1_ID", "SAT2_ID", "CDM_ID", "CDM_TCA", "CDM_DCA", "SGP4_DCA@CDM_TCA", "SGP4_SAT1_POS@CDM_TCA", "SGP4_SAT2_POS@CDM_TCA", "PPDB_TCA", "PPDB_DCA", "SGP4_DCA@PPDB_TCA", "SGP4_SAT1_POS@PPDB_TCA", "SGP4_SAT2_POS@PPDB_TCA", "PPDB_TCA - CDM_TCA", "PPDB_DCA - CDM_DCA"]
    header = "\t".join(keys) + "\n"

    for PPDB in PPDBs:
        pass

    for CDM in CDMs:
        pass

    pass

def save_result_as_json_n_txt(json_cdm, filename, date):
    result_json = []
    result_txt = "CDM_ID\t\tSAT1ID\tSAT2ID\tTCA\t\t\t\t\t\t\tDCA\t\t\t\t\tMIN_RNG\tDIFF_DCA\n"

    for CDM in json_cdm:
        tca_date = CDM["TCA"].split("T")[0]
        if tca_date == date:
            temp_cdm = dict()

            temp_cdm["CDM_ID"] = CDM["CDM_ID"]
            temp_cdm["SAT_1_ID"] = CDM["SAT_1_ID"].zfill(5)
            temp_cdm["SAT_2_ID"] = CDM["SAT_2_ID"].zfill(5)
            temp_cdm["TCA"] = CDM["TCA"]
            temp_cdm["DCA"] = find_DCA_regard_to_tle(CDM, json_tle)

            if temp_cdm["DCA"] == 0:
                continue

            temp_cdm["MIN_RNG"] = float(CDM["MIN_RNG"])/1000
            temp_cdm["DIFF_DCA"] = abs(temp_cdm["MIN_RNG"] -  temp_cdm["DCA"])

            temp_cdm_txt = temp_cdm["CDM_ID"] + "\t" + temp_cdm["SAT_1_ID"] + "\t" + temp_cdm["SAT_2_ID"] + "\t" + temp_cdm["TCA"] + "\t" + str(temp_cdm["DCA"]) + "\t" + str(temp_cdm["MIN_RNG"]) + "\t" + str(temp_cdm["DIFF_DCA"]) + "\n"

            result_json.append(temp_cdm)
            result_txt += temp_cdm_txt
    return result_json


def find_DCA_regard_to_tle(one_CDM, TLE):
    import datetime as dt
    import numpy as np

    from sgp4.api import Satrec
    from sgp4.earth_gravity import wgs72
    from sgp4.io import twoline2rv

    one_CDM["SAT_1_ID"] = one_CDM["SAT_1_ID"].zfill(5)
    one_CDM["SAT_2_ID"] = one_CDM["SAT_2_ID"].zfill(5)

    satellite1 = None
    satellite2 = None

    for satellite in TLE:
        if satellite1 is not None and satellite2 is not None:
            break

        if satellite["SAT_ID"] == one_CDM["SAT_1_ID"]:
            satellite1 = twoline2rv(satellite["s"], satellite["t"], wgs72)
        elif satellite["SAT_ID"] == one_CDM["SAT_2_ID"]:
            satellite2 = twoline2rv(satellite["s"], satellite["t"], wgs72)

    if satellite1 is None:
        print("sat id " + one_CDM["SAT_1_ID"] + " not exist in TLE")
        return 0
    elif satellite2 is None:
        print("sat id " + one_CDM["SAT_2_ID"] + " not exist in TLE")
        return 0

    timeStr = one_CDM["TCA"].replace("T"," ")
    referenceTime = dt.datetime.strptime(timeStr, '%Y-%m-%d %H:%M:%S.%f')
    start_time = referenceTime
    referenceTimeTuple = referenceTime.timetuple()

    sat1_pos, sat1_vel = satellite1.propagate(referenceTimeTuple.tm_year, referenceTimeTuple.tm_mon, referenceTimeTuple.tm_mday, referenceTimeTuple.tm_hour, referenceTimeTuple.tm_min, referenceTimeTuple.tm_sec)
    sat2_pos, sat2_vel = satellite2.propagate(referenceTimeTuple.tm_year, referenceTimeTuple.tm_mon, referenceTimeTuple.tm_mday, referenceTimeTuple.tm_hour, referenceTimeTuple.tm_min, referenceTimeTuple.tm_sec)

    sat1_pos = np.asarray(sat1_pos)
    sat2_pos = np.asarray(sat2_pos)

    DCA = np.linalg.norm((sat2_pos - sat1_pos))

    return DCA


if __name__ == "__main__":
    tle_filename = "LEO_full_16709_210321_0800UTC.tle"
    cdm_filename = "CDM-20210403.json"
    ppdb_filename = "PPDB_0324.txt"

    json_tle = read_tle(tle_filename)
    json_cdm = read_cdm(cdm_filename)
    ppdb_json = read_ppdb(ppdb_filename)

    satellites = generate_satellite(json_tle)
    
    result_json = save_result_as_json_n_txt(json_cdm, 'result', date)

    import datetime as dt

    for data in result_json:
        for ppdb in ppdb_json:
            if int(ppdb["SAT1ID"]) == int(data["SAT_1_ID"]) and int(ppdb["SAT2ID"]) == int(data["SAT_2_ID"]):
                pass
            elif int(ppdb["SAT2ID"]) == int(data["SAT_1_ID"]) and int(ppdb["SAT1ID"]) == int(data["SAT_2_ID"]):
                pass
            else:
                continue

            tca = dt.datetime.strptime(data["TCA"], "%Y-%m-%dT%H:%M:%S.%f")
            ppdb_tca = dt.datetime.strptime(ppdb["TCA"], "%Y-%m-%dT%H:%M:%S.%f")

            if tca.hour == ppdb_tca.hour and tca.minute == ppdb_tca.minute:
                pass
            else:
                continue

            castart = dt.datetime.strptime(ppdb["CAStart"], "%Y-%m-%dT%H:%M:%S.%f")
            caend = dt.datetime.strptime(ppdb["CAEnd"], "%Y-%m-%dT%H:%M:%S.%f")

            data["PPDB_TCA"] = ppdb["TCA"]
            data["PPDB_MinDistance"] = ppdb["MinDistance"]
            data["DIFF_TCA"] = str((tca-ppdb_tca).total_seconds())

            if castart <= tca and tca <= caend:
                data["EXIST"] = True
                break

    result_txt = "CDM_ID\t\tSAT1ID\tSAT2ID\tTCA\t\t\t\t\t\t\tDCA\t\t\t\t\tPPDB_TCA\t\t\t\t\tPPDB_DCA\tMIN_RNG\tDIFF_TCA\tDIFF_DCA |MIN_RNG - DCA|\n"

    for data in result_json:
        result_txt += data["CDM_ID"] + "\t"
        result_txt += data["SAT_1_ID"] + "\t" + data["SAT_2_ID"] + "\t"
        result_txt += data["TCA"] + "\t" + str(data["DCA"]) + "\t"
        result_txt += data["PPDB_TCA"] + "\t" + data["PPDB_MinDistance"] + "\t"
        result_txt += str(data["MIN_RNG"]) + "\t"
        result_txt += data["DIFF_TCA"] + "\t" + str(data["DIFF_DCA"])
        result_txt += "\n"        

    with open("PPDB_check.txt","w") as f:
        f.write(str(result_txt))

    pass