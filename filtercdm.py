class CDM_parser():
    def __init__(self, cdm_filename, ppdb_filenames, tle_filename, dates):
        self._cdm_filename = cdm_filename
        self._dates = dates
        self._cdm_dict = self.read_cdm(self._cdm_filename)
        self._cdm_at_dates = self.filter_cdm_with_dates(self._cdm_dict, self._dates)
        self._latest_nonoverlap_cdm = CDM()
        self.update_cdm_to_latest()

        self._tle_dict = self.read_tle(tle_filename)

        self._ppdb = self.read_ppdb(ppdb_filenames)


    def read_ppdb(self, filenames):
        import datetime as dt

        print("Reading PPDB")

        ppdb = []
        for i, filename in enumerate(filenames):
            with open(filename, "r") as ppdb_data:
                for date in self._dates:
                    date = date + "T00:00:00.000"

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

                        tca = dt.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f") + dt.timedelta(seconds=float(data[3]))
                        castart = tca + dt.timedelta(seconds=del_castart_sec)
                        caend = castart + dt.timedelta(seconds=float(data[5]))

                        cdm["TCA"] = tca.strftime("%Y-%m-%dT%H:%M:%S.%f")
                        cdm["CAStart"] = castart.strftime("%Y-%m-%dT%H:%M:%S.%f")
                        cdm["CAEnd"] = caend.strftime("%Y-%m-%dT%H:%M:%S.%f")

                        ppdb.append(cdm)
                print(filename + " Done")
        print("Done")
        return ppdb


    def read_cdm(self, filename):
        import json

        print("Reading CDM public")
        with open(filename, "r") as cdm:
            json_cdm = json.load(cdm)

        print("Done")
        return json_cdm


    def read_tle(self, filename):
        import json

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

            json_tle.append(satellite)
        return json_tle


    def filter_cdm_with_dates(self, cdm_dict, dates):
        cdm_at_dates = list()

        for cdm in cdm_dict:
            date = cdm["TCA"].split("T")[0]

            if date in dates:
                cdm_at_dates.append(cdm)

        return cdm_at_dates

    
    def update_cdm_to_latest(self):
        import datetime as dt

        for new_cdm in self._cdm_at_dates:
            index, old_cdm = self._latest_nonoverlap_cdm.contains(new_cdm)

            if old_cdm is None:
                new_cdm["Count"] = 1
                self._latest_nonoverlap_cdm.add(new_cdm)

            else:
                new_date = dt.datetime.strptime(new_cdm["CREATED"], "%Y-%m-%d %H:%M:%S.%f")
                old_date = dt.datetime.strptime(old_cdm["CREATED"], "%Y-%m-%d %H:%M:%S.%f")

                self._latest_nonoverlap_cdm._cdm[index]["Count"] += 1

                if new_date > old_date:
                    new_cdm["Count"] = self._latest_nonoverlap_cdm._cdm[index]["Count"]
                    self._latest_nonoverlap_cdm._cdm[index] = new_cdm


    def compare_ppdb_and_cdm(self):
        import datetime as dt

        for cdm in self._cdm_at_dates:
            for ppdb in self._ppdb:
                if int(ppdb["SAT1ID"]) == int(cdm["SAT_1_ID"]) and int(ppdb["SAT2ID"]) == int(cdm["SAT_2_ID"]):
                    pass
                elif int(ppdb["SAT2ID"]) == int(cdm["SAT_1_ID"]) and int(ppdb["SAT1ID"]) == int(cdm["SAT_2_ID"]):
                    pass
                else:
                    cdm["EXIST"] = "NONE"
                    # ppdb["EXIST"] = "NONE"
                    continue

                tca = dt.datetime.strptime(cdm["TCA"], "%Y-%m-%dT%H:%M:%S.%f")
                ppdb_tca = dt.datetime.strptime(ppdb["TCA"], "%Y-%m-%dT%H:%M:%S.%f")

                if tca.hour == ppdb_tca.hour and tca.minute == ppdb_tca.minute:
                    pass
                else:
                    cdm["EXIST"] = "NONE"
                    # ppdb["EXIST"] = "NONE"
                    continue

                castart = dt.datetime.strptime(ppdb["CAStart"], "%Y-%m-%dT%H:%M:%S.%f")
                caend = dt.datetime.strptime(ppdb["CAEnd"], "%Y-%m-%dT%H:%M:%S.%f")

                if castart <= tca and tca <= caend:
                    cdm["EXIST"] = "TRUE"
                    ppdb["EXIST"] = "TRUE"
                    break
                else : 
                    cdm["EXIST"] = "FALSE"
                    ppdb["EXIST"] = "FALSE"


    def save(self, filename):
        with open("01.total_" + filename, "w") as f:
            f.write("CDM_ID\tCREATED\tTCA\tSAT_1_ID\tSAT_1_NAME\tSAT1_OBJECT_TYPE\tSAT_2_ID\tSAT_2_NAME\tSAT2_OBJECT_TYPE\tEXIST\n")
            for cdm in self._cdm_at_dates:
                for key in cdm.keys():
                    if key in ["EMERGENCY_REPORTABLE", "MIN_RNG", "PC", "SAT1_RCS", "SAT_1_EXCL_VOL", "SAT2_RCS", "SAT_2_EXCL_VOL", "Count"]:
                        continue

                    f.write(str(cdm[key]))
                    f.write("\t")
                f.write("\n")

        with open("02.cdm_" + filename, "w") as f:
            f.write("CDM_ID\tCREATED\tTCA\tMIN_RNG\tSAT_1_ID\tSAT_1_NAME\tSAT1_OBJECT_TYPE\tSAT_2_ID\tSAT_2_NAME\tSAT2_OBJECT_TYPE\tEXIST\tDCA\n")
            for cdm in self._latest_nonoverlap_cdm._cdm:
                for key in cdm.keys():
                    if key in ["EMERGENCY_REPORTABLE", "PC", "SAT1_RCS", "SAT_1_EXCL_VOL", "SAT2_RCS", "SAT_2_EXCL_VOL", "Count"]:
                        continue
                    if key == "MIN_RNG":
                        f.write(str(float(cdm[key])/1000))
                        f.write("\t")
                        continue

                    f.write(str(cdm[key]))
                    f.write("\t")

                    
                f.write("\n")

        with open("03.ppdb_" + filename, "w") as f:
            f.write("SAT1ID\tSAT2ID\tMinDistance\tTCA\tCAStart\tCAEnd\tEXIST\tDCA\n")
            for ppdb in self._ppdb:
                for key in ppdb.keys():
                    f.write(str(ppdb[key]))
                    f.write("\t")
                f.write("\n")


    def propagate_with_sgp4(self):
        import datetime as dt
        import numpy as np

        from sgp4.api import Satrec
        from sgp4.earth_gravity import wgs72
        from sgp4.io import twoline2rv

        for cdm in self._latest_nonoverlap_cdm._cdm:
            s1 = None
            s2 = None

            # if cdm["EXIST"] == "NONE":
            for tle in self._tle_dict:
                if int(tle["SAT_ID"]) == int(cdm["SAT_1_ID"]):
                    s1 = twoline2rv(tle["s"], tle["t"], wgs72)
                elif int(tle["SAT_ID"]) == int(cdm["SAT_2_ID"]):
                    s2 = twoline2rv(tle["s"], tle["t"], wgs72)

            if s1 != None and s2 != None:
                referenceTime = dt.datetime.strptime(cdm["TCA"], '%Y-%m-%dT%H:%M:%S.%f')
                start_time = referenceTime
                referenceTimeTuple = referenceTime.timetuple()

                sat1_pos, sat1_vel = s1.propagate(referenceTimeTuple.tm_year, referenceTimeTuple.tm_mon, referenceTimeTuple.tm_mday, referenceTimeTuple.tm_hour, referenceTimeTuple.tm_min, referenceTimeTuple.tm_sec)
                sat2_pos, sat2_vel = s2.propagate(referenceTimeTuple.tm_year, referenceTimeTuple.tm_mon, referenceTimeTuple.tm_mday, referenceTimeTuple.tm_hour, referenceTimeTuple.tm_min, referenceTimeTuple.tm_sec)

                sat1_pos = np.asarray(sat1_pos)
                sat2_pos = np.asarray(sat2_pos)

                DCA = np.linalg.norm((sat2_pos - sat1_pos))

                cdm["DCA"] = str(DCA)

        for ppdb in self._ppdb:
            s1 = None
            s2 = None

            # if cdm["EXIST"] == "NONE":
            for tle in self._tle_dict:
                if int(tle["SAT_ID"]) == int(ppdb["SAT1ID"]):
                    s1 = twoline2rv(tle["s"], tle["t"], wgs72)
                elif int(tle["SAT_ID"]) == int(ppdb["SAT2ID"]):
                    s2 = twoline2rv(tle["s"], tle["t"], wgs72)

            if s1 != None and s2 != None:
                referenceTime = dt.datetime.strptime(ppdb["TCA"], '%Y-%m-%dT%H:%M:%S.%f')
                start_time = referenceTime
                referenceTimeTuple = referenceTime.timetuple()

                sat1_pos, sat1_vel = s1.propagate(referenceTimeTuple.tm_year, referenceTimeTuple.tm_mon, referenceTimeTuple.tm_mday, referenceTimeTuple.tm_hour, referenceTimeTuple.tm_min, referenceTimeTuple.tm_sec)
                sat2_pos, sat2_vel = s2.propagate(referenceTimeTuple.tm_year, referenceTimeTuple.tm_mon, referenceTimeTuple.tm_mday, referenceTimeTuple.tm_hour, referenceTimeTuple.tm_min, referenceTimeTuple.tm_sec)

                sat1_pos = np.asarray(sat1_pos)
                sat2_pos = np.asarray(sat2_pos)

                DCA = np.linalg.norm((sat2_pos - sat1_pos))

                ppdb["DCA"] = str(DCA)



class CDM():
    def __init__(self):
        self._cdm = list()

    
    def add(self, cdm):
        self._cdm.append(cdm)

    
    def contains(self, item):
        if len(self._cdm) == 0:
            return None, None

        for i, cdm in enumerate(self._cdm):
            if (item["SAT_1_ID"], item["SAT_2_ID"]) == (cdm["SAT_1_ID"], cdm["SAT_2_ID"]):
                return i, cdm
            elif (item["SAT_1_ID"], item["SAT_2_ID"]) == (cdm["SAT_2_ID"], cdm["SAT_1_ID"]):
                return i, cdm

        return None, None


if __name__ == "__main__":
    import datetime as dt

    cdm_filename = "TCA.2021.04.03.txt"
    # ppdb_filenames = ["PPDB_0322.txt", "PPDB_0323.txt", "PPDB_0324.txt"]
    ppdb_filenames = ["PPDB2_new.txt"]
    tle_filename = "TLE_in_CDM_new.tle"
    dates = ["2021-03-22", "2021-03-23", "2021-03-24"]

    save_filename = "parsing_result.txt"

    cdm_parser = CDM_parser(cdm_filename, ppdb_filenames, tle_filename, dates)

    cdm_parser.compare_ppdb_and_cdm()
    cdm_parser.propagate_with_sgp4()

    cdm_parser.save(save_filename)

    pass
    







