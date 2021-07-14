import os
import json
import math
import tle2czml
from datetime import datetime, timezone, timedelta
import sys
if __name__ == "__main__":
    _, year, month, day, hour, minute = sys.argv
    # You can specify the time range you would like to visualise
    year = int(year)
    month = int(month)
    day = int(day)
    hour = int(hour)
    minute = int(minute)
    start_time  =    datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    end_time    =    start_time + timedelta(days=1)
    tle2czml.create_czml("/home/shchoi/new_coop/DB/latest_all_starlink.tle", start_time=start_time, end_time=end_time)

    json_path = "orbit.czml"
    with open(json_path, 'r') as f:
        json_data = f.read()
        json_data = json.loads(json_data)
        
    will_removed_data = []

    for i, json_datum in enumerate(json_data):
        if "billboard" in json_datum:
            json_datum.pop("billboard", None)


        if "label" in json_datum:
            json_datum.pop("label", None)


        if "path" in json_datum:
            json_datum.pop("path", None)
            json_datum["point"] = {
                    "color" : 
                    {
                        "rgba" : [255,255,255,255]
                    },
                    "outlineColor" : 
                    {
                        "rgba" : [0,255,255,255]
                    },
                    "outlineWidth" : 1,
                    "pixelSize" : 3,
                }


        if "position" in json_datum:
            for data in json_datum["position"]["cartesian"]:
                if math.isnan(data):
                    # print(json_data[json_datum])
                    if json_datum in json_data:
                        print(json_datum)
                        will_removed_data.append(json_datum)
                        # json_data.remove(json_datum)
                        break

    for removed_datum in will_removed_data:
        if removed_datum in json_data:
            print(removed_datum)
            json_data.remove(removed_datum)


    pretty_json = open("/home/shchoi/new_coop/DB/pretty_orbit.czml", "w")
    pretty_json.write(json.dumps(json_data, indent=2))
    pretty_json.close()