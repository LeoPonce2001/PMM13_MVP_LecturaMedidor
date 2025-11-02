import requests as rq
import random as rnd
from datetime import datetime, timedelta

url = "http://3.90.200.123:5009/api/data/insertData"


start = datetime.now() - timedelta(days=1.5)
times = [start + timedelta(minutes=i) for i in range(24 * 60)]
utcstrings = [t.isoformat() for t in times]


startms = 3486777.0

rqdata = {
    "homeId": 2,
    "data": []
}

burst_remaining = 0
for utime in utcstrings:
    if burst_remaining > 0:
        # still in an active burst
        delta = rnd.uniform(1, 4)
        burst_remaining -= 1
    else:
        # chance to start a new burst
        if rnd.random() < 0.03:  # 3% chance any minute starts a burst
            burst_length = int(rnd.uniform(2, 15))  # burst lasts 2–15 minutes
            burst_remaining = burst_length - 1
            delta = rnd.uniform(1, 4)
        else:
            delta = 0

    ndata = startms + delta
    rqdata["data"].append({
        "time": utime,
        "data": ndata
    })
    startms = ndata

resp = rq.post(url, json=rqdata)
print(resp)