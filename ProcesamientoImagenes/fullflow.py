import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

from digit_ocr import prepareImage
from tflitetest import predictDigit
from agujas import determineNeedle
import agujas

from PIL import Image
import json
import shutil
from datetime import datetime

def getMeasurement(sourceImage):
    #img = cv2ToPILImage(img)
    finalLitros = 0.0
    #out = {}

    with open("config.json") as cfgfile:
        opts = json.load(cfgfile)
        agujas.debugMode = opts["debug"]
    
    with open("lastdigits.json") as dfile:
        digitMap = json.load(dfile)
        prevOut = 0.0

        for i, (dname, val) in enumerate(digitMap.items()):
            amount = opts["digits"][i]["amount"]
            prevOut += amount * val

    for digitopt in opts["digits"]:
        points = digitopt["points"]
        name = digitopt["name"]

        img = prepareImage(sourceImage, points)
        
        prediction, outvec = predictDigit(img)

        #print(f"{name} => predicted: {prediction}")
        if opts["debug"]: img.save(f"cutouts/{name}.jpg")
        #print([f"{i:.2f}" for i in outvec])

        if prediction is not None:
            digitMap[name] = prediction
    
        finalLitros += digitMap[name] * digitopt["amount"]
    
    for dial in opts["dials"]:
        measurement = determineNeedle(sourceImage, dial)
        #print(f"{dial["name"]} => {measurement:.2f}")
        finalLitros += measurement * dial["amount"]
        prevOut += measurement * dial["amount"]

    if finalLitros >= prevOut and finalLitros <= prevOut + 10:
        with open("lastdigits.json", "w") as dfile:
            json.dump(digitMap, dfile)
    else:
        finalLitros = prevOut

    out = {
        "fecha_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "display": 1000 * finalLitros,
    }
    return out
    

debugDirs = [
    "cutouts",
    "debug_output"
]

import requests
from io import BytesIO
import time
import envio

if __name__ == "__main__":
    try:
        url = "http://192.168.1.177/capture"

        for dir in debugDirs:
            if os.path.exists(dir):
                shutil.rmtree(dir)
            os.mkdir(dir)

        i = 0
        forEveryN = 10
        lecturas = []

        while True:
            print("Request")
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                continue

            img = Image.open(BytesIO(response.content))

            start = time.time()
            #print(f"{1000 * getMeasurement(img):.2f}")
            data = getMeasurement(img)
            print(f"Display reading: {data["display"]:.2f}")
            print(f"Took {(time.time() - start):.2f} seconds.")
            i += 1
            lecturas.append(data)

            if i == forEveryN:
                envio.publish_json(lecturas)
                i = 0
                lecturas = []
            envio.client.loop()
            

    finally:
        envio.client.disconnect()