from PIL import Image, ImageDraw
import numpy as np
import math
import json
from scipy import ndimage
import os, shutil
import cv2

debugMode = False

def gaugeCrop(img, dialopt):
    cx, cy = dialopt["center"]
    r = dialopt["radius"]

    # Determine the crop box in the original image coordinates
    left = cx - r
    top = cy - r
    right = cx + r
    bottom = cy + r

    # Compute offsets if the circle extends outside the image
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - img.width)
    pad_bottom = max(0, bottom - img.height)

    # Create a new black canvas big enough for the circle
    new_width = img.width + pad_left + pad_right
    new_height = img.height + pad_top + pad_bottom
    canvas = Image.new("RGB", (new_width, new_height), (0, 0, 0))

    # Paste original image onto canvas
    canvas.paste(img, (pad_left, pad_top))

    # Adjust center coordinates due to padding
    cx += pad_left
    cy += pad_top

    # Create mask for the circle
    mask = Image.new("L", canvas.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

    # Apply mask: keep circle, black elsewhere
    result = Image.composite(canvas, Image.new("RGB", canvas.size, (0, 0, 0)), mask)

    # Crop to bounding box of mask
    bbox = mask.getbbox()
    result = result.crop(bbox)

    return result


def determineNeedle(img: Image.Image, dialopt):
    name = dialopt["name"]
    img = gaugeCrop(img, dialopt).convert("RGB")

    if debugMode:
        img.save(f"debug_output/{name}_cropped.png")
    
    zeroAngle = dialopt["zeroAngle"]
    targetColor = np.array(dialopt["needleColor"]).reshape(1, 1, 3)
    delta = dialopt["colorTolerance"]

    imgArray = np.array(img)
    
    dist = np.sqrt(np.sum((imgArray - targetColor) ** 2, axis=2))
    mask = dist <= delta

    labeledMask, numLabels = ndimage.label(mask)
    if numLabels == 0:
        print("No blobs.")
        return 0

    sizes = ndimage.sum(mask, labeledMask, range(1, numLabels + 1))

    largest = (np.argmax(sizes) + 1)
    largestBlobMask = (labeledMask == largest)
    mask = largestBlobMask

    coords = np.argwhere(mask)

    if len(coords) == 0:
        print("Not found")
        return 0
    
    ymean, xmean = coords.mean(axis=0)

    imgCenterX = img.width / 2
    imgCenterY = img.height / 2

    dx = xmean - imgCenterX
    dy = ymean - imgCenterY

    angle = math.degrees(math.atan2(dx, -dy)) % 360
    
    if dialopt["reversed"]:
        if angle < 180:
            angle += 180
        else:
            angle -= 180

    #print(f"Angle: {angle}")

    relativeAngle = (angle - zeroAngle) % 360
    
    #print(f"Relative: {relativeAngle}")

    if dialopt["clockwise"]:
        readout = (10 / 360) * relativeAngle
    else:
        readout = (10 / 360) * ((360 - relativeAngle) % 360)
    
    #print(f"Readout: {readout}")

    if debugMode:
        maskImg = Image.fromarray((mask * 255).astype(np.uint8))
        maskImg.save(f"debug_output/{name}_colormask.png")

        zeroAngleRad = math.radians(zeroAngle)

        lineLen = min(img.size) / 2

        endX = imgCenterX + lineLen * math.sin(zeroAngleRad)
        endY = imgCenterY - lineLen * math.cos(zeroAngleRad)

        imgVis = img.copy()
        draw = ImageDraw.Draw(imgVis)

        draw.line([(imgCenterX, imgCenterY), (endX, endY)], fill=(255, 0, 0), width=1)

        needleRad = math.radians(angle)
        nEndX = imgCenterX + lineLen * math.sin(needleRad)
        nEndY = imgCenterY - lineLen * math.cos(needleRad)
        draw.line([(imgCenterX, imgCenterY), (nEndX, nEndY)], fill=(0, 0, 255), width=1)

        draw.circle((xmean, ymean), 2, fill=(0, 255, 0))

        draw.text((0, 0), f"VAL: {readout:.2f}", fill=(255, 0, 0))
        imgVis.save(f"debug_output/{name}_markers.png")

    return readout


def cv2ToPILImage(img):
    converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(converted)

def getNeedlePosition(img, dialopt, isOpenCvImage):
    if isOpenCvImage:
        img = cv2ToPILImage(img)
    
    img = gaugeCrop(img, dialopt)
    img = img.convert("RGB")
    reading = determineNeedle(img, dialopt)

    return reading



"""
def main():
    opts = None
    with open("config.json") as cfgfile:
        opts = json.load(cfgfile)
    
    if opts is None:
        print("Error opening file")
        return
    
    if os.path.exists("debug_output"):
        shutil.rmtree("debug_output")
    os.mkdir("debug_output")
    
    global debugMode 
    debugMode = opts["debug"]

    img = Image.open("imgs/img3.png")

    for dial in opts["dials"]:
        gaugeImg = img.copy()
        reading = getNeedlePosition(gaugeImg, dial, False)

        print(f"[{dial["name"]}] Got reading of {reading}")

if __name__ == "__main__":
    main()
"""