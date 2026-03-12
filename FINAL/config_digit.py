#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import json
import os
import sys

# =====================================================
#           CONFIGURACIÓN INICIAL
# =====================================================

nombre_imagen = "captura_20251203_171032.png"   
puntos_digit = []
img_raw = None
img_prep = None
img_hsv = None

# HSV inicial
h1_min, h1_max = 0, 25
s_min, s_max = 30, 255
v_min, v_max = 70, 255
h2_min, h2_max = 160, 179

x0_roi = 0
y0_roi = 0


# =====================================================
#         PREPROCESAMIENTO DEL DÍGITO
# =====================================================

def gray_world(img):
    img = img.astype(np.float32)
    b,g,r = cv2.split(img)
    mb,mg,mr = b.mean(), g.mean(), r.mean()
    mean = (mb+mg+mr)/3.0
    b *= mean/(mb+1e-6)
    g *= mean/(mg+1e-6)
    r *= mean/(mr+1e-6)
    return np.clip(cv2.merge([b,g,r]),0,255).astype(np.uint8)

def clahe_l(img, clip=2.5):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L,A,B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip,tileGridSize=(8,8))
    L2 = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L2,A,B]), cv2.COLOR_LAB2BGR)

def preprocess_digit(img):
    img = gray_world(img)
    img = clahe_l(img, clip=2.5)

    # realzar blancos y rojo/verde del dígito
    b,g,r = cv2.split(img)
    r = cv2.multiply(r, 1.35)
    g = cv2.multiply(g, 0.85)
    img = cv2.merge([b,g,r])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    v = cv2.add(v, 35)
    v = np.clip(v, 0,255)
    img = cv2.cvtColor(cv2.merge([h,s,v]), cv2.COLOR_HSV2BGR)
    return img


# =====================================================
#         SELECCIÓN DE ROI (4 CLICS)
# =====================================================

def mouse_roi(event, x, y, flags, param):
    global puntos_digit, img_display

    if event == cv2.EVENT_LBUTTONDOWN and len(puntos_digit) < 4:
        puntos_digit.append((x,y))
        print(f" Punto {len(puntos_digit)} = {x,y}")

    img_display = img_prep.copy()

    for p in puntos_digit:
        cv2.circle(img_display, p, 4, (0,255,255), -1)

    if len(puntos_digit) == 4:
        cv2.polylines(img_display, [np.array(puntos_digit)], True, (0,255,0), 2)

    cv2.putText(img_display, "Selecciona las 4 esquinas del digito d8",
                (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)


# =====================================================
#         AJUSTE HSV POST-ROI
# =====================================================

def mouse_hsv(event, x, y, flags, param):
    global img_hsv, x0_roi, y0_roi
    global h1_min, h1_max, h2_min, h2_max, s_min, s_max, v_min, v_max

    if event == cv2.EVENT_LBUTTONDOWN:
        xo = x + x0_roi
        yo = y + y0_roi
        h,s,v = img_hsv[yo, xo]
        print(f" HSV clic = ({h},{s},{v})")

        tol_h = 12
        tol_sv = 50

        h1_min = max(0, h - tol_h)
        h1_max = min(179, h + tol_h)
        s_min = max(0, s - tol_sv)
        s_max = min(255, s + tol_sv)
        v_min = max(0, v - tol_sv)
        v_max = min(255, v + tol_sv)

        if h < 10 or h > 170:
            h2_min = 170
            h2_max = 179
        else:
            h2_min = 0
            h2_max = 0

        # actualizar sliders automáticamente
        cv2.setTrackbarPos("H Min1", "HSV Panel", h1_min)
        cv2.setTrackbarPos("H Max1", "HSV Panel", h1_max)
        cv2.setTrackbarPos("S Min",  "HSV Panel", s_min)
        cv2.setTrackbarPos("S Max",  "HSV Panel", s_max)
        cv2.setTrackbarPos("V Min",  "HSV Panel", v_min)
        cv2.setTrackbarPos("V Max",  "HSV Panel", v_max)
        cv2.setTrackbarPos("H Min2", "HSV Panel", h2_min)
        cv2.setTrackbarPos("H Max2", "HSV Panel", h2_max)



# =====================================================
#                  MAIN
# =====================================================

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, nombre_imagen)

    img_raw = cv2.imread(img_path)
    if img_raw is None:
        print("ERROR: no se pudo cargar la imagen.")
        sys.exit()

    img_prep = preprocess_digit(img_raw)
    img_display = img_prep.copy()

    # ---------------------------------------------
    # PARTE 1: SELECCIONAR 4 PUNTOS DEL DÍGITO
    # ---------------------------------------------

    print("PARTE 1 -> Selecciona las 4 esquinas del dígito d8.")
    print("Cuando termines, pulsa 'c'. Para salir: 'q'.")

    cv2.namedWindow("ROI Digit")
    cv2.setMouseCallback("ROI Digit", mouse_roi)

    while True:
        cv2.imshow("ROI Digit", img_display)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            sys.exit()
        if k == ord('c') and len(puntos_digit) == 4:
            break

    cv2.destroyWindow("ROI Digit")

    p = np.array(puntos_digit)
    x0 = np.min(p[:,0])
    y0 = np.min(p[:,1])
    x1 = np.max(p[:,0])
    y1 = np.max(p[:,1])

    x0_roi = x0
    y0_roi = y0

    roi = img_prep[y0:y1, x0:x1]
    img_hsv = cv2.cvtColor(img_prep, cv2.COLOR_BGR2HSV)

    # ---------------------------------------------
    # PARTE 2: AJUSTAR HSV
    # ---------------------------------------------

    cv2.namedWindow("HSV Panel")
    cv2.createTrackbar("H Min1","HSV Panel",h1_min,179,lambda x:None)
    cv2.createTrackbar("H Max1","HSV Panel",h1_max,179,lambda x:None)
    cv2.createTrackbar("S Min","HSV Panel",s_min,255,lambda x:None)
    cv2.createTrackbar("S Max","HSV Panel",s_max,255,lambda x:None)
    cv2.createTrackbar("V Min","HSV Panel",v_min,255,lambda x:None)
    cv2.createTrackbar("V Max","HSV Panel",v_max,255,lambda x:None)
    cv2.createTrackbar("H Min2","HSV Panel",h2_min,179,lambda x:None)
    cv2.createTrackbar("H Max2","HSV Panel",h2_max,179,lambda x:None)

    cv2.namedWindow("HSV Preview")
    cv2.setMouseCallback("HSV Preview", mouse_hsv)

    print("\nPARTE 2 -> Ajusta HSV. Clickeando sobre el dígito se autoadapta.")
    print("Pulsa 'g' para guardar. 'q' para salir.")

    while True:
        h1_min = cv2.getTrackbarPos("H Min1","HSV Panel")
        h1_max = cv2.getTrackbarPos("H Max1","HSV Panel")
        s_min  = cv2.getTrackbarPos("S Min","HSV Panel")
        s_max  = cv2.getTrackbarPos("S Max","HSV Panel")
        v_min  = cv2.getTrackbarPos("V Min","HSV Panel")
        v_max  = cv2.getTrackbarPos("V Max","HSV Panel")
        h2_min = cv2.getTrackbarPos("H Min2","HSV Panel")
        h2_max = cv2.getTrackbarPos("H Max2","HSV Panel")

        lower1 = np.array([h1_min, s_min, v_min])
        upper1 = np.array([h1_max, s_max, v_max])
        lower2 = np.array([h2_min, s_min, v_min])
        upper2 = np.array([h2_max, s_max, v_max])

        mask1 = cv2.inRange(img_hsv[y0:y1, x0:x1], lower1, upper1)
        mask2 = cv2.inRange(img_hsv[y0:y1, x0:x1], lower2, upper2)
        mask = mask1 + mask2

        filtered = cv2.bitwise_and(roi, roi, mask=mask)

        cv2.imshow("HSV Preview", filtered)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('g'):
            config = {
                "digit8": {
                    "points": puntos_digit,
                    
                }
            }
            with open("config.json", "w") as f:
                json.dump(config,f,indent=4)

            print("\nGuardado en config.json")
            break

        if k == ord('q'):
            break

    cv2.destroyAllWindows()
