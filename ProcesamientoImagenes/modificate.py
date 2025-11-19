import cv2
import numpy as np
import os

# -----------------------------
# Utilidades
# -----------------------------
def gray_world(img_bgr):
    img = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    mb, mg, mr = b.mean(), g.mean(), r.mean()
    m = (mb + mg + mr) / 3.0
    b = b * (m / (mb + 1e-6))
    g = g * (m / (mg + 1e-6))
    r = r * (m / (mr + 1e-6))
    return np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)

def clahe_l(img_bgr, clip=2.0, tiles=(8,8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

def mask_red_hsv(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 70, 60], np.uint8)
    upper1 = np.array([10, 255, 255], np.uint8)
    lower2 = np.array([170, 70, 60], np.uint8)
    upper2 = np.array([180, 255, 255], np.uint8)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                          cv2.inRange(hsv, lower2, upper2))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return cv2.dilate(mask, k, iterations=1)

def boost_red_area(img_bgr, mask):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.where(mask>0, np.clip(s*1.6, 0, 255), s).astype(np.uint8)
    bgr_sat = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    overlay = bgr_sat.copy()
    overlay[mask>0] = (0, 0, 255)
    return cv2.addWeighted(bgr_sat, 1.0, overlay, 0.25, 0)

# -----------------------------
# Pipeline principal
# -----------------------------
def enhance_red_dial(filename, out_prefix="sensus"):
    # Ruta absoluta del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, filename)

    print(f"Buscando imagen en: {img_path}")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"No se encontró la imagen: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        raise IOError(f"No se pudo leer correctamente: {img_path}")

    wb = gray_world(img)
    enh = clahe_l(wb, clip=2.5)
    mask = mask_red_hsv(enh)
    highlighted = boost_red_area(enh, mask)

    dbg = highlighted.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(dbg, cnts, -1, (0,255,0), 2)

    out_dir = os.path.join(script_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(os.path.join(out_dir, f"{out_prefix}_wb.jpg"), wb)
    cv2.imwrite(os.path.join(out_dir, f"{out_prefix}_enhanced.jpg"), enh)
    cv2.imwrite(os.path.join(out_dir, f"{out_prefix}_mask.png"), mask)
    cv2.imwrite(os.path.join(out_dir, f"{out_prefix}_dial_highlight.jpg"), highlighted)
    cv2.imwrite(os.path.join(out_dir, f"{out_prefix}_debug_contours.jpg"), dbg)

    print("Resultados guardados en carpeta 'outputs' dentro del script.")

# -----------------------------
# Ejecución
# -----------------------------
if __name__ == "__main__":
    enhance_red_dial("3.jpg", out_prefix="sensus")
