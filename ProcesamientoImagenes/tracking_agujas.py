import cv2
import numpy as np
import math
import json
import os
import sys
from typing import Optional, Tuple

def cargar_config(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    try:
        cx, cy = cfg["centro"]
        r = int(cfg["radio"])
        ang0 = float(cfg["angulo_cero"])
        ang1 = float(cfg["angulo_uno"])
        tot = int(cfg["total_numeros"])
        sentido = cfg.get("sentido", "CW").upper()
        assert sentido in ("CW", "CCW")
        hsv_ranges = cfg["hsv_ranges"]
        return (int(cx), int(cy)), r, ang0, ang1, tot, sentido, hsv_ranges
    except KeyError as e:
        print(f"Error: archivo de configuración inválido. Falta llave: {e}")
        sys.exit()

def angulo_grados(center, tip):
    (cx, cy), (px, py) = center, tip
    ang = math.degrees(math.atan2(cy - py, px - cx))
    return (ang + 360) % 360

def distancia(a, b):
    return float(np.hypot(a[0]-b[0], a[1]-b[1]))

def normalizar_ruta(base_file: str):
    if os.path.isabs(base_file):
        return base_file
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir, base_file)
    except NameError:
        return os.path.abspath(base_file)

def dibujar_resultado(img, center, r, tip, angle_deg, valor, metodo, out_path="salida_reloj.png"):
    out = img.copy()
    cx, cy = map(int, center)
    cv2.circle(out, (cx, cy), r, (0, 255, 255), 2)
    cv2.circle(out, (cx, cy), 4, (0, 255, 0), -1)
    cv2.circle(out, (int(tip[0]), int(tip[1])), 6, (0, 0, 255), -1)
    cv2.line(out, (cx, cy), (int(tip[0]), int(tip[1])), (255, 255, 255), 2)
    texto1 = f"Angulo: {angle_deg:.2f}  Valor: {valor:.3f}"
    texto2 = f"Metodo: {metodo}"
    pad = 10
    cv2.rectangle(out, (pad, pad), (pad + 460, pad + 52), (0, 0, 0), -1)
    cv2.putText(out, texto1, (pad+8, pad+22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, texto2, (pad+8, pad+44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, out)
    print(texto1 + f"  [{metodo}]")
    #print(f"Imagen anotada guardada en: {out_path}")

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

def preprocesar(img_bgr):
    print("[Preprocesar] Gray World")
    img_wb = gray_world(img_bgr)
    print("[Preprocesar] CLAHe")
    img_clahe = clahe_l(img_wb, clip=2.5)
    print("[Preprocesar] MedianBlur")
    blur = cv2.medianBlur(img_clahe, 5)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
    return hsv, img_clahe

def generar_masks(img_shape, center, r):
    h, w = img_shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, center, r, 255, -1)
    ring = np.zeros_like(mask)
    cv2.circle(ring, center, r, 255, 2)
    return mask, ring

def bordes_y_umbral(hsv_img, roi_mask, hsv_ranges):
    r1 = hsv_ranges['range1']
    r2 = hsv_ranges['range2']
    lower1 = np.array([r1['h_min'], r1['s_min'], r1['v_min']])
    upper1 = np.array([r1['h_max'], r1['s_max'], r1['v_max']])
    lower2 = np.array([r2['h_min'], r2['s_min'], r2['v_min']])
    upper2 = np.array([r2['h_max'], r2['s_max'], r2['v_max']])

    mask_color = cv2.inRange(hsv_img, lower1, upper1) + cv2.inRange(hsv_img, lower2, upper2)
    v_channel = hsv_img[:, :, 2]

    mask_brightness = cv2.adaptiveThreshold(
        v_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )

    th = cv2.bitwise_and(mask_color, mask_brightness)
    th = cv2.bitwise_and(th, roi_mask)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_open, iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c_main = max(contours, key=cv2.contourArea)
        th_cleaned = np.zeros_like(th)
        cv2.drawContours(th_cleaned, [c_main], -1, 255, -1)
        th = th_cleaned
    else:
        print("[bordes_y_umbral] No se encontraron contornos en th.")

    gray_for_edges = hsv_img[:, :, 2]
    edges = cv2.Canny(gray_for_edges, 10, 60)
    edges = cv2.bitwise_and(edges, roi_mask)

    return th, edges

def estimar_centro_desde_cfg(center):
    return center

def punta_por_bordes(edges, center, r) -> Optional[Tuple[int, int]]:
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[bordes] No se encontraron contornos.")
        return None
    candidatos = []
    for c in contours:
        if cv2.arcLength(c, True) < 10:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if distancia((x+w/2, y+h/2), center) > r * 1.5:
            continue
        candidatos.append(c)
    if not candidatos:
        print("[bordes] No hay contornos válidos cerca del centro.")
        return None
    c = max(candidatos, key=lambda cnt: cv2.arcLength(cnt, True))
    far = None
    maxd = -1
    for p in c.reshape(-1, 2):
        p = (int(p[0]), int(p[1]))
        d = distancia(p, center)
        if d <= r * 1.05 and d > maxd:
            maxd = d
            far = p
    if far is not None:
        print(f"[bordes] Punta encontrada en {far} (distancia max: {maxd:.2f})")
    else:
        print("[bordes] No se pudo encontrar un punto lejano.")
    return far

def punta_por_contorno(mask, center, r) -> Optional[Tuple[int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cx, cy = center
    candidatos = []
    for c in contours:
        if cv2.contourArea(c) < 100:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        mx, my = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        if distancia((mx, my), center) <= r * 1.05:
            candidatos.append(c)
    if not candidatos:
        return None
    #print(f"[contorno] {len(candidatos)} candidatos. Usando el mayor.")
    c = max(candidatos, key=cv2.contourArea)
    far = None
    maxd = -1
    for p in c.reshape(-1, 2):
        p = (int(p[0]), int(p[1]))
        if distancia(p, center) <= r * 1.05:
            d = distancia(p, center)
            if d > maxd:
                maxd = d
                far = p
    if far is not None:
        print(f"[contorno] Punta {far} (dist {maxd:.2f})")
    else:
        print("[contorno] No se encontró un punto lejano en el contorno.")
    return far

def punta_por_vertices(
    mask,
    center,
    r,
    img_debug: Optional[np.ndarray] = None,
    debug_dir: Optional[str] = "agujas_tracking/resultados",
    tag: str = "vertices",
    dist_min_ratio: float = 0.6,
    dist_max_ratio: float = 1.15,
    ang_max_deg: float = 65.0,
    approx_eps_scale: float = 0.01,
):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print("[vertices] No hay contornos en mask.")
        return None

    c = max(contours, key=cv2.contourArea)
    per = cv2.arcLength(c, True)
    eps = max(1.0, approx_eps_scale * per)
    approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        print("[vertices] approx tiene menos de 3 vértices.")
        return None

    if img_debug is None:
        canvas = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        canvas = img_debug.copy()
    cx, cy = center
    cv2.circle(canvas, (cx, cy), int(r), (0, 255, 255), 1)
    cv2.drawContours(canvas, [c], -1, (255, 255, 0), 1)
    cv2.polylines(canvas, [approx.reshape(-1, 1, 2)], True, (0, 255, 255), 2)

    def ang(a, b, c_):
        ba = a - b
        bc = c_ - b
        denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosang = float(np.dot(ba, bc) / denom)
        return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

    n = len(approx)
    candidatos = []

    dist_min_abs = dist_min_ratio * r
    dist_max_abs = dist_max_ratio * r

    #print(f"[vertices] Dist [{dist_min_abs:.1f}, {dist_max_abs:.1f}], AngMax {ang_max_deg:.1f}")

    for i in range(n):
        p = approx[i].astype(np.float32)
        p_tuple = (int(p[0]), int(p[1]))
        d = distancia((float(p[0]), float(p[1])), center)
        p_prev = approx[(i - 1 + n) % n].astype(np.float32)
        p_next = approx[(i + 1) % n].astype(np.float32)
        a = ang(p_prev, p, p_next)

        label = f"d={d:.1f} a={a:.1f}"
        cv2.circle(canvas, p_tuple, 3, (255, 255, 255), -1)

        es_dist_ok = (dist_min_abs <= d <= dist_max_abs)
        es_ang_ok = (a <= ang_max_deg)

        if es_dist_ok and es_ang_ok:
            candidatos.append((p_tuple, d, a))
            cv2.circle(canvas, p_tuple, 8, (0, 165, 255), 2)
            cv2.putText(canvas, label, (p_tuple[0] + 5, p_tuple[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        else:
            cv2.circle(canvas, p_tuple, 8, (120, 120, 120), 1)
            cv2.putText(canvas, label, (p_tuple[0] + 5, p_tuple[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    if not candidatos:
        print("[vertices] Sin candidatos validos.")
        if debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)
            dbg_full = os.path.join(debug_dir, f"debug_{tag}_FALLO.png")
            cv2.imwrite(dbg_full, canvas)
        return None

    mejor_punta, _, _ = max(candidatos, key=lambda x: x[1])

    cv2.circle(canvas, mejor_punta, 10, (0, 255, 0), 2)
    label = "PUNTA"
    cv2.putText(canvas, label, (mejor_punta[0] + 8, mejor_punta[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        dbg_full = os.path.join(debug_dir, f"debug_{tag}.png")
        cv2.imwrite(dbg_full, canvas)
        x0, y0 = max(0, cx - int(r)), max(0, cy - int(r))
        x1, y1 = min(canvas.shape[1], cx + int(r)), min(canvas.shape[0], cy + int(r))
        roi = canvas[y0:y1, x0:x1]
        dbg_roi = os.path.join(debug_dir, f"debug_{tag}_roi.png")
        cv2.imwrite(dbg_roi, roi)
        #print(f"[vertices] Debug guardado: {dbg_full}")
        #print(f"[vertices] Debug ROI guardado: {dbg_roi}")

    return mejor_punta

def angle_to_value(angle_deg, angulo_cero, angulo_uno, total_numeros, sentido="CW"):
    if sentido == "CW":
        angulo_paso = (angulo_cero - angulo_uno + 360) % 360
    else:
        angulo_paso = (angulo_uno - angulo_cero + 360) % 360
    if angulo_paso > 180:
        angulo_paso = 360 - angulo_paso
    if angulo_paso < 0.1:
        angulo_paso = 360.0 / total_numeros
    if sentido == "CW":
        delta = (angulo_cero - angle_deg) % 360.0
    else:
        delta = (angle_deg - angulo_cero) % 360.0

    if sentido == "CCW":
        valor = (10 / 360) * ((360 - delta) % 360)
    else:
        valor = (10 / 360) * delta

    #print(f"[Calculo] paso={angulo_paso:.2f}, delta={delta:.2f}, valor={valor:.3f}")
    return delta, valor

def determineNeedle(img, cfg_path="config_dial.json"):
    output_dir = "agujas_tracking/resultados"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Guardando resultados en: {output_dir}")

    cfg_path = normalizar_ruta(cfg_path)
    if not os.path.exists(cfg_path):
        print(f"No se encontró config en: {cfg_path}")
        sys.exit(1)

    center, r, ang0, ang1, total_nums, sentido, hsv_ranges = cargar_config(cfg_path)
    print(f"Config: centro={center}, r={r}, ang0={ang0:.2f}, ang1={ang1:.2f}, sentido={sentido}")

    img = np.array(img)
    if img is None:
        raise Exception("img None")

    hsv, img_mejorada = preprocesar(img)

    roi_mask, _ = generar_masks(hsv.shape, center, r)
    mask, edges = bordes_y_umbral(hsv, roi_mask, hsv_ranges)

    h_img, w_img = img.shape[:2]
    x_min_roi = max(0, center[0] - r)
    y_min_roi = max(0, center[1] - r)
    x_max_roi = min(w_img, center[0] + r)
    y_max_roi = min(h_img, center[1] + r)

    mask_rec = mask[y_min_roi:y_max_roi, x_min_roi:x_max_roi]
    cv2.imwrite(os.path.join(output_dir, "debug_mask.png"), mask_rec)

    edges_rec = edges[y_min_roi:y_max_roi, x_min_roi:x_max_roi]
    cv2.imwrite(os.path.join(output_dir, "debug_edges.png"), edges_rec)

    print("=" * 50)
    print("INICIANDO DETECCIONES")
    print("=" * 50)

    tip = punta_por_vertices(
        mask,
        center,
        r,
        img_debug=img_mejorada.copy(),
        debug_dir=output_dir,
        tag="vertices_debug"
    )
    metodo = "vertices"

    if tip is None:
        #print("No hay punta por vertices, probando contorno")
        tip = punta_por_contorno(mask, center, r)
        metodo = "contorno"

    if tip is None:
        #print("No hay punta por contorno, probando bordes")
        tip = punta_por_bordes(edges, center, r)
        metodo = "bordes"

    if tip is None:
        #print("No se logro detectar la punta con ningun metodo.")
        sys.exit(2)

    angle = angulo_grados(center, tip)
    _, valor = angle_to_value(angle, ang0, ang1, total_nums, sentido=sentido)

    path_salida = os.path.join(output_dir, "salida_reloj.png")
    dibujar_resultado(img, center, r, tip, angle, valor, metodo, out_path=path_salida)
    return valor

if __name__ == "__main__":
    import shutil
    try:
        shutil.rmtree("agujas_tracking/resultados")
    except FileNotFoundError:
        pass
    img_arg = sys.argv[1] if len(sys.argv) > 1 else "3.jpg"
    cfg_arg = sys.argv[2] if len(sys.argv) > 2 else "config_dial.json"
    v = determineNeedle(img_arg, cfg_arg)
    print(f"VALOR FINAL: {v}")
