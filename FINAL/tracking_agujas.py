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

        sys.exit()

def angulo_grados(center, tip):
    (cx, cy), (px, py) = center, tip
    ang = math.degrees(math.atan2(cy - py, px - cx))
    return (ang + 360) % 360

def distancia(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

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
    if tip is not None:
        cv2.circle(out, (int(tip[0]), int(tip[1])), 6, (0, 0, 255), -1)
        cv2.line(out, (cx, cy), (int(tip[0]), int(tip[1])), (255, 255, 255), 2)
    texto1 = f"Angulo: {angle_deg:.2f}  Valor: {valor:.3f}"
    texto2 = f"Metodo: {metodo}"
    pad = 10
    cv2.rectangle(out, (pad, pad), (pad + 460, pad + 52), (0, 0, 0), -1)
    cv2.putText(out, texto1, (pad + 8, pad + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, texto2, (pad + 8, pad + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, out)
    print(texto1 + f"  [{metodo}]")

def gray_world(img_bgr):
    img = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    mb, mg, mr = b.mean(), g.mean(), r.mean()
    m = (mb + mg + mr) / 3.0
    b = b * (m / (mb + 1e-6))
    g = g * (m / (mg + 1e-6))
    r = r * (m / (mr + 1e-6))
    return np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)

def clahe_l(img_bgr, clip=2.0, tiles=(8, 8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

def preprocesar(img_bgr):
    #print("[Preprocesar] Gray World")
    img_wb = gray_world(img_bgr)
    #print("[Preprocesar] CLAHe")
    img_clahe = clahe_l(img_wb, clip=2.5)
    #print("[Preprocesar] MedianBlur")
    blur = cv2.medianBlur(img_clahe, 5)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
    return hsv, img_clahe

def generar_masks(img_shape, center, r):
    h, w = img_shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, center, r, 255, -1)   # ROI real

    ring = np.zeros_like(mask)

    return mask, ring

def enfatizar_rojo_hsv(hsv_img, hsv_ranges,
                       sat_factor: float = 1.4,
                       val_factor: float = 1.10):
    """
    Aumenta saturación (S) y brillo (V) sólo en los píxeles
    que ya caen dentro de los rangos de rojo (hsv_ranges).

    sat_factor : cuánto multiplicar S (>=1.0)
    val_factor : cuánto multiplicar V (>=1.0)
    """
    r1 = hsv_ranges['range1']
    r2 = hsv_ranges['range2']
    lower1 = np.array([r1['h_min'], r1['s_min'], r1['v_min']])
    upper1 = np.array([r1['h_max'], r1['s_max'], r1['v_max']])
    lower2 = np.array([r2['h_min'], r2['s_min'], r2['v_min']])
    upper2 = np.array([r2['h_max'], r2['s_max'], r2['v_max']])

    mask1 = cv2.inRange(hsv_img, lower1, upper1)
    mask2 = cv2.inRange(hsv_img, lower2, upper2)
    mask_red = mask1 + mask2    

    # Separar canales HSV
    h, s, v = cv2.split(hsv_img)
    s = s.astype(np.float32)
    v = v.astype(np.float32)

    red_pixels = (mask_red > 0)

    s[red_pixels] *= sat_factor
    v[red_pixels] *= val_factor


    s = np.clip(s, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)

    hsv_boost = cv2.merge([h, s, v])
    return hsv_boost

def bordes_y_umbral(
    hsv_img,
    roi_mask,
    hsv_ranges,
    center,
    r,
    center_ratio: float = 0.25,          
    min_area: float = 80.0,              
    min_radial_span_ratio: float = 0.2  
):
    """
    Genera:
      - mask_final: máscara binaria que intenta dejar SOLO la aguja (posible-
        mente extendida hasta el radio si la segmentación no llega).
      - edges_final: bordes Canny recortados a la misma región.
    """

    # ---- 1) Segmentación por color + brillo ----
    r1 = hsv_ranges['range1']
    r2 = hsv_ranges['range2']
    lower1 = np.array([r1['h_min'], r1['s_min'], r1['v_min']])
    upper1 = np.array([r1['h_max'], r1['s_max'], r1['v_max']])
    lower2 = np.array([r2['h_min'], r2['s_min'], r2['v_min']])
    upper2 = np.array([r2['h_max'], r2['s_max'], r2['v_max']])

    mask_color = cv2.inRange(hsv_img, lower1, upper1) + cv2.inRange(hsv_img, lower2, upper2)

    v_channel = hsv_img[:, :, 2]
    mask_brightness = cv2.adaptiveThreshold(
        v_channel, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )

    th = cv2.bitwise_and(mask_color, mask_brightness)
    th = cv2.bitwise_and(th, roi_mask)

    # ---- 2) Limpieza morfológica ----
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # ---- 3) Eliminar un disco PEQUEÑO en el centro ----
    h, w = th.shape[:2]
    cx, cy = center
    r_center = int(r * center_ratio)

    center_mask = np.zeros_like(th)
    cv2.circle(center_mask, (cx, cy), r_center, 255, -1)
    th = cv2.bitwise_and(th, cv2.bitwise_not(center_mask))

    # ---- 4) Elegir el blob "más aguja" según span radial ----
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_final = np.zeros_like(th)
    best_score = 0.0
    best_contour = None

    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue

            pts = c.reshape(-1, 2).astype(np.float32)
            dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
            if len(dists) == 0:
                continue

            d_min = float(dists.min())
            d_max = float(dists.max())
            radial_span = d_max - d_min

            span_ratio = radial_span / float(r + 1e-6)
            if span_ratio < min_radial_span_ratio:
                continue

            mean_dist = float(dists.mean())
            score = radial_span + 0.3 * mean_dist

            if score > best_score:
                best_score = score
                best_contour = c
                mask_final[:] = 0
                cv2.drawContours(mask_final, [c], -1, 255, -1)

    if best_score == 0.0 or best_contour is None:
        
        mask_final = th.copy()
    else:

        pts = best_contour.reshape(-1, 2).astype(np.float32)
        dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        d_max = float(dists.max())
        idx_max = int(np.argmax(dists))
        px, py = float(pts[idx_max, 0]), float(pts[idx_max, 1])

        if d_max < 0.2 * r:
            vx = px - cx
            vy = py - cy
            L = math.hypot(vx, vy)
            if L > 1e-3:
                vx /= L
                vy /= L
                r_ext = r * 1.02
                ex = int(round(cx + vx * r_ext))
                ey = int(round(cy + vy * r_ext))
                cv2.line(mask_final, (cx, cy), (ex, ey), 255, 3)

    # ---- 5) Bordes Canny restringidos a la misma región ----
    gray_for_edges = hsv_img[:, :, 2]
    edges = cv2.Canny(gray_for_edges, 10, 60)
    edges = cv2.bitwise_and(edges, roi_mask)
    edges = cv2.bitwise_and(edges, mask_final)

    return mask_final, edges


def estimar_centro_desde_cfg(center):
    return center

def punta_por_bordes(edges, center, r) -> Optional[Tuple[int, int]]:
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    candidatos = []
    for c in contours:
        if cv2.arcLength(c, True) < 10:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if distancia((x + w / 2, y + h / 2), center) > r * 1.5:
            continue
        candidatos.append(c)
    if not candidatos:
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
        print("")
    else:
        print("")

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
        mx, my = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        if distancia((mx, my), center) <= r * 1.05:
            candidatos.append(c)
    if not candidatos:
        return None
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
        print("")

    else:
        print("")
    return far
def punta_por_vector_masa(
    mask: np.ndarray,
    center: Tuple[int, int],
    r: float,
    min_pixels: int = 50,
    debug_canvas: Optional[np.ndarray] = None,
    color_vector: Tuple[int, int, int] = (0, 255, 0)
) -> Optional[Tuple[int, int]]:
    """
    Estima el eje de la aguja usando un 'vector de masa':
      - Toma todos los píxeles blancos de la máscara.
      - Suma los vectores desde el centro, ponderados por la distancia.
      - Usa la dirección resultante para extrapolar la punta en el círculo de radio r.
    """

    ys, xs = np.where(mask > 0)
    if len(xs) < min_pixels:
        return None

    cx, cy = center
    dx = xs.astype(np.float32) - float(cx)
    dy = ys.astype(np.float32) - float(cy)

    dist = np.hypot(dx, dy)

    w = dist
    w[dist < 1.0] = 1.0

    # Vector resultante ponderado
    vx = float(np.sum(dx * w))
    vy = float(np.sum(dy * w))

    norm = math.hypot(vx, vy)
    if norm < 1e-3:
        return None

    ux, uy = vx / norm, vy / norm

    tip_x = int(round(cx + ux * r))
    tip_y = int(round(cy + uy * r))
    tip = (tip_x, tip_y)


    if debug_canvas is not None:
        cv2.line(debug_canvas, (cx, cy), tip, color_vector, 2)
        cv2.circle(debug_canvas, tip, 7, color_vector, 2)

    return tip

def punta_por_vertices(
    mask,
    center,
    r,
    img_debug: Optional[np.ndarray] = None,
    debug_dir: Optional[str] = "Aguja_resultados",
    tag: str = "vertices",
    ang_max_deg: float = 53.0,
    approx_eps_scale: float = 0.01,
):
    """
    Detecta la punta usando vértices del contorno principal.

    - Selección por rango radial + ángulo + soporte radial.
    - Si NO hay candidatos -> fallback a punta_por_vector_masa().
    - Si la mejor punta está MUY desalineada con la “masa” de la máscara
      y tiene poco soporte radial -> también fallback a punta_por_vector_masa().
    """

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    # Contorno principal
    c = max(contours, key=cv2.contourArea)
    per = cv2.arcLength(c, True)
    eps = max(1.0, approx_eps_scale * per)
    approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    if len(approx) < 3:
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

    # --- distancias y rango dinámico ---
    dists = np.array(
        [distancia((float(p[0]), float(p[1])), center) for p in approx],
        dtype=np.float32
    )
    d_max = float(dists.max())
    d_med = float(np.median(dists))

    dist_min_abs = d_med + 0.3 * (d_max - d_med)
    dist_max_abs = d_max + 3.0

    # --------- ÁNGULO DE MASA (del parcial exterior de la máscara) ----------
    ang_masa = None
    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        total_pix = len(xs)
        dx = xs.astype(np.float32) - cx
        dy = cy - ys.astype(np.float32)  
        radios = np.hypot(dx, dy)

        outer_ratio = 0.55
        sel = radios >= (outer_ratio * r)

        if np.any(sel):
            outer_count = int(sel.sum())
            min_outer_frac = 0.15  
            if outer_count >= min_outer_frac * total_pix:
                dx_sel = dx[sel]
                dy_sel = dy[sel]
                ang_rad = np.arctan2(dy_sel, dx_sel)
                sin_mean = float(np.sin(ang_rad).mean())
                cos_mean = float(np.cos(ang_rad).mean())
                ang_masa = (math.degrees(math.atan2(sin_mean, cos_mean)) + 360.0) % 360.0

            else:
                print("")
        else:
            print("")

    # --------- soporte radial ----------
    def soporte_radial(mask_local, center_local, p, max_len):
        h, w = mask_local.shape[:2]
        cx_, cy_ = center_local
        px, py = p
        dx_ = cx_ - px
        dy_ = cy_ - py
        L = math.hypot(dx_, dy_)
        if L < 1:
            return 0
        ux, uy = dx_ / L, dy_ / L

        pasos = int(min(L, max_len))
        count = 0
        for k in range(pasos):
            x = int(round(px + ux * k))
            y = int(round(py + uy * k))
            if 0 <= x < w and 0 <= y < h and mask_local[y, x] > 0:
                count += 1
            else:
                break
        return count

    n = len(approx)
    candidatos = []
    max_pasillos = int(r)

    for i in range(n):
        p = approx[i].astype(np.float32)
        p_tuple = (int(p[0]), int(p[1]))
        d = dists[i]

        p_prev = approx[(i - 1 + n) % n].astype(np.float32)
        p_next = approx[(i + 1) % n].astype(np.float32)
        a = ang(p_prev, p, p_next)

        es_dist_ok = (dist_min_abs <= d <= dist_max_abs)
        es_ang_ok = (a <= ang_max_deg)

        label = f"d={d:.1f} a={a:.1f}"
        cv2.circle(canvas, p_tuple, 3, (255, 255, 255), -1)

        if es_dist_ok and es_ang_ok:
            sr = soporte_radial(mask, center, p_tuple, max_pasillos)
            candidatos.append((p_tuple, d, a, sr))
            label_sr = f"{label} s={sr}"
            cv2.circle(canvas, p_tuple, 8, (0, 165, 255), 2)
            cv2.putText(canvas, label_sr, (p_tuple[0] + 5, p_tuple[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        else:
            cv2.circle(canvas, p_tuple, 8, (120, 120, 120), 1)
            cv2.putText(canvas, label, (p_tuple[0] + 5, p_tuple[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    # ---------- CASO 1: SIN CANDIDATOS -> VECTOR DE MASA ----------
    if not candidatos:

        if debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)
            dbg_full = os.path.join(debug_dir, f"debug_{tag}_FALLO.png")
            cv2.imwrite(dbg_full, canvas)


        tip_masa = punta_por_vector_masa(mask, center, r)
        if tip_masa is not None:
            cv2.circle(canvas, tip_masa, 10, (0, 255, 0), 2)
            cv2.putText(canvas, "PUNTA_MASA",
                        (tip_masa[0] + 8, tip_masa[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            if debug_dir is not None:
                dbg_full_masa = os.path.join(debug_dir, f"debug_{tag}_MASA.png")
                cv2.imwrite(dbg_full_masa, canvas)

            return tip_masa

        return None

    # ---------- selección normal por soporte radial ----------
    min_soporte = 0.25 * r
    candidatos_filtrados = [c for c in candidatos if c[3] >= min_soporte]

    if not candidatos_filtrados:

        candidatos_filtrados = candidatos

    mejor_punta, _, _, mejor_sr = max(
        candidatos_filtrados,
        key=lambda x: (x[3], x[1])
    )

    # ---------- CASO 2: PUNTA SOSPECHOSA vs MASA -> VECTOR DE MASA ----------
    def angulo_grados_local(center_local, tip_local):
        (cx_, cy_), (px_, py_) = center_local, tip_local
        ang_ = math.degrees(math.atan2(cy_ - py_, px_ - cx_))
        return (ang_ + 360.0) % 360.0

    if ang_masa is not None:
        ang_punta = angulo_grados_local(center, mejor_punta)
        diff = abs(((ang_punta - ang_masa + 180.0) % 360.0) - 180.0)


        MAX_DIFF_MASA = 50.0       # muy desalineado
        MIN_SR_CONFIANZA = 0.35*r  # poco soporte radial

        if diff > MAX_DIFF_MASA and mejor_sr < MIN_SR_CONFIANZA:

            tip_masa = punta_por_vector_masa(mask, center, r)
            if tip_masa is not None:
                mejor_punta = tip_masa

    # ---------- dibujo final ----------
    cv2.circle(canvas, mejor_punta, 10, (0, 255, 0), 2)
    cv2.putText(canvas, "PUNTA",
                (mejor_punta[0] + 8, mejor_punta[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # dibujar dirección de masa para debug
    if ang_masa is not None:
        ang_rad = math.radians(ang_masa)
        x_m = int(round(cx + r * math.cos(ang_rad)))
        y_m = int(round(cy - r * math.sin(ang_rad)))
        cv2.line(canvas, (cx, cy), (x_m, y_m), (0, 0, 255), 2)
        cv2.putText(canvas, "MASA",
                    (x_m + 5, y_m + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        dbg_full = os.path.join(debug_dir, f"debug_{tag}.png")
        cv2.imwrite(dbg_full, canvas)

        x0, y0 = max(0, cx - int(r)), max(0, cy - int(r))
        x1, y1 = min(canvas.shape[1], cx + int(r)), min(canvas.shape[0], cy + int(r))
        roi = canvas[y0:y1, x0:x1]
        dbg_roi = os.path.join(debug_dir, f"debug_{tag}_roi.png")
        cv2.imwrite(dbg_roi, roi)


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

    return delta, valor

def determineNeedle(img, cfg_path="config.json", prev_value=None):
    output_dir = "Agujas_resultados/"
    os.makedirs(output_dir, exist_ok=True)


    cfg_path = normalizar_ruta(cfg_path)
    if not os.path.exists(cfg_path):
        sys.exit(1)

    center, r, ang0, ang1, total_nums, sentido, hsv_ranges = cargar_config(cfg_path)
    print(f"Config: centro={center}, r={r}, ang0={ang0:.2f}, ang1={ang1:.2f}, sentido={sentido}")


    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise Exception(f"No se pudo leer la imagen desde ruta: {img}")
    else:
        img = np.array(img)

    # --- preprocesamiento base ---
    hsv, img_mejorada = preprocesar(img)
    hsv = enfatizar_rojo_hsv(
        hsv,
        hsv_ranges,
        sat_factor=0.8,
        val_factor=0.8
    )

    # --- máscaras del dial ---
    roi_mask, ring_vis = generar_masks(hsv.shape, center, r)

    # --- máscara de aguja + bordes ---
    mask, edges = bordes_y_umbral(
        hsv,
        roi_mask,
        hsv_ranges,
        center,
        r
    )

    # zona recortada para debug
    h_img, w_img = img.shape[:2]
    x_min_roi = max(0, center[0] - r)
    y_min_roi = max(0, center[1] - r)
    x_max_roi = min(w_img, center[0] + r)
    y_max_roi = min(h_img, center[1] + r)

    mask_rec = mask[y_min_roi:y_max_roi, x_min_roi:x_max_roi]
    cv2.imwrite(os.path.join(output_dir, "debug_mask.png"), mask_rec)

    edges_rec = edges[y_min_roi:y_max_roi, x_min_roi:x_max_roi]

    ring_rec = ring_vis[y_min_roi:y_max_roi, x_min_roi:x_max_roi]

    candidatos = []

    # ---------- 1) VÉRTICES (con fallback masa adentro) ----------
    tip_v = punta_por_vertices(
        mask,
        center,
        r,
        img_debug=img_mejorada.copy(),
        debug_dir=output_dir,
        tag="vertices_debug"
    )
    if tip_v is not None:
        ang_v = angulo_grados(center, tip_v)
        _, val_v = angle_to_value(ang_v, ang0, ang1, total_nums, sentido=sentido)
        candidatos.append(("vertices", tip_v, ang_v, val_v))
  
    else:
        print("")

    if not candidatos:
        valor_fallback = prev_value if prev_value is not None else 0.0
        return valor_fallback

    # ---------- Distancia circular 0..10 ----------
    def circ_dist(v, ref):
        d = abs(v - ref) % 10.0
        if d > 5.0:
            d = 10.0 - d
        return d

    # ------------------------------------------------------------------
    # CASO A: SIN prev_value -> prioridad fija vertices > bordes > contorno
    # ------------------------------------------------------------------
    if prev_value is None:
        metodo_final = None
        tip_final = None
        ang_final = None
        valor_final = None

        for m_prioritario in ("vertices", "bordes", "contorno"):
            for (m, tip, ang, val) in candidatos:
                if m == m_prioritario:
                    metodo_final = m
                    tip_final, ang_final, valor_final = tip, ang, val
                    break
            if metodo_final is not None:
                break

        path_salida = os.path.join(output_dir, "salida_reloj.png")
        dibujar_resultado(img, center, r, tip_final, ang_final, valor_final, metodo_final,
                          out_path=path_salida)
        valor_out = float(valor_final) % 10.0
        return valor_out

    # ------------------------------------------------------------------
    # CASO B: CON prev_value -> vertices manda, fallback muy restringido
    # ------------------------------------------------------------------
    prev = float(prev_value)

    MAX_DELTA_VERT = 9.0    
    MAX_DELTA_OTROS = 2.5   

    cand_dict = {}
    for (m, tip, ang, val) in candidatos:
        cand_dict[m] = (tip, ang, val)


    metodo_final = None
    tip_final = None
    ang_final = None
    valor_final = prev  # por defecto, mantenemos prev

    # --- 1) Intentar con vertices ---
    if "vertices" in cand_dict:
        tip_v, ang_v, val_v = cand_dict["vertices"]
        dv = circ_dist(val_v, prev)

        if dv <= MAX_DELTA_VERT:
            metodo_final = "vertices"
            tip_final, ang_final, valor_final = tip_v, ang_v, val_v

        else:

            pass

    if metodo_final is None:
        mejor_met = None
        mejor_tip = None
        mejor_ang = None
        mejor_val = None
        mejor_delta = None

        for met in ("contorno", "bordes"):
            if met not in cand_dict:
                continue
            tip_m, ang_m, val_m = cand_dict[met]
            d = circ_dist(val_m, prev)

            if mejor_delta is None or d < mejor_delta:
                mejor_delta = d
                mejor_met = met
                mejor_tip, mejor_ang, mejor_val = tip_m, ang_m, val_m

        if mejor_met is not None and mejor_delta is not None:
            if mejor_delta <= MAX_DELTA_OTROS:
                metodo_final = mejor_met
                tip_final, ang_final, valor_final = mejor_tip, mejor_ang, mejor_val
            else:
                pass
        else:
            pass

    # --- Dibujar resultado final ---
    path_salida = os.path.join(output_dir, "salida_reloj.png")
    if metodo_final is not None:
        dibujar_resultado(img, center, r, tip_final, ang_final, valor_final, metodo_final,
                          out_path=path_salida)
    else:
        print("[SALIDA] Se mantiene valor previo.")

    valor_final = float(valor_final) % 10.0


    return valor_final



if __name__ == "__main__":
    import shutil
    try:
        shutil.rmtree("agujas_tracking/resultados")
    except FileNotFoundError:
        pass
    img_arg = sys.argv[1] if len(sys.argv) > 1 else "3.jpg"
    cfg_arg = sys.argv[2] if len(sys.argv) > 2 else "config.json"
    v = determineNeedle(img_arg, cfg_arg)
    print(f"VALOR FINAL: {v}")
