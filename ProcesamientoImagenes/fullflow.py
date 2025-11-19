#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

from digit_ocr import prepareImage
from tflitetest import predictDigit
from tracking_agujas import determineNeedle

from PIL import Image
import json
import shutil
from datetime import datetime
import requests
from io import BytesIO
import time
import numpy as np
import random
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

import envio

# ---------------- utilidades ----------------
def ordered_digit_keys(digits_dict):
    return sorted(digits_dict.keys(), key=lambda k: int(''.join(ch for ch in k if ch.isdigit()) or 0))

def _display_value_from_digits(digits_map, base_amount_last):
    keys = ordered_digit_keys(digits_map)
    n = len(keys)
    acc = 0.0
    for i, k in enumerate(keys):
        power = (n - 1 - i)
        amount_i = base_amount_last * (10 ** power)
        acc += amount_i * digits_map[k]
    return acc

# ---- aHash simple para detectar escena estática en ROI del dial (debug) ----
def ahash_gray(img_np, size=8):
    if img_np.ndim == 3:
        if img_np.shape[2] == 3:
            g = (0.299*img_np[:,:,0] + 0.587*img_np[:,:,1] + 0.114*img_np[:,:,2]).astype(np.uint8)
        else:
            g = img_np[:,:,0].astype(np.uint8)
    else:
        g = img_np.astype(np.uint8)
    g_small = cv2_resize(g, (size, size))
    mean = g_small.mean()
    bits = (g_small > mean).astype(np.uint8)
    h = 0
    for b in bits.flatten():
        h = (h << 1) | int(b)
    return int(h)

def hamming_dist64(a, b):
    return bin((a ^ b) & ((1 << 64) - 1)).count("1")

def cv2_resize(gray, size):
    import cv2
    return cv2.resize(gray, size, interpolation=cv2.INTER_LINEAR)

# ----- helper visual del dial (no modifica dígitos) -----
def clamp_no_zero_cross(prev_phase_0_10, new_phase_0_10, ocr_step, stick_hi=8.0, stick_lo=2.0):
    prevp = float(prev_phase_0_10)
    newp  = float(new_phase_0_10) % 10.0
    if ocr_step == 0:
        if prevp > stick_hi and newp < stick_lo:
            return prevp
        if prevp < stick_lo and newp > stick_hi:
            return prevp
    return newp

# -------------- tracker de dígitos (OCR manda) --------------
class DigitTracker:
    """
    Reglas:
    - OCR manda.
    - Si prev==9 y OCR==0 => rollover con carry SIEMPRE.
    - Nunca decrecer la lectura total.
    - Máximo salto hacia adelante configurable (excepto 9->0 que es especial).
    - Resync opcional si OCR repite el mismo valor N frames con buena confianza.
    """
    def __init__(self, opts, last_state):
        self.opts = opts
        self.digits = last_state["digits"]
        self.stable = last_state.get("stable_counts", {k: 0 for k in self.digits})

        tr = opts.get("tracking", {})
        self.conf_min = float(tr.get("conf_min", 0.90))
        self.fast_accept_conf = float(tr.get("fast_accept_conf", 0.995))
        self.max_jump = int(tr.get("max_digit_jump", 2))  # p.ej. 2 (configurable)
        self.threshold = int(tr.get("stable_count_threshold", 3))
        self.resync_need = tr.get("resync_frames", 5)
        self.resync_conf = tr.get("resync_conf_min", self.conf_min)

        self.last_pred = None
        self.last_pred_count = 0
        self.prev_digit_raw = {}

    def _note_pred_for_resync(self, prediction, conf):
        if prediction is None or conf < self.resync_conf:
            self.last_pred = None
            self.last_pred_count = 0
            return
        if isinstance(prediction, np.generic):
            prediction = int(prediction)
        if self.last_pred == prediction:
            self.last_pred_count += 1
        else:
            self.last_pred = prediction
            self.last_pred_count = 1

    def _try_resync(self, name):
        if self.last_pred is None:
            return False
        if self.last_pred_count >= self.resync_need:
            prev_val = self.digits[name]
            self.digits[name] = int(self.last_pred)
            print(f"[RESYNC] OCR estable: {prev_val} -> {self.digits[name]} (frames={self.last_pred_count})")
            self.last_pred = None
            self.last_pred_count = 0
            self.stable[name] = 0
            return True
        return False

    def _step_increment(self, name):
        keys = ordered_digit_keys(self.digits)
        last_key = keys[-1]
        if self.digits[last_key] == 9:
            print("   ROLLOVER: propagando de derecha a izquierda")
            for i in reversed(range(len(keys))):
                k = keys[i]
                if self.digits[k] == 9:
                    print(f"      {k}: 9 - 0")
                    self.digits[k] = 0
                else:
                    print(f"      {k}: {self.digits[k]} -> {self.digits[k] + 1}")
                    self.digits[k] += 1
                    break
        else:
            prev = self.digits[last_key]
            self.digits[last_key] = prev + 1
            print(f"   INCR simple: {last_key}: {prev} -> {self.digits[last_key]}")

    def update_digit(self, name, prediction, conf, base_amount_last, value_before):
        prev_val = int(self.digits[name])
        self._note_pred_for_resync(prediction, conf)
        prev_prev_val = self.prev_digit_raw.get(name, prev_val)
        self.prev_digit_raw[name] = prev_val
        if prediction is None:
            self.stable[name] = 0
            return

        if isinstance(prediction, np.generic):
            prediction = int(prediction)
        prediction = int(prediction)

        # ---- CASO ESPECIAL: OCR 9 -> (0–2) => rollover con carry ----
        if prev_val == 9 and prediction in (0, 1, 2):
            self._step_increment(name)
            self.stable[name] = 0
            return

        # ---- Reglas normales ----
        delta_forward = (prediction - prev_val) % 10
        going_backward = (prediction < prev_val and not (prev_val == 9 and prediction == 0))
        expected_next = (prev_val + 1) % 10

        # Si no es el incremento natural y delta_forward > 1,
        # entonces no se acepta. Hay que esperar al next.
        if delta_forward > 1 and prediction != expected_next:
            print(f"   RECHAZADO: solo se acepta avance natural {expected_next}, se recibió {prediction}")
            self.stable[name] = 0
            return
        if prev_prev_val == 5 and prev_val == 7 and prediction == 6:
            print("[EXCEPCION] Corrección especial 5 → 7 → 6 permitida.")
            self.digits[name] = 6
            self.stable[name] = 0
            return
        print(
            f"[{name}] Prev={prev_val} | Pred={prediction} | delta+={delta_forward} | "
            f"Conf={conf:.3f} | Backward={going_backward}"
        )

        if going_backward:
            if not self._try_resync(name):
                print("RECHAZADO: retroceso no permitido")
                self.stable[name] = 0
            return

        if delta_forward == 0:
            self.stable[name] = min(self.stable.get(name, 0) + 1, self.threshold)
            print("MANTENER: sin cambio")
            return

        # Aceptación rápida (delta=1 y alta confianza)
        if delta_forward == 1 and conf >= self.fast_accept_conf:
            self._step_increment(name)
            self.stable[name] = 0
            return

        # Confianza mínima
        if conf < self.conf_min:
            if not self._try_resync(name):
                print(f"   RECHAZADO: conf {conf:.3f} < {self.conf_min}")
                self.stable[name] = 0
            return

        # ---- Salto grande o tiempo prolongado sin lecturas ----
        from datetime import datetime

        try:
            last_ts_str = self.opts.get("last_timestamp", None)
            if last_ts_str:
                last_ts = datetime.fromisoformat(last_ts_str)
                minutes_passed = (datetime.now() - last_ts).total_seconds() / 60.0
            else:
                minutes_passed = 0
        except Exception:
            minutes_passed = 0

        if delta_forward > self.max_jump:
            if minutes_passed > 10 or conf >= 0.98:
                print(f"[INFO] Salto grande permitido: delta={delta_forward}, mins={minutes_passed:.1f}, conf={conf:.3f}")
                for _ in range(delta_forward):
                    self._step_increment(name)
                self.stable[name] = 0
                return
            else:
                if not self._try_resync(name):
                    print(f"   RECHAZADO: salto {delta_forward} > max_digit_jump={self.max_jump}")
                    self.stable[name] = 0
                return

        # ---- Aceptación normal de salto válido ----
        for _ in range(delta_forward):
            self._step_increment(name)
        self.stable[name] = 0

    def save_state(self, path, extra_state=None):
        st = {
            "digits": self.digits,
            "stable_counts": self.stable,
            "last_timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        }
        if isinstance(extra_state, dict):
            st.update(extra_state)
        with open(path, "w") as f:
            json.dump(st, f, indent=2)

# -------------- camera helpers --------------
def make_session(total_retries=2, backoff_factor=0.1, pool_size=10):
    s = requests.Session()
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
        raise_on_redirect=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool_size, pool_maxsize=pool_size)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

def fetch_frame(session, url, max_attempts=5, timeout_s=5.0, base_sleep=0.5):
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, timeout=timeout_s)
            if resp.status_code == 200:
                return Image.open(BytesIO(resp.content)).convert("RGB")
            else:
                print(f"[CAM] HTTP {resp.status_code} en intento {attempt}/{max_attempts}")
        except requests.exceptions.RequestException as e:
            print(f"[CAM] Error intento {attempt}/{max_attempts}: {e}")
        sleep_s = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, base_sleep)
        time.sleep(min(sleep_s, 5.0))
    print("[CAM] No se pudo obtener frame tras reintentos")
    return None

# -------------- medición por frame --------------
def getMeasurement(sourceImage):
    with open("config.json") as cfgfile:
        opts = json.load(cfgfile)

    with open("lastdigits.json") as dfile:
        last_state = json.load(dfile)

    tracker = DigitTracker(opts, last_state)

    # Solo inferimos el último dígito configurado
    last_digit_cfg = opts["digits"][-1]
    name = last_digit_cfg["name"]
    points = last_digit_cfg["points"]
    base_amount_last = float(last_digit_cfg["amount"])

    img_crop = prepareImage(sourceImage, points)
    prediction, outvec = predictDigit(img_crop)
    conf = float(np.max(outvec)) if (outvec is not None and len(outvec) > 0) else 0.0

    if opts.get("debug", False):
        os.makedirs("cutouts", exist_ok=True)
        img_crop.save(f"cutouts/{name}.jpg")

    # valor antes (para “nunca decrecer”)
    digits_before = last_state["digits"].copy()
    val_before = _display_value_from_digits(digits_before, base_amount_last)

    # ======= ACTUALIZAR DÍGITO (OCR manda; rollover 9->0 forzado) =======
    tracker.update_digit(name, prediction, conf, base_amount_last, val_before)

    # ======= DIAL SOLO PARA FRACCIÓN (NO mueve dígitos) =======
    measurement = determineNeedle(sourceImage, cfg_path="config_dial.json")

    # Filtro visual de fase para no saltar al cruzar 0 si OCR no cambió
    last_digits_prev = last_state["digits"].copy()
    prev_last = int(last_digits_prev[name])
    curr_last = int(tracker.digits[name])
    ocr_step = (curr_last - prev_last) % 10

    with open("config_dial.json", "r", encoding="utf-8") as f:
        dial_cfg = json.load(f)
    cx, cy = dial_cfg["centro"]; r = int(dial_cfg["radio"])
    img_np = np.array(sourceImage); h_img, w_img = img_np.shape[:2]
    x0 = max(0, cx - r); y0 = max(0, cy - r)
    x1 = min(w_img, cx + r); y1 = min(h_img, cy + r)
    roi = img_np[y0:y1, x0:x1]
    prev_hash = last_state.get("dial_roi_ahash", None)
    curr_hash = ahash_gray(roi, size=8)
    is_static = False
    if prev_hash is not None:
        dist = hamming_dist64(int(prev_hash), int(curr_hash))
        is_static = (dist <= 4)
    else:
        dist = None

    prev_phase_shown = float(last_state.get("dial_phase_shown", float(measurement)))
    phase_raw = float(measurement) % 10.0
    phase_shown = clamp_no_zero_cross(prev_phase_shown, phase_raw, ocr_step, stick_hi=8.0, stick_lo=2.0)

    # Guardia extra: nunca decrecer el valor total por solo cambiar dígitos
    val_after_digits = _display_value_from_digits(tracker.digits, base_amount_last)
    if val_after_digits < val_before:
        print(f"[GUARD] Dígitos bajarían ({val_after_digits:.6f} < {val_before:.6f}) -> descartar cambio")
        tracker.digits = digits_before
        val_after_digits = val_before

    # ======= Lectura total =======
    keys_order = ordered_digit_keys(tracker.digits)
    n = len(keys_order)
    display_value = 0.0
    for i, k in enumerate(keys_order):
        power = (n - 1 - i)
        amount_i = base_amount_last * (10 ** power)
        display_value += amount_i * tracker.digits[k]

    add_dial = opts.get("tracking", {}).get("add_dial_to_display", True)
    dial_amount = float(opts.get("dial_amount", base_amount_last))
    dial_fraction = phase_shown
    dial_contrib_sum = (dial_fraction * dial_amount) if add_dial else 0.0

    final_m3 = display_value + dial_contrib_sum

    total_litros   = final_m3 * 1000.0
    digits_litros  = display_value * 1000.0
    dial_litros    = dial_contrib_sum * 1000.0

    escena_estatica = 1 if is_static else 0
    print(f"[DIAL] escena_estatica={escena_estatica} | val={phase_raw:.2f} | shown={phase_shown:.2f}")
    print(f"TOTAL={total_litros:.2f} L")

    # Guardar estado
    extra_state = {
        "dial_roi_ahash": int(curr_hash),
        "dial_phase_shown": float(phase_shown),
    }
    tracker.save_state("lastdigits.json", extra_state=extra_state)

    return {
        "fecha_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_litros": total_litros,
        "digits_litros": digits_litros,
        "dial_litros": dial_litros,
        "dial_value_0_10": phase_shown
    }

# -------------- loop principal --------------
debugDirs = ["cutouts", "debug_output"]

if __name__ == "__main__":
    try:
        with open("config.json") as cfgfile:
            cfg_main = json.load(cfgfile)
        cam_cfg = cfg_main.get("camera", {})
        url = cam_cfg.get("url", "http://192.168.1.177/capture")
        retry_max_attempts = int(cam_cfg.get("retry_max_attempts", 5))
        timeout_s = float(cam_cfg.get("timeout_s", 5.0))
        base_sleep = float(cam_cfg.get("base_sleep", 0.5))

        for d in debugDirs:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.mkdir(d)

        session = make_session(total_retries=2, backoff_factor=0.1, pool_size=10)

        i = 0
        forEveryN = 10
        lecturas = []

        connected = False
        while True:
            try:
                print("Request")

                img = fetch_frame(session, url, max_attempts=retry_max_attempts, timeout_s=timeout_s, base_sleep=base_sleep)
                if img is None:
                    continue

                start = time.time()

                data = getMeasurement(img)
                #print(f"TOTAL: {data['total_litros']:.2f} L  |  digits={data['digits_litros']:.2f} L  + dial={data['dial_litros']:.2f} L  (dial={data['dial_value_0_10']:.2f}/10)")
                #print(f"Took {(time.time() - start):.2f} seconds.")

                i += 1
                lecturas.append(data)

                if not connected:
                    print("Connected with result code 0")
                    connected = True

                if i == forEveryN:
                    print(f"Datos a publicar: {lecturas}")
                    envio.publish_json(lecturas)
                    i = 0
                    lecturas = []
                envio.client.loop()

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[ERROR ciclo]: {e}")
                time.sleep(0.5)

    finally:
        try:
            envio.client.disconnect()
        except Exception:
            pass