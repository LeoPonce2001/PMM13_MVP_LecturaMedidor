import json
import os
from datetime import datetime

import numpy as np
from PIL import Image

from digit_ocr import prepareImage
from tflitetest import predictDigit
from tracking_agujas import determineNeedle

from digit_tracker import (
    DigitTracker,
    ordered_digit_keys,
    _display_value_from_digits,
    ahash_gray,
    hamming_dist64,
)


def _increment_digits_by_one(tracker: DigitTracker):
    """
    Incrementa en +1 el dígito de menor peso (con cascade 9->0, etc.)
    y marca el candado just_incremented en ese dígito.
    """
    digits = tracker.digits
    keys_order = ordered_digit_keys(digits)
    if not keys_order:
        return

    carry = 1
    for k in reversed(keys_order):
        if carry == 0:
            break
        v = int(digits[k])
        new_v = v + carry
        if new_v >= 10:
            digits[k] = 0
            carry = 1
        else:
            digits[k] = new_v
            carry = 0

    # candado: marcamos que acabamos de incrementar el último dígito
    last_key = keys_order[-1]
    if hasattr(tracker, "just_incremented"):
        tracker.just_incremented[last_key] = True


def guard_rollover_with_dial(prev_digit: int, new_digit: int, dial_val: float) -> bool:
    if dial_val is None:
        return False

    # Guard sólo para 9 -> 0 del último dígito
    if prev_digit == 9 and new_digit == 0:
        if dial_val > 2.0:  # si el dial no está cerca de 0, sospechoso
            return True

    return False


def dial_filter(prev: float, raw: float, ocr_step: int, escena_estatica: int, wrap_pending: int):
    """
    Filtro de la aguja con detección de wrap alrededor de cero.
    Devuelve:
      - val: raw (para logs)
      - cand: valor filtrado que se usará como dial_shown
      - wrap_pending: contador actualizado
      - wrap_flag: 1 si se detectó wrap en este frame, 0 si no
    """
    MAX_DELTA    = 9.0
    TH_PREV_ALTO = 7.0
    TH_RAW_BAJO  = 6.0

    val = raw
    delta = raw - prev
    allow_wrap = 0

    wrap_event = (prev > TH_PREV_ALTO and 0.0 < raw < TH_RAW_BAJO)

    if wrap_event:
        cand = raw
        allow_wrap = 1
    else:
        if wrap_pending > 0:
            wrap_pending -= 1

        if abs(delta) > MAX_DELTA:
            cand = prev
        else:
            cand = raw


    return val, cand, wrap_pending, int(wrap_event)


def getMeasurement(sourceImage):
    # --------- CONFIG + ESTADO ANTERIOR ---------
    with open("config.json") as cfgfile:
        opts = json.load(cfgfile)
    tr_cfg = opts.get("tracking", {})
    dial_zero_hold_s = float(tr_cfg.get("dial_zero_hold_s", 2.0))

    with open("lastdigits.json") as dfile:
        last_state = json.load(dfile)

    prev_total_litros = last_state.get("total_litros", None)
    if prev_total_litros is not None:
        try:
            prev_total_litros = float(prev_total_litros)
        except Exception:
            prev_total_litros = None

    prev_digits = last_state["digits"].copy()
    wrap_pending = int(last_state.get("wrap_pending_frames", 0))

    zero_lock_active = bool(last_state.get("dial_zero_lock_active", False))
    zero_lock_until_str = last_state.get("dial_zero_lock_until", None)
    if zero_lock_until_str:
        try:
            zero_lock_until = datetime.fromisoformat(zero_lock_until_str)
        except Exception:
            zero_lock_until = None
    else:
        zero_lock_until = None

    tracker = DigitTracker(opts, last_state)

    # --------- OCR ÚLTIMO DÍGITO (MODELO) ---------
    last_digit_cfg = opts["digits"][-1]
    name = last_digit_cfg["name"]
    points = last_digit_cfg["points"]
    base_amount_last = float(last_digit_cfg["amount"])

    img_crop = prepareImage(sourceImage, points)
    prediction, outvec = predictDigit(img_crop)
    conf = float(np.max(outvec)) if (outvec is not None and len(outvec) > 0) else 0.0

    if opts.get("debug", False):
        import cv2
        os.makedirs("Digito_resultado", exist_ok=True)

        img_crop.save(f"Digito_resultado/{name}_prep.jpg")

        src_np = np.array(sourceImage)
        pts = np.array(points, dtype=np.int32)
        x_min = np.min(pts[:, 0])
        x_max = np.max(pts[:, 0])
        y_min = np.min(pts[:, 1])
        y_max = np.max(pts[:, 1])

        roi_color = src_np[y_min:y_max, x_min:x_max]
        roi_color_bgr = cv2.cvtColor(roi_color, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"Digito_resultado/{name}_color.jpg", roi_color_bgr)

        full_dbg = src_np.copy()
        cv2.polylines(full_dbg, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        full_dbg_bgr = cv2.cvtColor(full_dbg, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"Digito_resultado/{name}_fullboxed.jpg", full_dbg_bgr)


    digits_before = prev_digits
    val_before = _display_value_from_digits(digits_before, base_amount_last)

    tracker.update_digit(name, prediction, conf, base_amount_last, val_before)

    # --------- DIAL: PREV + RAW (AGUJA) ---------
    prev_phase_shown = float(last_state.get("dial_phase_shown", 0.0))
    if prev_phase_shown < 0.0 or prev_phase_shown >= 10.0:
        old_prev = prev_phase_shown
        prev_phase_shown = prev_phase_shown % 10.0
        #print(
        #    f"[WARN] dial_phase_shown fuera de rango ({old_prev:.2f}), "
        #    f"normalizado a {prev_phase_shown:.2f}"
        #)

    phase_raw = float(
        determineNeedle(sourceImage, cfg_path="config.json", prev_value=prev_phase_shown)
    ) % 10.0

    with open("config.json", "r", encoding="utf-8") as f:
        dial_cfg = json.load(f)

    cx, cy = dial_cfg["centro"]
    r = int(dial_cfg["radio"])

    img_np = np.array(sourceImage)
    h_img, w_img = img_np.shape[:2]
    x0 = max(0, cx - r)
    y0 = max(0, cy - r)
    x1 = min(w_img, cx + r)
    y1 = min(h_img, cy + r)
    roi = img_np[y0:y1, x0:x1]

    DATASET_DIR = "dial_resultados_para_entrenar"
    MAX_SAMPLES = 100
    os.makedirs(DATASET_DIR, exist_ok=True)
    counter_path = os.path.join(DATASET_DIR, "counter.txt")

    try:
        with open(counter_path, "r") as f:
            n_saved = int(f.read().strip() or "0")
    except FileNotFoundError:
        n_saved = 0

    if n_saved < MAX_SAMPLES:
        filename = os.path.join(DATASET_DIR, f"dial_{n_saved:04d}.png")
        Image.fromarray(roi).save(filename)
        n_saved += 1
        with open(counter_path, "w") as f:
            f.write(str(n_saved))

    prev_hash = last_state.get("dial_roi_ahash", None)
    curr_hash = ahash_gray(roi, size=8)

    if prev_hash is not None:
        dist = hamming_dist64(int(prev_hash), int(curr_hash))
        is_static = (dist <= 4)
    else:
        dist = None
        is_static = False

    escena_estatica = 1 if is_static else 0

    # --------- OCR_STEP (cambio de último dígito, MODELO MANDA) ---------
    keys_order_prev = ordered_digit_keys(prev_digits)
    if keys_order_prev:
        last_key_prev = keys_order_prev[-1]
        prev_last_digit = int(prev_digits[last_key_prev])
    else:
        prev_last_digit = 0

    keys_order_curr = ordered_digit_keys(tracker.digits)
    if keys_order_curr:
        last_key_curr = keys_order_curr[-1]
        curr_last_digit = int(tracker.digits[last_key_curr])
    else:
        curr_last_digit = prev_last_digit


    ocr_step = (curr_last_digit - prev_last_digit) % 10

    if ocr_step > 0:
        if ocr_step != 1:
            tracker.digits = prev_digits.copy()
            curr_last_digit = prev_last_digit
            ocr_step = 0
            wrap_pending = 0
        else:
            # paso normal N->N+1: se acepta tal cual
            wrap_pending = 0
            if prev_last_digit == 9 and curr_last_digit == 0 and dial_zero_hold_s > 0.0:
                from datetime import timedelta  
                zero_lock_active = True
                zero_lock_until = datetime.now() + timedelta(seconds=dial_zero_hold_s)

    phase_val, phase_shown, wrap_pending, wrap_flag = dial_filter(
        prev=prev_phase_shown,
        raw=phase_raw,
        ocr_step=ocr_step,
        escena_estatica=escena_estatica,
        wrap_pending=wrap_pending,
    )
    phase_shown = phase_shown % 10.0

    # --------- MODELO MANDA: si hubo N->N+1, dial se resetea a 0 ---------
    if ocr_step > 0:
        phase_shown = 0.0


    now = datetime.now()
    if zero_lock_active:
        if escena_estatica == 1:

            phase_shown = 0.0
        else:

            if (zero_lock_until is None) or (now <= zero_lock_until):
                phase_shown = 0.0
            else:
                zero_lock_active = False  # liberamos candado

    # ----------------- GUARDIA: dígitos nunca decrecen -----------------
    val_after_digits = _display_value_from_digits(tracker.digits, base_amount_last)

    if val_after_digits < val_before:

        tracker.digits = digits_before.copy()
        val_after_digits = val_before

    # ----------------- TOTAL m3 + LITROS (dígitos + dial) -----------------
    keys_order = ordered_digit_keys(tracker.digits)
    display_value = 0.0
    n = len(keys_order)
    for i, k in enumerate(keys_order):
        power = (n - 1 - i)
        amount_i = base_amount_last * (10 ** power)
        display_value += amount_i * tracker.digits[k]

    add_dial = opts.get("tracking", {}).get("add_dial_to_display", True)
    dial_amount = float(opts.get("dial_amount", base_amount_last))

    # fracción 0..10 del dial
    dial_fraction = phase_shown
    dial_contrib_sum = (dial_fraction * dial_amount) if add_dial else 0.0

    final_m3 = display_value + dial_contrib_sum

    total_litros = final_m3 * 1000.0
    digits_litros = display_value * 1000.0
    dial_litros = dial_contrib_sum * 1000.0

    # --- GUARDIA: total nunca decrece ---
    if prev_total_litros is not None:
        if total_litros + 1e-3 < prev_total_litros:

            total_litros = prev_total_litros

            if digits_litros > total_litros:
                digits_litros = total_litros

            dial_litros = total_litros - digits_litros
            if dial_litros < 0.0:
                dial_litros = 0.0

            dial_contrib_sum = dial_litros / 1000.0
            if dial_amount > 0:
                phase_shown = (dial_contrib_sum / dial_amount)
                phase_val = phase_shown

    print(
        f"[DIAL-RESUMEN] escena_estatica={escena_estatica}"
    )
    print(f"TOTAL={total_litros:.2f} L")

    extra_state = {
        "dial_roi_ahash": int(curr_hash),
        "dial_phase_shown": float(phase_shown),
        "wrap_pending_frames": int(wrap_pending),
        "total_litros": float(total_litros),

        "dial_zero_lock_active": bool(zero_lock_active),
        "dial_zero_lock_until": zero_lock_until.isoformat() if zero_lock_until is not None else None,
    }

    tracker.save_state("lastdigits.json", extra_state=extra_state)

    return {
        "fecha_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_litros": total_litros,
        "digits_litros": digits_litros,
        "dial_litros": dial_litros,
        "dial_value_0_10": phase_raw,
        "dial_value_0_10_shown": phase_shown,
        "escena_estatica": int(escena_estatica),
    }

