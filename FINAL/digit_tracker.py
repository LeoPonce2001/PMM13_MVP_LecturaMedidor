# digit_tracker.py
import json
import numpy as np
from datetime import datetime
import cv2


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


def cv2_resize(gray, size):
    return cv2.resize(gray, size, interpolation=cv2.INTER_LINEAR)


def ahash_gray(img_np, size=8):
    if img_np.ndim == 3:
        if img_np.shape[2] == 3:
            g = (0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]).astype(np.uint8)
        else:
            g = img_np[:, :, 0].astype(np.uint8)
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


def clamp_no_zero_cross(prev_phase_0_10, new_phase_0_10, ocr_step, stick_hi=8.0, stick_lo=2.0):
    prevp = float(prev_phase_0_10)
    newp = float(new_phase_0_10) % 10.0

    if ocr_step == 0:
        if prevp < stick_lo and newp > stick_hi:
            return prevp

    return newp


class DigitTracker:
    def __init__(self, opts, last_state):
        self.opts = opts
        self.digits = last_state["digits"]
        self.stable = last_state.get("stable_counts", {k: 0 for k in self.digits})

        tr = opts.get("tracking", {})
        self.conf_min = float(tr.get("conf_min", 0.90))
        self.fast_accept_conf = float(tr.get("fast_accept_conf", 0.995))
        self.max_jump = int(tr.get("max_digit_jump", 2))
        self.threshold = int(tr.get("stable_count_threshold", 3))
        self.resync_need = tr.get("resync_frames", 5)
        self.resync_conf = tr.get("resync_conf_min", self.conf_min)

        self.last_pred = None
        self.last_pred_count = 0
        self.prev_digit_raw = {}

        # Candado para "acabamos de incrementar el dígito menos significativo"
        self.just_incremented = last_state.get(
            "just_incremented",
            {k: False for k in self.digits}
        )

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

            print("RESYNC for digit", name, "prev", prev_val, "new", self.digits[name], "frames", self.last_pred_count)

            self.last_pred = None
            self.last_pred_count = 0
            self.stable[name] = 0
            # si resync, eliminamos candado
            self.just_incremented[name] = False
            return True

        return False

    def _step_increment(self, name):
        keys = ordered_digit_keys(self.digits)
        last_key = keys[-1]

        if self.digits[last_key] == 9:
            # print("ROLLOVER cascade")

            for i in reversed(range(len(keys))):
                k = keys[i]

                if self.digits[k] == 9:
                    print("digit", k, "9 to 0")
                    self.digits[k] = 0
                else:
                    print("digit", k, "increment", self.digits[k], "to", self.digits[k] + 1)
                    self.digits[k] += 1
                    break
        else:
            prev = self.digits[last_key]
            self.digits[last_key] = prev + 1
            print("simple increment on", last_key, "from", prev, "to", self.digits[last_key])

        # marcamos que acabamos de incrementar el dígito de menor peso
        self.just_incremented[last_key] = True

    def update_digit(self, name, prediction, conf, base_amount_last, value_before):
        prev_val = int(self.digits[name])

        # registro para resync
        self._note_pred_for_resync(prediction, conf)

        prev_prev_val = self.prev_digit_raw.get(name, prev_val)
        self.prev_digit_raw[name] = prev_val

        if prediction is None:
            self.stable[name] = 0
            return

        if isinstance(prediction, np.generic):
            prediction = int(prediction)

        prediction = int(prediction)

        # manejo especial 9 -> (0,1,2)
        if prev_val == 9 and prediction in (0, 1, 2):
            self._step_increment(name)
            self.stable[name] = 0
            return

        delta_forward = (prediction - prev_val) % 10
        going_backward = (prediction < prev_val and not (prev_val == 9 and prediction == 0))
        expected_next = (prev_val + 1) % 10

        if prev_prev_val == 9 and prev_val == 0 and prediction == 1:
            self.stable[name] = 0
            return

        if self.just_incremented.get(name, False):

            if (prev_prev_val == 9 and prev_val == 0 and
                    prediction == 1 and delta_forward == 1):
                print(
                    "[JUST-INCREMENT-GUARD] Ignorando 0->1 inmediato tras rollover 9->0 "
                    f"(prev_prev={prev_prev_val}, prev={prev_val}, pred={prediction}, conf={conf})"
                )

                self.stable[name] = 0
                return

            if delta_forward == 0:
                # mismo valor que el dígito actual -> contamos estabilidad
                self.stable[name] = min(self.stable.get(name, 0) + 1, self.threshold)
                print("digit", name, "stable after increment, stable_count", self.stable[name])
                if self.stable[name] >= self.threshold:
                    # ya se estabilizó, levantamos el candado
                    self.just_incremented[name] = False
                return
            else:
                # cualquier otro cambio libera el candado; desde aquí el MODELO MANDA
                print(
                    "[JUST-INCREMENT] Liberando candado post-incremento para",
                    name, "delta_forward =", delta_forward,
                    "pred =", prediction, "prev =", prev_val
                )
                self.just_incremented[name] = False

        # si hay salto grande y raro (no es el siguiente número), lo rechazamos
        if delta_forward > 1 and prediction != expected_next:
            print("Rejected jump:", prediction, "expected", expected_next)
            self.stable[name] = 0
            return

        if prev_prev_val == 9 and prev_val == 0:
            self.digits[name] = 1
            self.stable[name] = 0
            return
        if prev_prev_val == 0 and prev_val == 1:
            self.digits[name] = 2
            self.stable[name] = 0
            return

        if prev_prev_val == 1 and prev_val == 2:
            self.digits[name] = 3
            self.stable[name] = 0
            return
        if prev_prev_val == 2 and prev_val == 3:
            self.digits[name] = 4
            self.stable[name] = 0
            return

        if prev_prev_val == 3 and prev_val == 4:
            self.digits[name] = 5
            self.stable[name] = 0
            return
        if prev_prev_val == 4 and prev_val == 5:
            self.digits[name] = 6
            self.stable[name] = 0
            return

        if prev_prev_val == 5 and prev_val == 6:
            self.digits[name] = 7
            self.stable[name] = 0
            return
        if prev_prev_val == 6 and prev_val ==7:
            self.digits[name] = 8
            self.stable[name] = 0
            return
        if prev_prev_val == 7 and prev_val == 8:
            self.digits[name] = 9
            self.stable[name] = 0
            return
        if prev_prev_val == 8 and prev_val == 9:
            self.digits[name] = 0
            self.stable[name] = 0
            return

        print("digit", name, "prev", prev_val, "pred", prediction, "delta", delta_forward, "conf", conf)

        if going_backward:
            if not self._try_resync(name):
                print("Rechazo el cambio hacia atras")
                self.stable[name] = 0
            return

        # mismo dígito -> solo estabilizamos
        if delta_forward == 0:
            self.stable[name] = min(self.stable.get(name, 0) + 1, self.threshold)
            print("digit", name, "stable")
            return

        if delta_forward == 1:
            print("simple increment on", name, "from", prev_val, "to", prediction, "conf", conf)
            self._step_increment(name)
            self.stable[name] = 0
            return

        # desde aquí, delta_forward >= 2 (saltos más grandes)

        if conf < self.conf_min:
            if not self._try_resync(name):
                self.stable[name] = 0
            return
        try:
            last_ts_str = self.opts.get("last_timestamp", None)
            if last_ts_str:
                last_ts = datetime.fromisoformat(last_ts_str)
                minutes_passed = (datetime.now() - last_ts).total_seconds() / 60.0
            else:
                minutes_passed = 0
        except:
            minutes_passed = 0

        if delta_forward > self.max_jump:
            if minutes_passed > 10 or conf >= 0.95:
                for _ in range(delta_forward):
                    self._step_increment(name)
                self.stable[name] = 0
                return
            else:
                if not self._try_resync(name):
                    self.stable[name] = 0
                return
        for _ in range(delta_forward):
            self._step_increment(name)

        self.stable[name] = 0



    def save_state(self, path, extra_state=None):
        st = {
            "digits": self.digits,
            "stable_counts": self.stable,
            "last_timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "just_incremented": self.just_incremented,
        }

        if isinstance(extra_state, dict):
            st.update(extra_state)

        with open(path, "w") as f:
            json.dump(st, f, indent=2)
