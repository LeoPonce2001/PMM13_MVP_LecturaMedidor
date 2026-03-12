#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import json
import time
from io import BytesIO
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import csv
import sys
import termios
import tty
import select

from measurement import getMeasurement
import envio


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
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=pool_size,
        pool_maxsize=pool_size
    )
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def capture_frame_with_fallback(session, cam_cfg, last_ok_img):
    url = cam_cfg.get("url", "http://192.168.1.177/capture")
    max_attempts_cfg = int(cam_cfg.get("retry_max_attempts", 5))
    timeout_cfg = float(cam_cfg.get("timeout_s", 5.0))
    base_sleep = float(cam_cfg.get("base_sleep", 0.5))

    max_attempts = min(max_attempts_cfg, 2)
    timeout_s = min(timeout_cfg, 2.0)

    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, timeout=timeout_s)

            if resp.status_code == 200:
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                print(f"[CAM] frame OK intento {attempt}/{max_attempts}")
                return img, True
            else:
                print(f"[CAM] HTTP {resp.status_code} en intento {attempt}/{max_attempts}")

        except requests.exceptions.RequestException as e:
            print(f"[CAM] Error intento {attempt}/{max_attempts}: {e}")

        if attempt < max_attempts:
            time.sleep(base_sleep)

    print("[CAM] sin nueva imagen, reutilizo última válida (si existe)")
    return last_ok_img, False


def append_to_csv(file_path, registros):
    if not registros:
        return

    file_exists = os.path.isfile(file_path)

    if file_exists:
        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        if header:
            fieldnames = header
        else:
            fieldnames = list(registros[0].keys())
    else:
        fieldnames = list(registros[0].keys())

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for row in registros:
            safe_row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(safe_row)


# ----------------- CSV DE EVENTOS DE PRUEBA -----------------

def append_event_to_csv(file_path, event_row):
    """
    CSV de eventos de prueba:
    Numero Evento,Fecha inicio,Total Inicio,Total Final,Fechas final,LitrosConsumidos
    """
    file_exists = os.path.isfile(file_path)
    fieldnames = [
        "Numero Evento",
        "Fecha inicio",
        "Total Inicio",
        "Total Final",
        "Fechas final",
        "LitrosConsumidos",
    ]

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(event_row)



def configurar_teclado_cbreak():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return old_settings


def restaurar_teclado(old_settings):
    fd = sys.stdin.fileno()
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def espacio_presionado():
    """
    Devuelve True si se presionó la barra espaciadora desde la última llamada.
    No bloquea el loop.
    """
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        ch = sys.stdin.read(1)
        return ch == " "
    return False


if __name__ == "__main__":
    old_kbd_settings = None
    try:
        with open("config.json") as cfgfile:
            cfg_main = json.load(cfgfile)

        cam_cfg = cfg_main.get("camera", {})
        PERIOD = float(cfg_main.get("loop_period_s", 0.5))
        publish_every = int(cfg_main.get("publish_every", 5))
        excel_path = cfg_main.get("excel_path", "Lecturas_Medidor.csv")

        # CSV de eventos (se puede configurar en config.json)
        eventos_path = cfg_main.get("eventos_excel_path", "Eventos_Pruebas.csv")

        session = make_session(total_retries=2, backoff_factor=0.1, pool_size=2)

        i = 0
        forEveryN = publish_every
        lecturas = []

        connected = False
        last_ok_img = None

        evento_id = 0
        evento_activo = False
        evento_vio_movimiento = False
        evento_total_inicio = None
        evento_fecha_inicio = None

        try:
            old_kbd_settings = configurar_teclado_cbreak()
            print("[INFO] Teclado listo: presiona ESPACIO para iniciar una prueba.")
        except Exception as e:
            print(f"[WARN] No se pudo configurar el teclado para leer ESPACIO: {e}")
            old_kbd_settings = None

        while True:
            loop_start = time.time()
            print("=" * 50)
            print("INICIANDO DETECCIONES")
            print("=" * 50)
            try:
                print("Request")

                img, fresh = capture_frame_with_fallback(session, cam_cfg, last_ok_img)

                if img is None:
                    print("[LOOP] no hay imagen válida (ni nueva ni previa), se omite medición")
                else:
                    if fresh:
                        last_ok_img = img

                    if fresh:
                        data = getMeasurement(img)

                        escena_estatica = data.get("escena_estatica", 0)
                        total_litros = data.get("total_litros", None)
                        fecha_hora = data.get("fecha_hora", "")
                        if old_kbd_settings is not None and espacio_presionado():
                            if not evento_activo:
                                if escena_estatica == 1 or escena_estatica == 1.0:
                                    evento_activo = True
                                    evento_vio_movimiento = False
                                    evento_id += 1
                                    evento_total_inicio = total_litros
                                    evento_fecha_inicio = fecha_hora

                                    print("el usuario dio inicio a prueba")
                                    print(
                                        f"[EVENTO #{evento_id}] INICIO: "
                                        f"fecha_inicio={evento_fecha_inicio}, "
                                        f"total_inicio={evento_total_inicio}"
                                    )
                                else:
                                    print(
                                        "[PRUEBA] ESPACIO presionado pero escena_estatica != 1; "
                                        "no se inicia prueba (espera escena estática)."
                                    )
                            else:
                                print(
                                    f"[PRUEBA] Ya hay una prueba activa (evento #{evento_id}), "
                                    "se ignora nueva presión de ESPACIO."
                                )

                        # Durante la prueba: esperamos movimiento y luego vuelta a estática
                        if evento_activo:
                            if escena_estatica == 0 or escena_estatica == 0.0:
                                evento_vio_movimiento = True

                            if evento_vio_movimiento and (escena_estatica == 1 or escena_estatica == 1.0):
                                total_fin = total_litros
                                fecha_fin = fecha_hora

                                litros_consumidos = ""
                                if (
                                    evento_total_inicio is not None
                                    and total_fin is not None
                                ):
                                    litros_consumidos = total_fin - evento_total_inicio

                                event_row = {
                                    "Numero Evento": evento_id,
                                    "Fecha inicio": evento_fecha_inicio,
                                    "Total Inicio": evento_total_inicio,
                                    "Total Final": total_fin,
                                    "Fechas final": fecha_fin,
                                    "LitrosConsumidos": litros_consumidos,
                                }
                                append_event_to_csv(eventos_path, event_row)

                                print("La prueba ha finalizado")
                                print(
                                    f"[EVENTO #{evento_id}] FIN: "
                                    f"fecha_fin={fecha_fin}, total_fin={total_fin}, "
                                    f"litros_consumidos={litros_consumidos}"
                                )

                                evento_activo = False
                                evento_vio_movimiento = False
                                evento_total_inicio = None
                                evento_fecha_inicio = None


                        i += 1
                        lecturas.append(data)

                        if not connected:
                            print("Connected with result code 0")
                            connected = True

                        if i == forEveryN:
                            print(f"Datos a publicar: {lecturas}")
                            envio.publish_json(lecturas)

                            append_to_csv(excel_path, lecturas)
                            print(f"[CSV] {len(lecturas)} filas agregadas a {excel_path}")

                            i = 0
                            lecturas = []
                    else:
                        print("[LOOP] ciclo sin nueva imagen, no actualizo medición")

                envio.client.loop()

            except KeyboardInterrupt:
                break

            except Exception as e:
                print(f"[ERROR ciclo]: {e}")
                time.sleep(0.5)

            elapsed = time.time() - loop_start
            if elapsed < PERIOD:
                time.sleep(PERIOD - elapsed)

    finally:
        if old_kbd_settings is not None:
            try:
                restaurar_teclado(old_kbd_settings)
                print("[INFO] Configuración de teclado restaurada.")
            except Exception as e:
                print(f"[WARN] No se pudo restaurar el teclado: {e}")

        try:
            envio.client.disconnect()
        except Exception:
            pass
