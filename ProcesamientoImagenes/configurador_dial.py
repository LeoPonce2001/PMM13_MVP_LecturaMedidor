import cv2
import numpy as np
import math
import json
import os
import sys

# --- Variables Globales ---
puntos_calibracion = []
imagen_original = None
imagen_display_geo = None
imagen_display_hsv = None
imagen_hsv_completa = None
imagen_hsv_aislada = None
imagen_aislada_bgr = None
nombre_imagen = '1.jpg' # Tu imagen
h_min1_val, h_max1_val = 0, 10
s_min_val, s_max_val = 120, 255
v_min_val, v_max_val = 70, 255
h_min2_val, h_max2_val = 170, 179
x_min_roi, y_min_roi = 0, 0

# --- Funciones de Ayuda ---

def calcular_angulo(centro, punto_cero):
    dx = punto_cero[0] - centro[0]
    dy = -(punto_cero[1] - centro[1])
    angulo_rad = math.atan2(dy, dx)
    angulo_grados = (np.degrees(angulo_rad) + 360) % 360
    return angulo_grados

def nada(x):
    pass

# --- ¡NUEVAS FUNCIONES DE MEJORA DE IMAGEN! ---

def gray_world(img_bgr):
    """Aplica balance de blancos 'Gray World'."""
    img = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    mb, mg, mr = b.mean(), g.mean(), r.mean()
    m = (mb + mg + mr) / 3.0
    b = b * (m / (mb + 1e-6))
    g = g * (m / (mg + 1e-6))
    r = r * (m / (mr + 1e-6))
    return np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)

def clahe_l(img_bgr, clip=2.0, tiles=(8,8)):
    """Aplica CLAHE al canal L en espacio LAB."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

# --- Callbacks del Mouse ---

def mouse_callback_geometria(event, x, y, flags, param):
    """Manejador de eventos del mouse para la Parte 1 (Geometría)."""
    global puntos_calibracion, imagen_display_geo
    
    imagen_display_geo = imagen_original.copy() 
    instruccion = ""
    centro = None
    radio = 0
    
    if len(puntos_calibracion) >= 1:
        centro = puntos_calibracion[0]
    if len(puntos_calibracion) >= 2:
        radio = int(math.dist(centro, puntos_calibracion[1]))

    # (Lógica de instrucciones MODIFICADA para 4 clics)
    if len(puntos_calibracion) == 0:
        instruccion = "PARTE 1: 1. Haz clic en el CENTRO del dial"
    elif len(puntos_calibracion) == 1:
        instruccion = "PARTE 1: 2. Haz clic en el BORDE (para el radio)"
        cv2.circle(imagen_display_geo, centro, 5, (0, 0, 255), -1)
    elif len(puntos_calibracion) == 2:
        instruccion = "PARTE 1: 3. Haz clic en la marca del '0' (CERO)"
        cv2.circle(imagen_display_geo, centro, 5, (0, 0, 255), -1)
        cv2.circle(imagen_display_geo, centro, radio, (0, 255, 0), 2)
    elif len(puntos_calibracion) == 3:
        instruccion = "PARTE 1: 4. Haz clic en la marca del '1' (UNO)"
        punto_cero = puntos_calibracion[2]
        cv2.circle(imagen_display_geo, centro, 5, (0, 0, 255), -1)
        cv2.circle(imagen_display_geo, centro, radio, (0, 255, 0), 2)
        cv2.circle(imagen_display_geo, punto_cero, 5, (255, 0, 0), -1)
        cv2.line(imagen_display_geo, centro, punto_cero, (255, 255, 0), 2)
    elif len(puntos_calibracion) == 4:
        instruccion = "PARTE 1: ¡Lista! Presiona 'c' para CONTINUAR a color"
        punto_cero = puntos_calibracion[2]
        punto_uno = puntos_calibracion[3]
        cv2.circle(imagen_display_geo, centro, 5, (0, 0, 255), -1)
        cv2.circle(imagen_display_geo, centro, radio, (0, 255, 0), 2)
        cv2.circle(imagen_display_geo, punto_cero, 5, (255, 0, 0), -1)
        cv2.circle(imagen_display_geo, punto_uno, 5, (255, 0, 255), -1) # Punto '1' en magenta
        cv2.line(imagen_display_geo, centro, punto_cero, (255, 255, 0), 2)
        cv2.line(imagen_display_geo, centro, punto_uno, (255, 255, 0), 2)

    cv2.putText(imagen_display_geo, instruccion, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # (Lógica de Clic MODIFICADA para 4 clics)
    if event == cv2.EVENT_LBUTTONDOWN and len(puntos_calibracion) < 4:
        puntos_calibracion.append((x, y))
        print(f"Punto {len(puntos_calibracion)} agregado: {(x, y)}")

def mouse_callback_hsv(event, x, y, flags, param):
    """Manejador de eventos del mouse para la Parte 2 (Selección HSV con clic)."""
    global h_min1_val, h_max1_val, s_min_val, s_max_val, v_min_val, v_max_val, h_min2_val, h_max2_val, imagen_display_hsv
    global x_min_roi, y_min_roi 

    if event == cv2.EVENT_LBUTTONDOWN:
        x_orig = x + x_min_roi
        y_orig = y + y_min_roi
        pixel_hsv = imagen_hsv_completa[y_orig, x_orig]
        h, s, v = pixel_hsv[0], pixel_hsv[1], pixel_hsv[2]
        print(f"Clic en (Recortada: {x},{y} -> Original: {x_orig},{y_orig}) | HSV: ({h}, {s}, {v})")

        tolerancia_h = 10 
        tolerancia_s = 50
        tolerancia_v = 50

        h_min1_val = max(0, h - tolerancia_h)
        h_max1_val = min(179, h + tolerancia_h)
        s_min_val = max(0, s - tolerancia_s)
        s_max_val = min(255, s + tolerancia_s)
        v_min_val = max(0, v - tolerancia_v)
        v_max_val = min(255, v + tolerancia_v)
        
        if h < 10 or h > 170: 
             if h < 10: 
                 h_min2_val = max(0, 180 - tolerancia_h)
                 h_max2_val = 179 
             else: 
                 h_min2_val = 0
                 h_max2_val = min(179, tolerancia_h)
        else: 
             h_min2_val = 0
             h_max2_val = 0
            
        cv2.setTrackbarPos('H Min 1', 'Panel de Control HSV', h_min1_val)
        cv2.setTrackbarPos('H Max 1', 'Panel de Control HSV', h_max1_val)
        cv2.setTrackbarPos('S Min', 'Panel de Control HSV', s_min_val)
        cv2.setTrackbarPos('S Max', 'Panel de Control HSV', s_max_val)
        cv2.setTrackbarPos('V Min', 'Panel de Control HSV', v_min_val)
        cv2.setTrackbarPos('V Max', 'Panel de Control HSV', v_max_val)
        cv2.setTrackbarPos('H Min 2', 'Panel de Control HSV', h_min2_val)
        cv2.setTrackbarPos('H Max 2', 'Panel de Control HSV', h_max2_val)

        imagen_display_hsv = imagen_aislada_bgr.copy()
        cv2.circle(imagen_display_hsv, (x, y), 5, (0, 255, 255), -1) 

# --- Bloque principal del Configurador ---
if __name__ == "__main__":
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_imagen = os.path.join(script_dir, nombre_imagen)
    
    # --- ¡MODIFICACIÓN! APLICAR PREPROCESAMIENTO ---
    img_raw = cv2.imread(ruta_imagen)
    if img_raw is None:
        print(f"Error: No se pudo cargar la imagen en '{ruta_imagen}'")
        sys.exit()
    
    print("Aplicando preprocesamiento (Gray World + CLAHE) a la imagen...")
    
    # 1. Aplicar balance de blancos
    img_wb = gray_world(img_raw)
    
    # 2. Aplicar mejora de contraste (usamos clip=2.5 como en tu script anterior)
    img_enhanced = clahe_l(img_wb, clip=2.5)
    
    # 3. La 'imagen_original' para el resto del script será la versión MEJORADA
    imagen_original = img_enhanced
    print("Preprocesamiento aplicado.")
    # --- FIN DE LA MODIFICACIÓN ---
        
    imagen_display_geo = imagen_original.copy()
    # Convertimos la imagen MEJORADA a HSV
    imagen_hsv_completa = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2HSV)
    
    # --- PARTE 1: CALIBRACIÓN DE GEOMETRÍA (CLICS) ---
    cv2.namedWindow("Configurador de Dial")
    cv2.setMouseCallback("Configurador de Dial", mouse_callback_geometria)
    
    print("--- PARTE 1: CALIBRACIÓN DE GEOMETRÍA ---")
    print("Sigue las instrucciones en la ventana (4 clics).")
    print("Presiona 'c' para continuar a la calibración de color.")
    print("Presiona 'q' para salir sin guardar.")

    while True:
        cv2.imshow("Configurador de Dial", imagen_display_geo)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            sys.exit()
        if key == ord('c') and len(puntos_calibracion) == 4:
            print("Parte 1 (Geometría) completada.")
            break
            
    cv2.destroyAllWindows()

    # --- Extraer datos de la Parte 1 ---
    centro = puntos_calibracion[0]
    punto_borde = puntos_calibracion[1]
    punto_cero = puntos_calibracion[2]
    punto_uno = puntos_calibracion[3]
    
    radio = int(math.dist(centro, punto_borde))
    angulo_cero = calcular_angulo(centro, punto_cero)
    angulo_uno = calcular_angulo(centro, punto_uno)

    # --- CALCULAR Y APLICAR RECORTE (ROI) ---
    print("Creando máscara y recortando el dial para la Parte 2...")
    h_img, w_img = imagen_original.shape[:2]
    x_min_roi = max(0, centro[0] - radio)
    y_min_roi = max(0, centro[1] - radio)
    x_max_roi = min(w_img, centro[0] + radio)
    y_max_roi = min(h_img, centro[1] + radio)
    mask_dial = np.zeros(imagen_original.shape[:2], dtype="uint8")
    cv2.circle(mask_dial, centro, radio, 255, -1)
    
    # imagen_original y imagen_hsv_completa ya están mejoradas
    imagen_aislada_bgr_full = cv2.bitwise_and(imagen_original, imagen_original, mask=mask_dial)
    imagen_hsv_aislada_full = cv2.bitwise_and(imagen_hsv_completa, imagen_hsv_completa, mask=mask_dial)
    
    imagen_aislada_bgr = imagen_aislada_bgr_full[y_min_roi:y_max_roi, x_min_roi:x_max_roi]
    imagen_hsv_aislada = imagen_hsv_aislada_full[y_min_roi:y_max_roi, x_min_roi:x_max_roi]
    print("Recorte aplicado. Iniciando Parte 2.")
    # -----------------------------------------------------------------

    # --- PARTE 2: CALIBRACIÓN DE COLOR (SOBRE IMAGEN RECORTADA) ---
    print("\n--- PARTE 2: CALIBRACIÓN DE COLOR HSV ---")
    print("1. HAZ CLIC en la aguja en la ventana 'Click para Color (Solo Dial)'.")
    print("2. AFINA los deslizadores en la ventana 'Panel de Control HSV'.")
    print("3. Presiona 'g' para GUARDAR.")
    print("4. Presiona 'q' para salir.")

    cv2.namedWindow("Panel de Control HSV")
    cv2.resizeWindow("Panel de Control HSV", 600, 400) 
    cv2.namedWindow("Click para Color (Solo Dial)")
    cv2.namedWindow("Mascara (Solo Dial)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Resultado Filtrado (Solo Dial)", cv2.WINDOW_NORMAL) 

    cv2.createTrackbar('H Min 1', 'Panel de Control HSV', h_min1_val, 179, nada)
    cv2.createTrackbar('H Max 1', 'Panel de Control HSV', h_max1_val, 179, nada)
    cv2.createTrackbar('S Min', 'Panel de Control HSV', s_min_val, 255, nada)
    cv2.createTrackbar('S Max', 'Panel de Control HSV', s_max_val, 255, nada)
    cv2.createTrackbar('V Min', 'Panel de Control HSV', v_min_val, 255, nada)
    cv2.createTrackbar('V Max', 'Panel de Control HSV', v_max_val, 255, nada)
    cv2.createTrackbar('H Min 2', 'Panel de Control HSV', h_min2_val, 179, nada)
    cv2.createTrackbar('H Max 2', 'Panel de Control HSV', h_max2_val, 179, nada)

    imagen_display_hsv = imagen_aislada_bgr.copy()
    cv2.setMouseCallback("Click para Color (Solo Dial)", mouse_callback_hsv) 

    while True:
        current_h_min1 = cv2.getTrackbarPos('H Min 1', 'Panel de Control HSV')
        current_h_max1 = cv2.getTrackbarPos('H Max 1', 'Panel de Control HSV')
        current_s_min = cv2.getTrackbarPos('S Min', 'Panel de Control HSV')
        current_s_max = cv2.getTrackbarPos('S Max', 'Panel de Control HSV')
        current_v_min = cv2.getTrackbarPos('V Min', 'Panel de Control HSV')
        current_v_max = cv2.getTrackbarPos('V Max', 'Panel de Control HSV')
        current_h_min2 = cv2.getTrackbarPos('H Min 2', 'Panel de Control HSV')
        current_h_max2 = cv2.getTrackbarPos('H Max 2', 'Panel de Control HSV')

        lower1 = np.array([current_h_min1, current_s_min, current_v_min])
        upper1 = np.array([current_h_max1, current_s_max, current_v_max])
        lower2 = np.array([current_h_min2, current_s_min, current_v_min]) 
        upper2 = np.array([current_h_max2, current_s_max, current_v_max])
        
        # imagen_hsv_aislada ya proviene de la imagen mejorada
        mask1 = cv2.inRange(imagen_hsv_aislada, lower1, upper1)
        mask2 = cv2.inRange(imagen_hsv_aislada, lower2, upper2)
        mask_total = mask1 + mask2 

        cv2.imshow("Click para Color (Solo Dial)", imagen_display_hsv) 
        cv2.imshow("Mascara (Solo Dial)", mask_total)
        resultado = cv2.bitwise_and(imagen_aislada_bgr, imagen_aislada_bgr, mask=mask_total)
        cv2.imshow("Resultado Filtrado (Solo Dial)", resultado)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        if key == ord('g'):
            hsv_ranges = {
                "range1": {"h_min": current_h_min1, "s_min": current_s_min, "v_min": current_v_min, 
                           "h_max": current_h_max1, "s_max": current_s_max, "v_max": current_v_max},
                "range2": {"h_min": current_h_min2, "s_min": current_s_min, "v_min": current_v_min, 
                           "h_max": current_h_max2, "s_max": current_s_max, "v_max": current_v_max}
            }
            
            # --- Guardar datos (MODIFICADO) ---
            config_data = {
                "centro": centro,
                "radio": radio,
                "angulo_cero": angulo_cero,
                "angulo_uno": angulo_uno,
                "total_numeros": 10,
                "sentido": "CW", 
                "hsv_ranges": hsv_ranges
            }
            
            ruta_config = os.path.join(script_dir, 'config_dial.json')
            with open(ruta_config, 'w') as f:
                json.dump(config_data, f, indent=4)
                
            print(f"--- ¡Configuración Completa Guardada! ---")
            print(f"Archivo guardado en: {ruta_config}")
            break

    cv2.destroyAllWindows()