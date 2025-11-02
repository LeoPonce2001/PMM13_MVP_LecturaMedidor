# 💧 PMM13_MVP_LecturaMedidor proyecto de Lectura Automática de Medidores de Agua
## Leonardo Ponce Toledo

## Descripción
Este proyecto implementa un **sistema IoT de lectura automática de medidores de agua**, combinando visión por computadora, sensado físico y Envío de lectuara a la Nube AWS IoT Core.  
El sistema está dividido en tres módulos principales:

1. **Sensado (ESP32-CAM + Óptica + Raspberry Pi)**  
2. **Desagregación y Procesamiento de Datos (Modelo TFLite de Jomjol)**  
3. **Envío (AWS IoT Core)**

---

## Módulo de Sensado
  - ESP32-CAM con flash integrado y lupa óptica 5×.  
  - Montaje en tubo PVC 110 mm con tapa removible.  
  - Flash interno como fuente de iluminación.  
  - Conectividad Wi-Fi para envío de imágenes a la Raspberry Pi.

---

## Procesamiento Local (Raspberry Pi 4)
- Recepción de imágenes vía **HTTPS**.  
- Inferencia mediante **modelo TFLite** basado en [jomjol/AI-on-the-edge-device](https://github.com/jomjol/AI-on-the-edge-device).  
- Detección de dígitos y lectura de agujas rojas (HSV).  
- Conversión de los resultados en un archivo temporal `Lecturas.json`.

---

## Envío a la Nube (AWS IoT Core)
- Transmisión de datos vía **MQTT** hacia **AWS IoT Core**.  
- Envío tras completar 10 lecturas consecutivas VALIDAS.  
- Formato de mensaje JSON:
  {
    "fecha_hora": "2025-10-31 18:29:37",
    "display": 3487710.534968271
  }
