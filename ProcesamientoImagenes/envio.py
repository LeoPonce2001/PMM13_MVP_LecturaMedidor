import os
import ssl
import json
import paho.mqtt.client as mqtt

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))

client = mqtt.Client()
client.on_connect = on_connect

current_dir = os.path.dirname(os.path.abspath(__file__))
client.tls_set(
    ca_certs=os.path.join(current_dir, 'rootCA.pem'),
    certfile=os.path.join(current_dir, '5ffe1fa04f09a724a10e169ff98366ba0fca87894de45cf5da43cf825fd1d29e-certificate.pem.crt'),
    keyfile=os.path.join(current_dir, '5ffe1fa04f09a724a10e169ff98366ba0fca87894de45cf5da43cf825fd1d29e-private.pem.key'),
    tls_version=ssl.PROTOCOL_TLSv1_2
)

client.connect("a30btnoaw9tzxc-ats.iot.us-east-1.amazonaws.com", 8883, 60)

def publish_json(data):
    try:
        print("Datos a publicar:", json.dumps(data))

        client.publish("device/data", payload=json.dumps(data), qos=0, retain=False)
        print("Datos publicados en device/data.")
        
        # Procesar el bucle para asegurarse de que el mensaje se envíe

    except Exception as e:
        print(f"Error al leer o publicar Lecturas.json: {e}")

#publish_json()

# Desconectar después de publicarlo
#client.disconnect()
