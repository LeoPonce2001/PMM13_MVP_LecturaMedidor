#include <ESP8266WiFi.h>

// ---------- CONFIGURACIÓN ----------
#define SENSOR_PIN            D5         
#define SAMPLE_INTERVAL_MS    1000        // cálculo/impresión cada 1 s
#define POST_INTERVAL_MS      60000       
#define CALIBRATION_FACTOR    7.5        // calibrar

// WiFi + ThingSpeak
const char* WIFI_SSID = "TOMAS";
const char* WIFI_PASS = "20222022";
const char* TS_HOST   = "api.thingspeak.com";
const int   TS_PORT   = 80;
const char* TS_APIKEY = "YMXAA9TYAZUHDWII";  

// ---------- VARIABLES ----------
volatile unsigned long pulseCount = 0;
unsigned long prevMillis = 0;
unsigned long lastPost   = 0;

double flowRate_Lmin = 0.0;
double totalMilliLitres = 0.0;

// Acumuladores para promedio del caudal
double sumFlow = 0.0;
unsigned long samples = 0;

// ---------- cuenta pulsos con filtro ----------
IRAM_ATTR void pulseCounter() {
  static unsigned long lastMicros = 0;
  unsigned long now = micros();
  if (now - lastMicros > 500) {
    pulseCount++;
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN)); 
    lastMicros = now;
  }
}

// ---------- WiFi ----------
void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Conectando WiFi");
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
    if (++tries > 100) { Serial.println("\nReintentando..."); tries = 0; }
  }
  Serial.print("\nWiFi OK. IP: "); Serial.println(WiFi.localIP());
}

void ensureWiFi() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi caido. Re-conectando...");
    connectWiFi();
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);

  // Entrada del sensor SIN pull-up físico
  pinMode(SENSOR_PIN, INPUT_PULLUP);
  digitalWrite(SENSOR_PIN, HIGH); /
  attachInterrupt(digitalPinToInterrupt(SENSOR_PIN), pulseCounter, FALLING);

  connectWiFi();

  Serial.println(F("ms, pulses, Hz, L/min, mL_acum, L_acum"));
}


void loop() {
  unsigned long now = millis();
  unsigned long elapsed = now - prevMillis;

  // ---- Cálculo local y debug serial (cada 1 s) ----
  if (elapsed >= SAMPLE_INTERVAL_MS) {
    noInterrupts();
    unsigned long pulses = pulseCount;
    pulseCount = 0;
    interrupts();

    // Hz = pulsos/seg
    double freqHz = (elapsed > 0) ? (1000.0 * pulses / (double)elapsed) : 0.0;
    // L/min instantáneo
    flowRate_Lmin = freqHz / CALIBRATION_FACTOR;
    // Volumen en este intervalo
    double litresThisSample = flowRate_Lmin * ((double)elapsed / 60000.0);
    totalMilliLitres += litresThisSample * 1000.0;

    
    Serial.print(now); Serial.print(", ");
    Serial.print(pulses); Serial.print(", ");
    Serial.print(freqHz, 2); Serial.print(", ");
    Serial.print(flowRate_Lmin, 3); Serial.print(", ");
    Serial.print(totalMilliLitres, 1); Serial.print(", ");
    Serial.println(totalMilliLitres / 1000.0, 3);

    
    sumFlow += flowRate_Lmin;
    samples++;

    prevMillis = now;
  }

  if (now - lastPost >= POST_INTERVAL_MS) {
    ensureWiFi();

    double avgFlow = (samples > 0) ? (sumFlow / (double)samples) : 0.0;
    double totalLitres = totalMilliLitres / 1000.0;

    if (WiFi.status() == WL_CONNECTED) {
      WiFiClient client;
      if (client.connect(TS_HOST, TS_PORT)) {
        String postStr = String("api_key=") + TS_APIKEY +
                         "&field1=" + String(avgFlow, 3) +
                         "&field2=" + String(totalLitres, 3);

        client.print(String("POST /update HTTP/1.1\r\n") +
                     "Host: " + String(TS_HOST) + "\r\n" +
                     "Connection: close\r\n" +
                     "Content-Type: application/x-www-form-urlencoded\r\n" +
                     "Content-Length: " + postStr.length() + "\r\n\r\n" +
                     postStr + "\r\n");

        unsigned long t0 = millis();
        while (!client.available() && millis() - t0 < 2000) { delay(10); }
        String resp = client.readString();
        Serial.print("ThingSpeak resp: ");
        Serial.println(resp);
      } else {
        Serial.println(F("ThingSpeak: fallo de conexion."));
      }
    }

    // reinicia ventana de promedio
    sumFlow = 0.0;
    samples = 0;
    lastPost = now;
  }

  if (digitalRead(SENSOR_PIN) == LOW) {
    delayMicroseconds(20);
    digitalWrite(SENSOR_PIN, HIGH);
  }
}
