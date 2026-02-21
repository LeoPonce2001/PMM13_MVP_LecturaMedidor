#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

const char* ssid = "SSID";
const char* password = "PASS";

const auto flashGpio = GPIO_NUM_4;

WebServer server(80);

void handleCapture() {
    digitalWrite(flashGpio, HIGH);
    delay(500);

    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);

    fb = esp_camera_fb_get();

    digitalWrite(flashGpio, LOW);

    if (!fb) {
        server.send(500, "text/plain", "Camera capture failed");
        return;
    }

    server.sendHeader("Content-Type", "image/jpeg");
    server.sendHeader("Content-Length", String(fb->len));
    server.send_P(200, "image/jpeg", (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
}

void setup() {
    Serial.begin(115200);

    pinMode(flashGpio, OUTPUT);
    digitalWrite(flashGpio, LOW);

    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) delay(500);

    delay(1000);

    // Configure camera pins and settings
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0 = 5;
    config.pin_d1 = 18;
    config.pin_d2 = 19;
    config.pin_d3 = 21;
    config.pin_d4 = 36;
    config.pin_d5 = 39;
    config.pin_d6 = 34;
    config.pin_d7 = 35;
    config.pin_xclk = 0;
    config.pin_pclk = 22;
    config.pin_vsync = 25;
    config.pin_href = 23;
    config.pin_sscb_sda = 26;
    config.pin_sscb_scl = 27;
    config.pin_pwdn = 32;
    config.pin_reset = -1;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = FRAMESIZE_XGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;

    if (esp_camera_init(&config) != ESP_OK) {
        Serial.println("Camera init failed");
        return;
    }

    server.on("/capture", handleCapture);
    server.begin();

    Serial.print("Server started: ");
    Serial.println(WiFi.localIP());
}

void loop() {
    server.handleClient();
}
