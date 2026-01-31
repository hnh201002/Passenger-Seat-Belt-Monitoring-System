#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <WebServer.h>
#include <Base64.h>
#include <ESPmDNS.h>

// Wi-Fi credentials
const char* ssid = "Xiaomi AX3000";
const char* password = "tumotdenmot";

// Flask server address
const char* flask_server_ip = "http://127.0.0.1:5000/capture";

#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

WebServer server(80);

void sendDataToFlask(const String& base64_image) {
  HTTPClient http;
  http.begin(flask_server_ip);
  http.setTimeout(20000);
  http.addHeader("Content-Type", "application/json");

  String payload = "{\"image\": \"" + base64_image + "\"}";
  Serial.println("Payload size: " + String(payload.length()));
  int httpResponseCode = http.POST(payload);

  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.println("Response from Flask server: " + response);
  } else {
    Serial.println("Error sending POST request: " + String(httpResponseCode));
  }

  http.end();
}

void startCameraServer() {
  server.on("/capture", HTTP_GET, []() {
    // Capture image from camera
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      server.send(500, "text/plain", "Failed to capture image");
      return;
    }

    // Release the framebuffer to ensure a fresh capture
    esp_camera_fb_return(fb);
    delay(100); // Short delay to allow framebuffer to refresh

    // Re-capture the image
    fb = esp_camera_fb_get();
    if (!fb) {
      server.send(500, "text/plain", "Failed to capture refreshed image");
      return;
    }

    // Encode the image to base64
    String base64_image = base64::encode(fb->buf, fb->len);

    // Send CORS headers
    server.sendHeader("Access-Control-Allow-Origin", "*");

    // Send the image to Flask server
    sendDataToFlask(base64_image);

    // Respond with the base64 image
    String response = "{\"image\": \"" + base64_image + "\"}";
    server.send(200, "application/json", response);

    // Release framebuffer memory
    esp_camera_fb_return(fb);
  });

  server.begin();
  Serial.println("Camera Server started");
}

void setup() {
  Serial.begin(115200);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");

  if (!MDNS.begin("esp")) {
    Serial.println("Error starting mDNS responder");
  } else {
    Serial.println("mDNS responder started");
    Serial.println("Access the ESP at http://esp.local");
  }

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_UXGA;
  config.pixel_format = PIXFORMAT_JPEG;
  config.jpeg_quality = 4;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  }

  // Adjust brightness using sensor API
  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    s->set_brightness(s, 1); // Increase brightness (-2 to 2)
    Serial.println("Brightness set to 2");
  } else {
    Serial.println("Failed to access camera sensor");
  }

  startCameraServer();
  Serial.print("Camera Ready! Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");
}

void loop() {
  server.handleClient();
}
