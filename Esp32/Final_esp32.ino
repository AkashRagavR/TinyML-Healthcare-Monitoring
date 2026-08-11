#include <Wire.h>
#include "xgb_model.h"

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

// ================= MODEL DATA (from model_data.cc) =================
extern unsigned char ecg_model_int8_tflite[];
extern unsigned int  ecg_model_int8_tflite_len;

// ================= SENSOR CONFIG =================
#define MAX30100_ADDR 0x57
#define MLX90614_ADDR 0x5A
#define ECG_PIN 34

// MAX30100 registers
#define REG_FIFO_DATA   0x05
#define REG_MODE_CONFIG 0x06
#define REG_SPO2_CONFIG 0x07
#define REG_LED_CONFIG  0x09
#define REG_FIFO_WR_PTR 0x02
#define REG_OVF_COUNTER 0x03
#define REG_FIFO_RD_PTR 0x04

// ================= DATASET RANGES =================
const int HR_MIN   = 44,  HR_MAX   = 139;
const int SPO2_MIN = 83,  SPO2_MAX = 111;
const int TEMP_MIN = 21,  TEMP_MAX = 49;

// ================= CALIBRATION =================
const long  FINGER_THRESHOLD = 5000;
const long  irRawMin  = 5000,  irRawMax  = 60000;
const long  redRawMin = 1000,  redRawMax = 30000;
const float skinTempMin = 30.0, skinTempMax = 40.0;
const float TEMP_OFFSET = -2.0;

// ================= TFLITE =================
constexpr int INPUT_LENGTH = 187;
constexpr int NUM_CLASSES  = 5;

uint8_t tensor_arena[25 * 1024];

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor  = nullptr;
TfLiteTensor* output_tensor = nullptr;

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const char* cnn_labels[NUM_CLASSES] = {
  "NORMAL",
  "SUPRAVENTRICULAR BEAT",
  "VENTRICULAR BEAT",
  "FUSION BEAT",
  "UNKNOWN BEAT"
};

// ================= RUNTIME STATE =================
int   ecgBuffer[INPUT_LENGTH];
int   ecgIndex  = 0;
bool  ecgReady  = false;
float hr        = 0;
float spo2      = 0;
float temp      = 0;
int   cachedECG = 0;

unsigned long lastSample  = 0;
unsigned long lastI2CRead = 0;
unsigned long lastPrint   = 0;

#define SAMPLE_MS   5
#define I2C_READ_MS 50
#define PRINT_MS    3000

// ================================================
// HELPERS
// ================================================
void writeReg(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

// ================================================
// SETUP
// ================================================
void setup() {
  Serial.begin(115200);
  delay(2000);

  Wire.begin();
  Wire.setClock(100000);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  // ---------- MAX30100 INIT ----------
  writeReg(MAX30100_ADDR, REG_MODE_CONFIG, 0x40);
  delay(100);
  writeReg(MAX30100_ADDR, REG_FIFO_WR_PTR, 0x00);
  writeReg(MAX30100_ADDR, REG_OVF_COUNTER, 0x00);
  writeReg(MAX30100_ADDR, REG_FIFO_RD_PTR, 0x00);
  writeReg(MAX30100_ADDR, REG_MODE_CONFIG, 0x03);
  writeReg(MAX30100_ADDR, REG_SPO2_CONFIG, 0x27);
  writeReg(MAX30100_ADDR, REG_LED_CONFIG,  0x6F);

  // ---------- TFLITE INIT ----------
  const tflite::Model* model = tflite::GetModel(ecg_model_int8_tflite);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Model schema version mismatch!");
    while (1) { delay(1000); }
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, sizeof(tensor_arena), error_reporter);

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed!");
    while (1) { delay(1000); }
  }

  input_tensor  = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.println("SYSTEM READY.");
}

// ================================================
// LOOP
// ================================================
void loop() {
  unsigned long now = millis();

  // ---------- 1. ECG SAMPLING at 200Hz ----------
  if (now - lastSample >= SAMPLE_MS) {
    lastSample = now;
    cachedECG  = analogRead(ECG_PIN);
    ecgBuffer[ecgIndex++] = cachedECG;
    if (ecgIndex >= INPUT_LENGTH) {
      ecgIndex = 0;
      ecgReady = true;
    }
  }

  // ---------- 2. VITALS READ at 20Hz ----------
  if (now - lastI2CRead >= I2C_READ_MS) {
    lastI2CRead = now;

    // MAX30100 read
    Wire.beginTransmission(MAX30100_ADDR);
    Wire.write(REG_FIFO_DATA);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)MAX30100_ADDR, (uint8_t)4);

    uint16_t irRaw = 0, redRaw = 0;
    if (Wire.available() == 4) {
      irRaw  = ((uint16_t)Wire.read() << 8) | Wire.read();
      redRaw = ((uint16_t)Wire.read() << 8) | Wire.read();
    }

    // MLX90614 read
    float rawTemp = 0;
    Wire.beginTransmission(MLX90614_ADDR);
    Wire.write(0x07);
    if (Wire.endTransmission(false) == 0) {
      Wire.requestFrom((uint8_t)MLX90614_ADDR, (uint8_t)3);
      if (Wire.available() >= 3) {
        uint16_t data = Wire.read();
        data |= ((uint16_t)Wire.read() << 8);
        Wire.read(); // PEC discard
        rawTemp = (data * 0.02f) - 273.15f;
      }
    }

    // Process vitals
    if (irRaw > FINGER_THRESHOLD) {
      hr   = (float)map(irRaw,   irRawMin,  irRawMax,  HR_MIN,   HR_MAX);
      spo2 = (float)map(redRaw,  redRawMin, redRawMax, SPO2_MIN, SPO2_MAX);
      hr   = constrain(hr,   HR_MIN,   HR_MAX);
      spo2 = constrain(spo2, SPO2_MIN, SPO2_MAX);
      temp = rawTemp + TEMP_OFFSET;
    } else {
      hr   = 0;
      spo2 = 0;
      temp = rawTemp + TEMP_OFFSET;
    }
  }

  // ---------- 3. AI INFERENCE + PRINT every 3s ----------
  if (now - lastPrint >= PRINT_MS) {
    lastPrint = now;

    // CNN Inference (ECG)
    int   cnn_class = 0;
    float cnn_conf  = 0.0f;

    if (ecgReady) {
      int minV = ecgBuffer[0], maxV = ecgBuffer[0];
      for (int i = 1; i < INPUT_LENGTH; i++) {
        if (ecgBuffer[i] < minV) minV = ecgBuffer[i];
        if (ecgBuffer[i] > maxV) maxV = ecgBuffer[i];
      }
      int range = (maxV - minV == 0) ? 1 : maxV - minV;

      for (int i = 0; i < INPUT_LENGTH; i++) {
        float norm = (float)(ecgBuffer[i] - minV) / (float)range;
        float q    = (norm / input_tensor->params.scale)
                     + input_tensor->params.zero_point;
        q = constrain(q, -128.0f, 127.0f);
        input_tensor->data.int8[i] = (int8_t)q;
      }

      if (interpreter->Invoke() == kTfLiteOk) {
        float max_s = -128.0f;
        for (int i = 0; i < NUM_CLASSES; i++) {
          float s = (output_tensor->data.int8[i]
                     - output_tensor->params.zero_point)
                    * output_tensor->params.scale;
          if (s > max_s) { max_s = s; cnn_class = i; cnn_conf = s; }
        }
      }
    }

    // XGBoost Inference (Vitals)
    float xgb_probs[2] = {0.5f, 0.5f};
    int   xgb_class    = -1;

    if (hr > 0) {
      xgb_class = predict_xgb(hr, spo2, temp, xgb_probs);
    }

    // Serial Output
    Serial.println();
    Serial.println("========================================");
    Serial.println("           HEALTHCARE MONITOR           ");
    Serial.println("========================================");

    Serial.print("ECG RAW     : "); Serial.println(cachedECG);

    Serial.print("HEART RATE  : ");
    if (hr <= 0) Serial.println("-- (no finger)");
    else { Serial.print((int)hr); Serial.println(" BPM"); }

    Serial.print("SPO2        : ");
    if (spo2 <= 0) Serial.println("-- (no finger)");
    else { Serial.print((int)spo2); Serial.println(" %"); }

    Serial.print("TEMPERATURE : ");
    Serial.print(temp, 1); Serial.println(" C");

    Serial.println();
    Serial.println("CONDITION OF THE PATIENT");
    Serial.println("----------------------------------------");

    Serial.print("ECG         : ");
    if (!ecgReady) {
      Serial.println("COLLECTING DATA...");
    } else if (cnn_class == 0) {
      Serial.println("NORMAL");
    } else {
      Serial.print("RISK (");
      Serial.print(cnn_labels[cnn_class]);
      Serial.println(")");
    }

    Serial.print("VITALS      : ");
    if (xgb_class == -1) {
      Serial.println("NO FINGER DETECTED");
    } else if (xgb_class == 1) {
      Serial.println("NORMAL");
    } else {
      Serial.println("RISK");
    }

    if (xgb_class >= 0) {
      Serial.print("AI CONFID.  : ");
      Serial.print(xgb_probs[xgb_class] * 100.0f, 1);
      Serial.println("%");
    }

    Serial.println("========================================");
  }
}