// UNO Q MCU side: samples a Cytron MPU6050 at 50 Hz over I2C and streams the
// raw readings to the Linux side over the Bridge. The Linux side sends back the
// two models' predictions, shown on the LED matrix:
//   left letter = model 1 (peak accel only), right letter = model 2 (all IMU)
//   S = standing, W = walking, R = running, B = biking
//
// Wiring (MPU6050 -> UNO Q): VCC -> 3.3V, GND -> GND, SDA -> SDA, SCL -> SCL.

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <Wire.h>

const uint8_t MPU_ADDR = 0x68;          // AD0 low
const uint32_t SAMPLE_PERIOD_US = 20000; // 50 Hz, must match features.FS
const int BATCH = 10;                   // samples per Bridge message

Arduino_LED_Matrix matrix;

// 5x7 glyphs, one byte per row, low 5 bits used (bit 4 = leftmost column).
const uint8_t GLYPHS[4][7] = {
  {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E},  // S
  {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A},  // W
  {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11},  // R
  {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E},  // B
};

volatile int shownPeak = -1, shownFull = -1;
volatile bool displayDirty = false;

String batch;
int batchCount = 0;
uint32_t nextSampleUs;

void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

uint8_t readReg(uint8_t reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0xFF;
}

void setupMpu() {
  writeReg(0x6B, 0x80);  // PWR_MGMT_1: reset
  delay(100);
  writeReg(0x6B, 0x01);  // wake, clock = gyro X PLL
  writeReg(0x1A, 0x03);  // CONFIG: DLPF ~44 Hz (anti-aliasing for 50 Hz)
  writeReg(0x19, 19);    // SMPLRT_DIV: 1 kHz / (1 + 19) = 50 Hz
  writeReg(0x1B, 0x18);  // GYRO_CONFIG: +-2000 deg/s (16.4 LSB per deg/s)
  writeReg(0x1C, 0x18);  // ACCEL_CONFIG: +-16 g (2048 LSB per g); running at the ankle exceeds 8 g
}

bool readMpu(int16_t v[6]) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);  // ACCEL_XOUT_H; 14 bytes: accel xyz, temp, gyro xyz
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, (uint8_t)14) != 14) return false;
  uint8_t b[14];
  for (int i = 0; i < 14; i++) b[i] = Wire.read();
  v[0] = (b[0] << 8) | b[1];
  v[1] = (b[2] << 8) | b[3];
  v[2] = (b[4] << 8) | b[5];
  v[3] = (b[8] << 8) | b[9];
  v[4] = (b[10] << 8) | b[11];
  v[5] = (b[12] << 8) | b[13];
  return true;
}

void drawGlyph(uint32_t frame[4], int glyph, int col0) {
  if (glyph < 0 || glyph > 3) return;
  for (int r = 0; r < 7; r++) {
    for (int c = 0; c < 5; c++) {
      if (GLYPHS[glyph][r] & (0x10 >> c)) {
        int idx = r * 13 + col0 + c;
        frame[idx / 32] |= 1UL << (31 - idx % 32);
      }
    }
  }
}

void render() {
  uint32_t frame[4] = {0, 0, 0, 0};
  drawGlyph(frame, shownPeak, 0);
  drawGlyph(frame, shownFull, 8);
  matrix.loadFrame(frame);
}

// Runs in the Bridge RPC thread: only record the request, draw in loop().
void show_activity(int peakModel, int fullModel) {
  shownPeak = peakModel;
  shownFull = fullModel;
  displayDirty = true;
}

void setup() {
  Bridge.begin();
  Monitor.begin();
  matrix.begin();

  Wire.begin();
  Wire.setClock(400000);
  setupMpu();
  uint8_t who = readReg(0x75);
  Monitor.print("MPU6050 WHO_AM_I = 0x");
  Monitor.println(who, HEX);  // 0x68 for a genuine part; clones may report 0x70/0x72

  Bridge.provide("show_activity", show_activity);
  batch.reserve(BATCH * 48);
  nextSampleUs = micros();
}

void loop() {
  if (displayDirty) {
    displayDirty = false;
    render();
  }

  if ((int32_t)(micros() - nextSampleUs) < 0) return;
  nextSampleUs += SAMPLE_PERIOD_US;

  int16_t v[6];
  if (!readMpu(v)) return;

  batch += millis();
  for (int i = 0; i < 6; i++) {
    batch += ',';
    batch += v[i];
  }
  batch += ';';

  if (++batchCount >= BATCH) {
    Bridge.notify("imu_batch", batch);
    batch = "";
    batchCount = 0;
  }
}
