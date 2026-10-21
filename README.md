# Vortex Portal firmware

Prebuilt firmware for [Display Studio](https://ayelit.com). The Studio's
"Install on my board" button flashes these straight from the browser (Chrome or Edge),
no Arduino IDE needed. After installing, style and colours are sent from the Studio
over Bluetooth or USB and saved on the board.

| File | Board | Screen |
|---|---|---|
| `firmware/portal_s3_gc9a01.bin` | ESP32-S3 (SuperMini etc.), SCK 12, MOSI 11, CS 10, DC 9, RST 13 | 1.28" round GC9A01 |
| `firmware/portal_cyd35.bin` | 3.5" ESP32 display board (Hosyond / LCDwiki E32R35T) | built-in ST7796, landscape |

Each `.bin` is a merged image written at address `0x0`. `manifest.json` lists sizes and MD5s.
`esptool.js` is [esptool-js](https://github.com/espressif/esptool-js) v0.7.0 (Apache-2.0), bundled.
`src/` holds the exact sketches each firmware was built from (arduino-esp32 3.3.2, Adafruit GFX 1.12.6,
Adafruit ST7735/ST7789 1.11.0, Adafruit GC9A01A 1.1.1).
