#define BUZZER_PIN 9 // Chân D9 nối với còi

// void setup() {
//   pinMode(BUZZER_PIN, OUTPUT); // Thiết lập D9 làm đầu ra
//   digitalWrite(BUZZER_PIN, LOW); // Đảm bảo còi ban đầu tắt
//   Serial.begin(9600); // Bắt đầu giao tiếp Serial
// }

// void loop() {
//   if (Serial.available() > 0) { // Kiểm tra xem có dữ liệu từ máy tính gửi tới
//     char command = Serial.read(); // Đọc dữ liệu
//     if (command == '1') {
//       // Phát âm thanh với tần số 2000 Hz (hoặc cao hơn nếu cần)
//       tone(BUZZER_PIN, 4000); // 2000 Hz là tần số âm thanh
//     } else if (command == '0') {
//       noTone(BUZZER_PIN); // Tắt âm thanh
//     }
//   }
// }
#define BUZZER_PIN 9 // Chân D9 nối với còi

unsigned long lastSignalTime = 0; // Thời gian nhận tín hiệu cuối cùng
const unsigned long timeout = 1000; // Thời gian chờ 1 giây

void setup() {
  pinMode(BUZZER_PIN, OUTPUT); // Thiết lập D9 làm đầu ra
  noTone(BUZZER_PIN); // Đảm bảo còi ban đầu tắt
  Serial.begin(9600); // Bắt đầu giao tiếp Serial
}

void loop() {
  // Kiểm tra tín hiệu từ Serial
  if (Serial.available() > 0) {
    char command = Serial.read(); // Đọc dữ liệu
    if (command == '1') {
      tone(BUZZER_PIN, 4000); // Phát âm thanh với tần số 4000 Hz
      lastSignalTime = millis(); // Cập nhật thời gian nhận tín hiệu
    } else if (command == '0') {
      noTone(BUZZER_PIN); // Tắt âm thanh
      lastSignalTime = millis(); // Cập nhật thời gian nhận tín hiệu
    }
  }

  // Tắt còi nếu không nhận tín hiệu trong thời gian chờ
  if (millis() - lastSignalTime > timeout) {
    noTone(BUZZER_PIN); // Tắt âm thanh nếu quá thời gian chờ
  }
}

