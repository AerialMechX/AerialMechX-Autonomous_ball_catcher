/*
 * Arduino Mega 4WD Robot - PRECISE SPEED CONTROL (PID)
 * --- (Corrected Wiring for RPi Pipeline) ---
 * --- VERSION 3 ---
 * - Listens for serial commands from RPi
 * - 'w' = Forward, 'a' = Left, 'd' = Right, 's' = Stop
 * - 'x' = Backward (NEW)
*/

#include <PID_v1.h> // Include the PID library

// --- !! YOU MUST CHANGE THESE !! ---
const double PPR = 600; // <<< PUT YOUR MOTOR'S "Pulses Per Revolution" HERE
const int FORWARD_RPM = 120; // Target RPM for movement
const int TURN_RPM = 100;    // Target RPM for turning

// --- PID Tuning Parameters ---
double Kp = 1.5, Ki = 0.5, Kd = 0.1;

// --- Pin Definitions (Motors - YOUR WIRING) ---
// Driver 1
const int ENA_FL = 6, IN1_FL = 22, IN2_FL = 24; // Front-Left
const int ENB_FR = 7, IN1_FR = 26, IN2_FR = 28; // Front-Right
// Driver 2
const int ENA_RL = 8, IN1_RL = 30, IN2_RL = 32; // Rear-Left
const int ENB_RR = 9, IN1_RR = 34, IN2_RR = 36; // Rear-Right

// --- Pin Definitions (Encoders) ---
const int ENC_A_FL = 2;  // Front-Left (Interrupt 0)
const int ENC_A_FR = 18; // Front-Right (Interrupt 5)
const int ENC_A_RL = 3;  // Rear-Left (Interrupt 1)
const int ENC_A_RR = 19; // Rear-Right (Interrupt 4)

// --- Global PID & RPM Variables ---
double targetRPM_FL = 0, targetRPM_FR = 0, targetRPM_RL = 0, targetRPM_RR = 0;
double currentRPM_FL = 0, currentRPM_FR = 0, currentRPM_RL = 0, currentRPM_RR = 0;
double pwmOutput_FL = 0, pwmOutput_FR = 0, pwmOutput_RL = 0, pwmOutput_RR = 0;

// PID Controller Objects
PID pid_FL(&currentRPM_FL, &pwmOutput_FL, &targetRPM_FL, Kp, Ki, Kd, DIRECT);
PID pid_FR(&currentRPM_FR, &pwmOutput_FR, &targetRPM_FR, Kp, Ki, Kd, DIRECT);
PID pid_RL(&currentRPM_RL, &pwmOutput_RL, &targetRPM_RL, Kp, Ki, Kd, DIRECT);
PID pid_RR(&currentRPM_RR, &pwmOutput_RR, &targetRPM_RR, Kp, Ki, Kd, DIRECT);

// Interrupt Pulse Counters
volatile long pulseCount_FL = 0, pulseCount_FR = 0, pulseCount_RL = 0, pulseCount_RR = 0;
unsigned long lastCalcTime = 0;
const int calcInterval = 100; // Calculate RPM every 100ms

void setup() {
  Serial.begin(9600);

  // Motor pins
  pinMode(ENA_FL, OUTPUT); pinMode(IN1_FL, OUTPUT); pinMode(IN2_FL, OUTPUT);
  pinMode(ENB_FR, OUTPUT); pinMode(IN1_FR, OUTPUT); pinMode(IN2_FR, OUTPUT);
  pinMode(ENA_RL, OUTPUT); pinMode(IN1_RL, OUTPUT); pinMode(IN2_RL, OUTPUT);
  pinMode(ENB_RR, OUTPUT); pinMode(IN1_RR, OUTPUT); pinMode(IN2_RR, OUTPUT);

  // Encoder pins
  pinMode(ENC_A_FL, INPUT_PULLUP);
  pinMode(ENC_A_FR, INPUT_PULLUP);
  pinMode(ENC_A_RL, INPUT_PULLUP);
  pinMode(ENC_A_RR, INPUT_PULLUP);

  // Attach Interrupts
  attachInterrupt(digitalPinToInterrupt(ENC_A_FL), countPulseFL, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_A_FR), countPulseFR, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_A_RL), countPulseRL, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_A_RR), countPulseRR, RISING);

  // Turn on PID controllers
  pid_FL.SetMode(AUTOMATIC); pid_FL.SetOutputLimits(0, 255);
  pid_FR.SetMode(AUTOMATIC); pid_FR.SetOutputLimits(0, 255);
  pid_RL.SetMode(AUTOMATIC); pid_RL.SetOutputLimits(0, 255);
  pid_RR.SetMode(AUTOMATIC); pid_RR.SetOutputLimits(0, 255);

  Serial.println("RPi Motor Controller Ready. (w,a,s,d,x)");
  stopMotors();
}

void loop() {
  // Check for Serial commands
  if (Serial.available() > 0) {
    char command = Serial.read();
    switch (command) {
      case 'w':
        //Serial.println("Cmd: Forward");
        setDirectionForward();
        setAllTargetRPM(FORWARD_RPM);
        break;
      case 'x': // <-- NEW COMMAND
        //Serial.println("Cmd: Backward");
        setDirectionBackward();
        setAllTargetRPM(FORWARD_RPM);
        break;
      case 'a':
        //Serial.println("Cmd: Left");
        setDirectionTurnLeft();
        setTurnLeftRPM(TURN_RPM);
        break;
      case 'd':
        //Serial.println("Cmd: Right");
        setDirectionTurnRight();
        setTurnRightRPM(TURN_RPM);
        break;
      case 's':
        //Serial.println("Cmd: Stop");
        stopMotors();
        break;
    }
  }

  // Calculate RPM and run PID controllers every 'calcInterval' milliseconds
  if (millis() - lastCalcTime >= calcInterval) {
    calculateRPM();

    pid_FL.Compute();
    pid_FR.Compute();
    pid_RL.Compute();
    pid_RR.Compute();

    if (targetRPM_FL != 0 || targetRPM_FR != 0 || targetRPM_RL != 0 || targetRPM_RR != 0) {
      updateMotorPWM();
    }
    
    lastCalcTime = millis();
  }
}

// --- Interrupt Service Routines (ISRs) ---
void countPulseFL() { pulseCount_FL++; }
void countPulseFR() { pulseCount_FR++; }
void countPulseRL() { pulseCount_RL++; }
void countPulseRR() { pulseCount_RR++; }

// --- RPM Helper Functions ---
void setAllTargetRPM(int rpm) {
  targetRPM_FL = rpm;
  targetRPM_FR = rpm;
  targetRPM_RL = rpm;
  targetRPM_RR = rpm;
}

void setTurnRightRPM(int rpm) {
  targetRPM_FL = rpm; // Left wheels move
  targetRPM_RL = rpm;
  targetRPM_FR = 0;   // Right wheels stop
  targetRPM_RR = 0;
}

void setTurnLeftRPM(int rpm) {
  targetRPM_FL = 0;   // Left wheels stop
  targetRPM_RL = 0;
  targetRPM_FR = rpm; // Right wheels move
  targetRPM_RR = rpm;
}

// --- RPM Calculation ---
void calculateRPM() {
  double rpm_conversion_factor = (1000.0 * 60.0) / (PPR * calcInterval);

  currentRPM_FL = (pulseCount_FL * rpm_conversion_factor);
  currentRPM_FR = (pulseCount_FR * rpm_conversion_factor);
  currentRPM_RL = (pulseCount_RL * rpm_conversion_factor);
  currentRPM_RR = (pulseCount_RR * rpm_conversion_factor);

  pulseCount_FL = 0;
  pulseCount_FR = 0;
  pulseCount_RL = 0;
  pulseCount_RR = 0;
}

// --- Motor Control Functions ---

void setDirectionForward() {
  digitalWrite(IN1_FL, HIGH); digitalWrite(IN2_FL, LOW);
  digitalWrite(IN1_FR, HIGH); digitalWrite(IN2_FR, LOW);
  digitalWrite(IN1_RL, HIGH); digitalWrite(IN2_RL, LOW);
  digitalWrite(IN1_RR, HIGH); digitalWrite(IN2_RR, LOW);
}

// --- NEW FUNCTION ---
void setDirectionBackward() {
  digitalWrite(IN1_FL, LOW); digitalWrite(IN2_FL, HIGH);
  digitalWrite(IN1_FR, LOW); digitalWrite(IN2_FR, HIGH);
  digitalWrite(IN1_RL, LOW); digitalWrite(IN2_RL, HIGH);
  digitalWrite(IN1_RR, LOW); digitalWrite(IN2_RR, HIGH);
}

void setDirectionTurnRight() {
  digitalWrite(IN1_FL, HIGH); digitalWrite(IN2_FL, LOW); // Left Fwd
  digitalWrite(IN1_RL, HIGH); digitalWrite(IN2_RL, LOW);
  digitalWrite(IN1_FR, LOW); digitalWrite(IN2_FR, LOW);  // Right Stop
  digitalWrite(IN1_RR, LOW); digitalWrite(IN2_RR, LOW);
}

void setDirectionTurnLeft() {
  digitalWrite(IN1_FL, LOW); digitalWrite(IN2_FL, LOW);  // Left Stop
  digitalWrite(IN1_RL, LOW); digitalWrite(IN2_RL, LOW);
  digitalWrite(IN1_FR, HIGH); digitalWrite(IN2_FR, LOW); // Right Fwd
  digitalWrite(IN1_RR, HIGH); digitalWrite(IN2_RR, LOW);
}

void updateMotorPWM() {
  analogWrite(ENA_FL, pwmOutput_FL);
  analogWrite(ENB_FR, pwmOutput_FR);
  analogWrite(ENA_RL, pwmOutput_RL);
  analogWrite(ENB_RR, pwmOutput_RR);
}

void stopMotors() {
  setAllTargetRPM(0);

  digitalWrite(IN1_FL, LOW); digitalWrite(IN2_FL, LOW);
  digitalWrite(IN1_FR, LOW); digitalWrite(IN2_FR, LOW);
  digitalWrite(IN1_RL, LOW); digitalWrite(IN2_RL, LOW);
  digitalWrite(IN1_RR, LOW); digitalWrite(IN2_RR, LOW);

  analogWrite(ENA_FL, 0);
  analogWrite(ENB_FR, 0);
  analogWrite(ENA_RL, 0);
  analogWrite(ENB_RR, 0);
}
