import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT)

p = GPIO.PWM(11, 50)  # 50Hz frequency
p.start(0)
sleep(0.1)  # Initialization pause

try:
    while True:
        # Adjust these values for your specific servo's range
        p.ChangeDutyCycle(2.5)   # 0 degrees position
        sleep(1)
        p.ChangeDutyCycle(12.5)  # 180 degrees position
        sleep(1)
except KeyboardInterrupt:
    print("Stopped by user")
except Exception as e:
    print(f"Error: {e}")
finally:
    p.stop()
    GPIO.cleanup()