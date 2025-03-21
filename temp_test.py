import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT)
p = GPIO.PWM(11, 50)
p.start(0)

sleep(0.1)

try:
    while True:
        p.ChangeDutyCycle(3)
        sleep(1)
        p.ChangeDutyCycle(12)
        sleep(1)
except KeyboardInterrupt:
    print('s')
except Exception as e:
    print(f'An error occurred: {e}')
finally:
    p.stop()
    GPIO.cleanup()