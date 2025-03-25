# video - https://youtu.be/40tZQPd3z8g?si=qw_VVsbo0cjdLHbv&t=270
# article - https://core-electronics.com.au/guides/control-servo-raspberry-pi/

from gpiozero import AngularServo
from time import sleep

servo =AngularServo(18, min_angle=0, max_angle=270, min_pulse_width=0.0005, max_pulse_width=0.0025)

while (True):
    servo.angle = 0
    sleep(2)
    servo.angle = 135
    sleep(2)
    servo.angle = 260
    sleep(2)