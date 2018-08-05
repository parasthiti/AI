import time
from RPLCD.gpio import CharLCD
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)

#set GPIO Pins
GPIO_TRIGGER = 17
GPIO_ECHO = 18

#set GPIO direction (IN / OUT)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)

# Initialize lcd
lcd = CharLCD(pin_rs=26, pin_e=19, cols=16, rows=2, pins_data=[13, 6, 5, 11], 
          numbering_mode=GPIO.BCM)

def distance():
    # set Trigger to HIGH for 10us (according to document)
    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.00001)
    # set Trigger LOW after 10us
    GPIO.output(GPIO_TRIGGER, False)
    
    # save StartTime
    while GPIO.input(GPIO_ECHO) == 0:
        StartTime = time.time()
 
    # save time of arrival
    while GPIO.input(GPIO_ECHO) == 1:
        StopTime = time.time()
 
    # time difference between start and arrival
    TimeElapsed = StopTime - StartTime
    # multiply with the sonic speed (34300 cm/s) or (12716.5 in/s)
    # and divide by 2, because there and back
    distance = round((TimeElapsed * 12716.5) / 2, 1)
 
    return distance

def lcd_display(content, lcd):
##    lcd.cursor_pos = (0,0)
##    lcd.cursor_mode = 'blink'
    lcd.write_string(content)
##    lcd.clear()

if __name__ == '__main__':
    try:
        while True:
            dist = distance()
            content = "\rDist = %.1f in" % dist
            lcd_display(content, lcd)
            print(content)
            time.sleep(1)
 
        # Reset by pressing CTRL + C
    except KeyboardInterrupt:
        print("Measurement stopped by User")
        lcd.clear()
        GPIO.cleanup()
        



