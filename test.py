print("Hello World")
import time
tim = time.time()
print(tim)
time.sleep(1)
tim2 = time.time() +10000
time_difference = tim2-tim
days = int(time_difference // (24 * 3600))
time_difference %= (24 * 3600)
hours = int(time_difference // 3600)
time_difference %= 3600
minutes = int(time_difference // 60)
seconds = int(time_difference % 60)
st = f"Time difference: {days:02}:{hours:02}:{minutes:02}:{seconds:02}"
print(st)