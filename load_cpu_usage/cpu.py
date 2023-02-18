import time
import math
time_of_run = 0.1
percent_cpu = 80 # Should ideally replace this with a smaller number e.g. 20
cpu_time_utilisation = float(percent_cpu)/100
on_time = time_of_run * cpu_time_utilisation
off_time = time_of_run * (1-cpu_time_utilisation)
while True:
    start_time = time.time()
    while time.time() - start_time < on_time:
        math.factorial(100) #Do any computation here
    time.sleep(off_time)