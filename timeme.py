import time
previous = time.time()
start = time.time()
def time_here(marker):
    global previous
    global start
    now = time.time()
    print("{marker}: {time}s, {diff}s".format(marker = marker, time = now - start, diff = now - previous))
    previous = now


