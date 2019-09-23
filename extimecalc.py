import time

time1 = time.time()

def chrono(n=0, prt=True):
    global time1
    if (prt) : print(" "*n+"Computation time : {0:.2f} seconds".format(time.time() - time1))
    time1 = time.time()