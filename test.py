import multiprocessing, time

def do_something(i):
    print(i)
    time.sleep(i)



if __name__ == "__main__":
    print(len(multiprocessing.active_children()))
    manager = multiprocessing.Manager()
    #new_circuits = manager.list()
    #time.sleep(20)
    lock = multiprocessing.Lock()
    manager = None
    # For some reason pickleing every circuit indiviually and sending it via the process is WAY slower than pickleing all at once and sending over the pickle file
    print(len(multiprocessing.active_children()))
    #time.sleep(20)