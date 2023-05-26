import multiprocessing
import time

# Function for the first process (perpetual loop)
def process1(shared_variable):
    while True:
        shared_variable.value += 1
        # print("Process 1:", shared_variable.value)
        time.sleep(1)  # Delay for 1 second

# Function for the second process
def process2(shared_variable):
    while True:
        time.sleep(0.5)  # Delay for 2 seconds
        print("Process 2:", shared_variable.value)

if __name__ == "__main__":
    # Create a shared variable using multiprocessing.Value
    shared_variable = multiprocessing.Value('i', 0)

    # Create two separate processes
    p1 = multiprocessing.Process(target=process1, args=(shared_variable,))
    p2 = multiprocessing.Process(target=process2, args=(shared_variable,))

    # Start the processes
    p1.start()
    p2.start()

    # Wait for the processes to finish
    p1.join()
    p2.join()

    # Print the final value of the shared variable
    print("Final value:", shared_variable.value)
