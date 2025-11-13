# debug_test_2.py
# Purpose: Does putting a tensor on a Manager.Queue cause a VRAM leak
# when the worker dies?

import multiprocessing as mp
import time
import os
import torch
from multiprocessing import Manager # Import the Manager

# --- Config ---
N_WORKERS = 1
LIFESPAN = 1 # maxtasksperchild = 1. Worker dies after 1 task.
N_TASKS = 5  # Run 5 tasks to force 5 worker deaths.

# ~1.9GB tensor
TENSOR_SHAPE = (1, 1024, 1024, 500) 

def gpu_worker(task_id, tensor_queue): # <-- Pass the queue
    """
    A simple task that:
    1. Allocates VRAM.
    2. Puts the tensor on the Manager.Queue.
    3. Holds it for 3 seconds.
    4. Exits (dies).
    """
    worker_pid = os.getpid()
    print(f"[PID {worker_pid}]: Worker BORN. Running task {task_id}.")
    
    if not torch.cuda.is_available():
        print(f"[PID {worker_pid}]: ERROR: CUDA not found.")
        return

    print(f"[PID {worker_pid}]: Allocating {TENSOR_SHAPE} tensor on {torch.device('cuda:0')}...")
    
    try:
        # 1. Allocate VRAM
        leaky_tensor = torch.randn(TENSOR_SHAPE, device='cuda:0')
        
        # 2. --- THIS IS THE TEST ---
        #    Move tensor to CPU and put it on the Manager.Queue
        print(f"[PID {worker_pid}]: Putting CPU tensor onto Manager.Queue...")
        tensor_cpu = leaky_tensor.cpu()
        tensor_queue.put(tensor_cpu)
        print(f"[PID {worker_pid}]: Tensor is on the queue.")
        
        # 3. Hold VRAM
        print(f"[PID {worker_pid}]: VRAM allocated. Sleeping for 3 seconds...")
        time.sleep(3)
        
        # 4. Die
        print(f"[PID {worker_pid}]: Task complete. Worker DYING.")
        del leaky_tensor
        del tensor_cpu
        
    except Exception as e:
        print(f"[PID {worker_pid}]: FAILED: {e}")
    
    return

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True) 
        print("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        pass
        
    print(f"--- STARTING DEBUG TEST 2 ---")
    print("Testing if Manager.Queue causes VRAM leaks.")
    print("Watch your VRAM (e.g., nvtop).")

    # --- Create the Manager and the Queue ---
    manager = Manager()
    tensor_queue = manager.Queue()

    pool = mp.Pool(
        processes=N_WORKERS,
        maxtasksperchild=LIFESPAN 
    )

    for i in range(N_TASKS):
        # Pass the queue to the worker
        pool.apply_async(gpu_worker, args=(i + 1, tensor_queue))
        time.sleep(5) # Wait 5s for the task to finish
        
        # Manually clear the queue from the parent side
        try:
            while not tensor_queue.empty():
                item = tensor_queue.get_nowait()
                del item
                print("[Main]: Cleared an item from the queue.")
        except Exception as e:
            print(f"[Main]: Error clearing queue: {e}")


    print("\n--- All tasks submitted. Closing pool. ---")
    pool.close()
    pool.join()
    
    print("\n--- DEBUG TEST 2 COMPLETE ---")
    print("All workers are dead. Check your VRAM now.")
    print("Did the VRAM usage return to 0?")
    print("If NOT, the Manager.Queue is the culprit.")
