import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, world_size):
    # 1. Initialize the process group
    start = time.time()
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank,
    )

    torch.cuda.set_device(rank)

    if rank == 0:
        # Sender
        tensor = torch.tensor([1, 2, 3], device=f"cuda:{rank}")
        print(f"[Rank {rank}] Sending tensor: {tensor}")
        dist.send(tensor=tensor, dst=1)
        end = time.time()

        print(f"[Rank {rank}] Send time: {(end - start) * 1000:.3f} ms")
        
    elif rank == 1:
        # Receiver
        tensor = torch.empty(3, device=f"cuda:{rank}", dtype=torch.int64)
        dist.recv(tensor=tensor, src=0)
        end = time.time()

        print(f"[Rank {rank}] Received tensor: {tensor}")
        print(f"Sum = {tensor.sum().item()}")
        print(f"[Rank {rank}] Recv time: {(end - start) * 1000:.3f} ms")
    dist.destroy_process_group()

def main():
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()