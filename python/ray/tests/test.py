import sys
import time
import json
import torch
import ray
from ray.experimental.collective import create_collective_group


@ray.remote(num_gpus=1, num_cpus=0)
class GPUTestActor:
    @ray.method(tensor_transport="nccl")
    def echo(self, data):
        return data.to("cuda")

    def sum(self, data):
        return data.sum().item()


def main():
    ray.init(num_gpus=2)

    world_size = 2
    actors = [GPUTestActor.remote() for _ in range(world_size)]
    create_collective_group(actors, backend="nccl")

    src_actor, dst_actor = actors[0], actors[1]

    tensor = torch.tensor([1, 2, 3])

    start_time = time.time()

    gpu_ref = src_actor.echo.remote(tensor)
    remote_sum = ray.get(dst_actor.sum.remote(gpu_ref))

    end_time = time.time()

    print("Sum on destination actor:", remote_sum)
    print("E2E time: {:.3f} ms".format((end_time - start_time) * 1000))

    assert tensor.sum().item() == remote_sum

    time.sleep(2)  # Allow Ray events to flush

    with open("ray_timeline.json", "w") as f:
        f.write(json.dumps(ray.timeline(), indent=2))

    print("Timeline written to ray_timeline.json")

    ray.shutdown()


if __name__ == "__main__":
    main()