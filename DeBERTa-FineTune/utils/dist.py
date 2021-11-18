import pickle

import torch
import torch.distributed as dist

from contextlib import contextmanager


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def is_master():
    return get_rank() == 0


def kill_all_process():
    dist.destroy_process_group()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when using distributed training.
    """

    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    if dist.get_world_size() <= 1:
        return

    dist.barrier()


def init_process_group(backend='nccl', init_method='env://', world_size=-1, rank=-1):
    dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)
    synchronize()


def all_reduce(tensors_dict: dict):
    reduced = {}
    for k, v in tensors_dict.items():
        assert isinstance(v, torch.Tensor)

        v_ = v.clone()
        dist.all_reduce(v_)
        v_ /= get_world_size()

        reduced[k] = v_.item()
    
    return reduced


def gather(data, device):
    """将各进程的数据(不一定是tensor类型也不要求同样大小)收集起来并且同步到各个进程"""

    num_gpus = dist.get_world_size()

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device)

    # obtain Tensor size of each rank
    local_size = torch.IntTensor([tensor.numel()]).to(device)
    size_list = [torch.IntTensor([0]).to(device) for _ in range(num_gpus)]
    dist.all_gather(size_list, local_size)

    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    tensor_list = [torch.ByteTensor(size=(max_size,)).to(device) for _ in size_list]

    if local_size != max_size:
        # we pad the tensor because torch all_gather does not support
        # gathering tensors of different shapes
        # padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        padding = torch.ByteTensor(size=(max_size - local_size,)).to(device)
        tensor = torch.cat((tensor, padding), dim=0)

    # receiving Tensor from all ranks
    dist.all_gather(tensor_list, tensor)

    return tensor_list, size_list


@contextmanager
def distributed_master_first(rank: int):
    """
    分布式模式下，主进程优先执行上下文管理器下的操作，待主进程执行完，其余进程才得以开始执行
       用法：
           with distributed_master_first:
               do something
    """

    # rank > 1代表副进程
    # 在此先等待(同步)
    if rank:
        synchronize()

    # 相当于一个 return
    yield
    # yield 后面的语句待退出上下文环境时再执行
    # rank=0 代表主进程
    # 此时主进程已执行完上下文环境中的语句
    if not rank:
        # 主进程通知其余进程(同时也有等待其余进程的效果)
        synchronize()
