import random
import torch

RANDOM_RESOLUTION=2**31

def random_torch(high, to_int=True):
    if (to_int):
        return int((torch.rand(1)) * high)  # do not use numpy.random.random
    else:
        return (torch.rand(1)) * high  # do not use numpy.random.random

def shuffle_torch(list):
    length = len(list)
    res = []
    order = torch.randperm(length)
    for each in order:
        res.append(list[each])
    assert len(list) == len(res)
    return res

def random_choose_list(list):
    num = int(uniform_torch(0,len(list)))
    return list[num]

def normal_torch(mean=0, segma=1):
    return float(torch.normal(mean=mean,std=torch.Tensor([segma]))[0])

def uniform_torch(lower, upper):
    if(abs(lower-upper)<1e-5):
        return upper
    return (upper-lower)*torch.rand(1)+lower

def random_key(keys:list, weights:list):
    return random.choices(keys, weights=weights)[0]

def random_select(probs):
    res = []
    chance = random_torch(RANDOM_RESOLUTION)
    threshold = None
    for prob in probs:
        # if(threshold is None):threshold=prob
        # else:threshold*=prob
        threshold = prob
        res.append(chance < threshold*RANDOM_RESOLUTION)
    return res, chance
