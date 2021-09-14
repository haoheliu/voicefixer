import json
# from scipy.io import loadmat
from pathlib import Path
import yaml
import pickle

def read_list(fname):
    result = []
    with open(fname, "r") as f:
        for each in f.readlines():
            each = each.strip('\n')
            result.append(each)
    return result

def write_list(list, fname):
    with open(fname,'w') as f:
        for word in list:
            f.write(word)
            f.write('\n')

def write_json(my_dict, fname):
    # print("Save json file at "+fname)
    json_str = json.dumps(my_dict)
    with open(fname, 'w') as json_file:
        json_file.write(json_str)

def load_json(fname):
    with open(fname,'r') as f:
        data = json.load(f)
        return data


# def load_mat2numpy(fname=""):
#     '''
#     Args:
#         fname: pth to mat
#         type:
#     Returns: dic object
#     '''
#     if (len(fname) == 0):
#         return None
#     else:
#         return loadmat(fname)

def save_pickle(obj,fname):
    # print("Save pickle at "+fname)
    with open(fname,'wb') as f:
        pickle.dump(obj,f)

def load_pickle(fname):
    # print("Load pickle at "+fname)
    with open(fname,'rb') as f:
        res = pickle.load(f)
    return res

def read_yaml(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def write_yaml(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        yaml.dump(content, handle)

# if __name__ =="__main__":
#     a = {"sdf":float(1.334234)}
#     write_json(a,"temp.json")
