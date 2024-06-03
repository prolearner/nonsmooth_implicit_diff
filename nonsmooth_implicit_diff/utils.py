from datetime import datetime
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import warnings
from os.path import join


class LRScheduler:
    def __init__(self, optimizer, step_size_f):
        self.optimizer = optimizer
        self.iteration = 0
        self.lr_function = lambda : step_size_f(self.iteration)

    def get_lr(self):
        return self.lr_function()

    def step(self):
        self.iteration += 1
        new_lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

def when_is_support_identified(w_true, params_history):
    matching_vector = np.zeros(len(params_history), dtype=bool)

    for i, params_vector in enumerate(params_history):
        matching_vector[i] = np.all(~ np.logical_xor(params_vector != 0, w_true != 0))

    indices = np.where(matching_vector)[0]
    support_ideintification_index =  int(indices[0]) if indices.size > 0 else None
    
    lost_support=False
    if support_ideintification_index is not None and not np.all(matching_vector[support_ideintification_index:]):
        lost_support=True
    
    if matching_vector is not None:
        matching_vector=[bool(m) for m in matching_vector]
    


    return support_ideintification_index, matching_vector, lost_support


def convert_to_torch(tensors):
    return [torch.from_numpy(t.astype(np.float32)) for t in tensors]


def get_train_val_test_split(X, y, random_state=None, test_size = 0.25, val_size = 0.25):
    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=(test_size + val_size), random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=test_size, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


def copy_tensor_list(tensor_list):
    return [tensor.detach().clone() for tensor in tensor_list]

def vectorize(tensor_list):
    def unsqueeze_maybe(t):
        if len(t.shape) == 0:
            return t.unsqueeze(dim=0)
        else:
            return t
        
    return torch.cat([unsqueeze_maybe(t) for t in tensor_list], dim=0)

def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False

def is_list_of_numbers(v):
    if isinstance(v, list):
        return all(isinstance(e, (int, float)) for e in v)
    else:
        return False

def transform(object, check, f):
    if isinstance(object, dict):
        return {k:transform(v, check, f) for k, v in object.items()}
    if object is not None and check(object):
        return f(object)
    if is_iterable(object) and not isinstance(object, str):
        return [transform(t, check, f) for t in object]
    return object

def to_np(object):
    return transform(object,
                     check= lambda x: isinstance(x, torch.Tensor),
                     f= lambda x: x.detach().cpu().numpy())

def to_list(object):
    return transform(object, check=lambda x: isinstance(x, np.ndarray),
                      f= lambda x: x.tolist())

def to_torch(tensor_list_or_dict):
    return transform(tensor_list_or_dict, check=lambda x: is_list_of_numbers(x),
                      f= lambda x: torch.tensor(x))

def make_exp_folder(conf, params_in_name, save_path):
    def convert(v):
        if isinstance(v, float):
            return f"{v:.2e}"
        return str(v)

    folder_name = "".join([f'{k}_{convert(conf[k])}_' for k in params_in_name])[:-1]
    save_path = join(save_path, folder_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        with open(join(save_path, 'conf.json'), 'r') as json_file:
            conf_existing = json.load(json_file)
        if conf_existing != conf:
            save_path = save_path + '_' + datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
            os.makedirs(save_path)
    
    with open(join(save_path, 'conf.json'), 'w') as json_file:
        json.dump(conf, json_file, indent=2)

    print(f"Saving results in {save_path}")        
    return save_path

if __name__ == '__main__':
    # n_samples = 10
    # n_features = 10
    # noise = 0.1
    # effective_rank = 3
    # n_informative = 5
    # correlated = True
    # shuffle=False
    # random_state=None

    # X, y, w = make_sparse_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative, correlated=correlated,
    #                                 noise=noise, effective_rank=effective_rank, shuffle=shuffle)
    
    # print(X, y, w)


    # Example usage:
    w_true = np.array([0, 2, 0, 4])
    params_history = [np.array([1, 2, 3, 4]), np.array([0, 2, 0, 1]), np.array([0, 0, 0, 4])]

    result = when_is_support_identified(w_true, params_history)
    print(result)
    
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)