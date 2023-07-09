import torch
from multiprocessing import Pool
import Args
import Model
import torch.multiprocessing as mp
import DataGenerater
from torch.multiprocessing import SimpleQueue
import time
if mp.get_start_method() != 'spawn':
    mp.set_start_method('spawn')
queue = SimpleQueue()
done = mp.Event()

def train(tensor_: torch.nn.Module, q, device_):
    q.put(('a', 1))

    tensor_.to(device_)
    sta = tensor_.state_dict()
    sta['fc1.weight'] = sta['fc1.weight'] / 2
    tensor_.load_state_dict(sta)
    tensor_.to('cpu')
    q.put((tensor_, 1))
    print("sub",  tensor_.state_dict()['fc1.weight'])
    pr()
    q.put((None, 1))
    done.wait()


def pr():
    print("dad")



if __name__ == '__main__':
    args = Args.Arguments()
    pool: Pool =  Pool(3)
    data_ = DataGenerater.DatasetGen(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_processes = 4
    t1 = torch.tensor([1,2])
    # NOTE: this is required for the ``fork`` method to work
    model = Model.init_model('simple_mnist')

    print(model.state_dict()['fc1.weight'])



    p_list = []
    for rank in range(1):
        p = mp.Process(target=train, args=(model,queue, device))
        print(p.is_alive())
        p.start()

        print(p.is_alive())

        print(p.is_alive())

    print(model.state_dict()['fc1.weight'])
    while not queue.empty():
        res1, res2 = queue.get()
        if res1 is None:
            break
        else:
            print('queue, ', res1)
    done.set()