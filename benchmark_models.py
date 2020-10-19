"""Compare speed of different models with batch size 12"""
import torch
import torchvision.models as models
import platform,psutil
import torch.nn as nn
import time,os
import pandas as pd
import argparse
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from torch.cuda.amp import autocast

torch.backends.cudnn.benchmark = True
# https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
# This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware. 
# If you check it using the profile tool, the cnn method such as winograd, fft, etc. is used for the first iteration and the best operation is selected for the device.


MODEL_LIST = {

    # models.mnasnet:models.mnasnet.__all__[1:],
    models.resnet: models.resnet.__all__[1:],
    # models.densenet: models.densenet.__all__[1:],
    # models.squeezenet: models.squeezenet.__all__[1:],
    # models.vgg: models.vgg.__all__[1:],
    # models.mobilenet:models.mobilenet.__all__[1:],
    # models.shufflenetv2:models.shufflenetv2.__all__[1:]
}

precisions=["auto", "float", "half"]

# For post-voltaic architectures, there is a possibility to use tensor-core at half precision.
# Due to the gradient overflow problem, apex is recommended for practical use.
device_name=str(torch.cuda.get_device_name(0))
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Benchmarking')
parser.add_argument('--WARM_UP','-w', type=int,default=4, required=False, help="Num of warm up")
parser.add_argument('--NUM_TEST','-n', type=int,default=50,required=False, help="Num of Test")
parser.add_argument('--BATCH_SIZE','-b', type=int, default=32, required=False, help='Num of batch size')
parser.add_argument('--NUM_CLASSES','-c', type=int, default=1000, required=False, help='Num of class')
parser.add_argument('--NUM_GPU','-g', type=int, default=1, required=False, help='Num of gpus')
parser.add_argument('--folder','-f', type=str, default='result', required=False, help='folder to save results')
args = parser.parse_args()




class RandomDataset(Dataset):

    def __init__(self,  length):
        self.len = length
        self.data = torch.randn(length, 3, 224, 224, dtype=torch.float16)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

class Autocast(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    @autocast()
    def forward(self, input):
        return self.m.forward(input)




        

def prepare_model(model, precision):

    types = dict(auto = torch.float16, float = torch.float32, half = torch.float16)
    input_type = types[precision]
    if precision == "auto":
        model = Autocast(model)
    else:
        model = model.to(input_type)

    model = nn.DataParallel(model, device_ids=range(args.NUM_GPU))
    model = model.to('cuda')

    return model, input_type

def train(model, input_type, loader):
    target = torch.LongTensor(loader.batch_size).random_(args.NUM_CLASSES).cuda()
    criterion = nn.CrossEntropyLoss()

    model.train()

    for step, img in enumerate(loader):
        img = img.to(input_type)

        model.zero_grad()
        prediction = model(img)
        loss = criterion(prediction, target)
        loss.backward()    

def test(model, input_type, loader):
    model.eval()
    for step, img in enumerate(loader):
        model(img.to(input_type))

def gmean(input_x, dim=0):
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim))

def benchmark_models(name, task, precision="auto"):

    batch_scale = dict(auto = 2, half = 2, float = 1)
    batch_size = args.BATCH_SIZE * args.NUM_GPU * batch_scale[precision]  

    rand_loader = DataLoader(dataset=RandomDataset(batch_size * args.NUM_TEST), 
        batch_size=batch_size, shuffle=False,num_workers=0)
    
    warmup_loader = DataLoader(dataset=RandomDataset(batch_size * args.WARM_UP), 
        batch_size=batch_size, shuffle=False,num_workers=0)

    benchmark = {}
    for model_type in MODEL_LIST.keys():
        for model_name in MODEL_LIST[model_type]:
            model = getattr(model_type, model_name)(pretrained=False)
            model, input_type = prepare_model(model, precision)

            task(model, input_type, warmup_loader)

            torch.cuda.synchronize()
            start = time.time()

            task(model, input_type, rand_loader)

            torch.cuda.synchronize()
            end = time.time()

            rate = len(rand_loader) * batch_size  / (end - start)
            print(model_name, precision,  rate, 'images/sec')
            del model

            benchmark[model_name] = rate

    total = torch.tensor(benchmark.values())
    print("geometric mean total ({}, precision={}): {%.4f}".format(name, precision, gmean(total)))

    return benchmark



class no_op():
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False



if __name__ == '__main__':
    folder_name=args.folder
    device_name="".join((device_name, '_',str(args.NUM_GPU),'_gpus_'))
    system_configs=str(platform.uname())
    system_configs='\n'.join((system_configs,str(psutil.cpu_freq()),'cpu_count: '+str(psutil.cpu_count()),'memory_available: '+str(psutil.virtual_memory().available)))
    gpu_configs=[torch.cuda.device_count(),torch.version.cuda,torch.backends.cudnn.version(),torch.cuda.get_device_name(0)]
    gpu_configs=list(map(str,gpu_configs))
    temp=['Number of GPUs on current device : ','CUDA Version : ','Cudnn Version : ','Device Name : ']

    os.makedirs(folder_name, exist_ok=True)
    now = time.localtime()
    start_time=str("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
    
    print('benchmark start : ',start_time)

    for idx,value in enumerate(zip(temp,gpu_configs)):
        gpu_configs[idx]=''.join(value)
        print(gpu_configs[idx])

    print(system_configs)

    with open(os.path.join(folder_name,"system_info.txt"), "w") as f:
        f.writelines('benchmark start : '+start_time+'\n')
        f.writelines('system_configs\n\n')
        f.writelines(system_configs)
        f.writelines('\ngpu_configs\n\n')
        f.writelines(s + '\n' for s in gpu_configs )

        for precision in precisions:
            train_result=benchmark_models("train", train, precision)
            test_result=benchmark_models("test", test, precision)


    now = time.localtime()
    end_time=str("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
    print('benchmark end : ',end_time)
    with open(os.path.join(folder_name,"system_info.txt"), "a") as f:
        f.writelines('benchmark end : '+end_time+'\n')


