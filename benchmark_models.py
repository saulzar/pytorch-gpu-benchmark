"""Compare speed of different models with batch size 12"""
import torch
import torchvision.models as models
import platform,psutil
import torch.nn as nn
import time,os
import pandas as pd
import argparse
from torch.utils.data import Dataset, DataLoader

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

precisions=["amp", "float","half"]
# For post-voltaic architectures, there is a possibility to use tensor-core at half precision.
# Due to the gradient overflow problem, apex is recommended for practical use.
device_name=str(torch.cuda.get_device_name(0))
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Benchmarking')
parser.add_argument('--WARM_UP','-w', type=int,default=5, required=False, help="Num of warm up")
parser.add_argument('--NUM_TEST','-n', type=int,default=50,required=False, help="Num of Test")
parser.add_argument('--BATCH_SIZE','-b', type=int, default=32, required=False, help='Num of batch size')
parser.add_argument('--NUM_CLASSES','-c', type=int, default=1000, required=False, help='Num of class')
parser.add_argument('--NUM_GPU','-g', type=int, default=1, required=False, help='Num of gpus')
parser.add_argument('--folder','-f', type=str, default='result', required=False, help='folder to save results')
args = parser.parse_args()
args.BATCH_SIZE*=args.NUM_GPU
class RandomDataset(Dataset):

    def __init__(self,  length):
        self.len = length
        self.data = torch.randn( 3, 224, 224,length)

    def __getitem__(self, index):
        return self.data[:,:,:,index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset( args.BATCH_SIZE*(args.WARM_UP + args.NUM_TEST)),
                         batch_size=args.BATCH_SIZE, shuffle=False,num_workers=8)
def train():
    """use fake image for training speed test"""
    target = torch.LongTensor(args.BATCH_SIZE).random_(args.NUM_CLASSES).cuda()
    criterion = nn.CrossEntropyLoss()
    benchmark = {}
    for model_type in MODEL_LIST.keys():
        for model_name in MODEL_LIST[model_type]:
            model = getattr(model_type, model_name)(pretrained=False)
            if args.NUM_GPU > 1:
                model = nn.DataParallel(model,device_ids=range(args.NUM_GPU))

            model=model.to('cuda')
            durations = []
            print('Benchmarking Training {} '.format(model_name))

            with autocast():

                for step,img in enumerate(rand_loader):

                    torch.cuda.synchronize()
                    start = time.time()
                    model.zero_grad()
                    prediction = model(img.to('cuda'))
                    loss = criterion(prediction, target)
                    loss.backward()
                    torch.cuda.synchronize()
                    end = time.time()
                    if step >= args.WARM_UP:
                        durations.append((end - start))

                rate = args.BATCH_SIZE  / (sum(durations)/len(durations))        
                print(model_name,' model average train time : ',  rate, 'images/sec')
                del model
                benchmark[model_name] = durations

    return benchmark


class no_op():
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False


def inference():
    benchmark = {}
    with torch.no_grad():
        for model_type in MODEL_LIST.keys():
            for model_name in MODEL_LIST[model_type]:
                model = getattr(model_type, model_name)(pretrained=False)
                if args.NUM_GPU > 1:
                    model = nn.DataParallel(model,device_ids=range(args.NUM_GPU))

                model=model.to('cuda')
                model.eval()

                loss_fn = torch.nn.MSELoss(reduction='sum')
                target = torch.randn(args.BATCH_SIZE, 1000)

                print('Benchmarking Inference {} '.format(model_name))
                durations = []

                with autocast():
                    for step,img in enumerate(rand_loader):
                        
                        torch.cuda.synchronize()
                        start = time.time()
                        result = model(img.to('cuda')).sum()

                        torch.cuda.synchronize()
                        end = time.time()
                        if step >= args.WARM_UP:
                            durations.append((end - start))

                    rate = args.BATCH_SIZE  / (sum(durations)/len(durations))        
                    print(model_name,' model average inference time : ',  rate, 'images/sec')
                    del model
                    benchmark[model_name] = durations
    return benchmark




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

        train_result=train()
        train_result_df = pd.DataFrame(train_result)
        path=''.join((folder_name,'/',device_name, '_model_train_benchmark.csv'))
        train_result_df.to_csv(path, index=False)

        inference_result=inference()
        inference_result_df = pd.DataFrame(inference_result)
        path=''.join((folder_name,'/',device_name, '_model_inference_benchmark.csv'))
        inference_result_df.to_csv(path, index=False)

    now = time.localtime()
    end_time=str("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
    print('benchmark end : ',end_time)
    with open(os.path.join(folder_name,"system_info.txt"), "a") as f:
        f.writelines('benchmark end : '+end_time+'\n')


