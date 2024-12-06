import os, tqdm, random, torch
import numpy as np
import torch.nn as nn
from PIL import Image
from multiprocessing import Manager as SharedMemoryManager
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
from torchvision.models.resnet import ResNet, BasicBlock
from typing import Optional, Union, Iterable, Tuple

# Config parameters
data_dir = '/Users/jayas/Documents/Privacy_models/imagenet_highresolution/super resolution/high_resolution'
epochs = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_channels = 3
first_fmap_channels = 64
last_fmap_channels = 512
output_channels = 3
time_embedding = 256
learning_rate = 1e-4
min_lr = 1e-6
weight_decay = 0.0
n_timesteps = 500
beta_min = 1e-4
beta_max = 2e-2
beta_scheduler = 'linear'
batch_size = 10
n_samples = 12_00
cache_size = 12_00
image_size = (640, 640)



class ImageDataset(Dataset):
    def __init__(self, img_dir:str, image_size:tuple, n_samples:int=4000, cache_size:int=1000):
        self.img_dir = img_dir
        self.n_samples = n_samples
        self.image_size = image_size
        self.cache_size = cache_size

        self.files = os.listdir(self.img_dir)
        self.files = random.sample(self.files, self.n_samples)

        r"""
        container for caching samples:
        structure
        -------------
        {idx:<image>}
        """
        self.cache = SharedMemoryManager().dict()
        
        self.transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.files)


    def _addToCache(self, idx:int, image:torch.Tensor):
        if self.cache_size > 0:
            if len(self.cache) >= self.cache_size:
                keys = list(self.cache.keys())
                key_to_remove = random.choice(keys)
                self.cache.pop(key_to_remove)
            self.cache[idx] = image

    def __getitem__(self, idx:int):
        if idx in self.cache.keys():return self.cache[idx]
            
        file_path = os.path.join(self.img_dir, self.files[idx])
        image = Image.open(file_path)
        if self.transforms:
            image = self.transforms(image)

        self._addToCache(idx, image)
        return image



class DiffusionUtils:
    def __init__(self, n_timesteps:int, beta_min:float, beta_max:float, device:str='cpu', scheduler:str='linear'):
        assert scheduler in ['linear', 'cosine'], 'scheduler must be linear or cosine'
        self.n_timesteps = n_timesteps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.device = device
        self.scheduler = scheduler
        self.betas = self.betaSamples()
        self.alphas = 1 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def betaSamples(self):
        if self.scheduler == 'linear':
            return torch.linspace(start=self.beta_min, end=self.beta_max, steps=self.n_timesteps).to(self.device)

        elif self.scheduler == 'cosine':
            betas = []
            for i in reversed(range(self.n_timesteps)):
                T = self.n_timesteps - 1
                beta = self.beta_min + 0.5*(self.beta_max - self.beta_min) * (1 + np.cos((i/T) * np.pi))
                betas.append(beta)
                
            return torch.Tensor(betas).to(self.device)

    def sampleTimestep(self, size:int):
        #the size argument will let you randomly sample a batch of timesteps
        #output shape: (N, )
        return torch.randint(low=1, high=self.n_timesteps, size=(size, )).to(self.device)

    def noiseImage(self, x:torch.Tensor, t:torch.LongTensor):
        #expected input is a batch of inputs.
        #image shape: (N, C, H, W)
        #t:torch.Tensor shape: (N, )
        assert len(x.shape) == 4, 'input must be 4 dimensions'
        alpha_hat_sqrts = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        one_mins_alpha_hat_sqrt = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x).to(self.device)
        return (alpha_hat_sqrts * x) + (one_mins_alpha_hat_sqrt * noise), noise

    def sample(self, x:torch.Tensor, model:nn.Module):
        #x shape: (N, C, H, W)
        assert len(x.shape) == 4, 'input must be 4 dimensions'
        model.eval()
        
        with torch.no_grad():
            iterations = range(1, self.n_timesteps)
            for i in tqdm.tqdm(reversed(iterations)):
                #batch of timesteps t
                t = (torch.ones(x.shape[0]) * i).long().to(self.device)

            #params
                alpha = self.alphas[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                one_minus_alpha = 1 - alpha
                one_minus_alpha_hat = 1 - alpha_hat
                
                #predict noise pertaining for a given timestep
                predicted_noise = model(x, t)
                
                if i > 1:noise = torch.randn_like(x).to(self.device)
                else:noise = torch.zeros_like(x).to(self.device)
                
                x = 1/torch.sqrt(alpha) * (x - ((one_minus_alpha / torch.sqrt(one_minus_alpha_hat)) * predicted_noise))
                x = x + (torch.sqrt(beta) * noise)
                
            return x

def main():
    files = os.listdir(data_dir)
    print(f'number of images: {len(files)}')
    n_rows, n_cols = 2,6
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(17,6))

    for i in range(n_rows):
        for j in range(n_cols):
            idx = random.randint(0,len(files))
            img = Image.open(os.path.join(data_dir, f'{files[idx]}'))
            axs[i,j].imshow(img)
    dataset = ImageDataset(data_dir, image_size, n_samples)
    sample_img = dataset[0]

    print(f'shape: {sample_img.shape}')
    print(f'min pixel value: {sample_img.min()}')
    print(f'mean pixel value: {sample_img.mean()}')
    print(f'max pixel value: {sample_img.max()}')

    plt.imshow(sample_img.permute(1, 2, 0))
    

    T = n_timesteps
    n_steps = 50
    alpha_values = {}
    
    for scheduler in ['linear', 'cosine']:
        print(f'{scheduler} beta scheduling...')
    
        diffusion = DiffusionUtils(T, beta_min, beta_max, scheduler=scheduler)
        alpha_values[scheduler] = diffusion.alphas
    
        fig, axs = plt.subplots(1, (T//n_steps)+1, figsize=(25, 15))
    
        axs[0].imshow(sample_img.permute(1, 2, 0))
        axs[0].set_title('t = 0')
    
        for idx, t in enumerate(range(n_steps-1, T, n_steps)):
            t = torch.Tensor([t]).long()
            x, _ = diffusion.noiseImage(sample_img.unsqueeze(0), t)
            axs[idx+1].imshow(x.squeeze(0).permute(1, 2, 0))
            axs[idx+1].set_title(f't = {t.item()}')
            plt.show()
            print('\n')

if __name__=="__main__":
    main()


























