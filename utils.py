import matplotlib.pyplot as plt
import numpy as np 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid


class ToBinary(object):
    """Home made transform to get 0-1 pixel as in paper"""
    def __call__(self, sample):
        sample[sample>0]=1
        return sample

def import_data(name, bs, plot=False, binary=False, normalize=True):
    
    if binary:
        transform = transforms.Compose([transforms.ToTensor(), ToBinary()])
    else:
        if normalize:
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.2860], [0.3530])])
        else:
            transform = transforms.Compose([transforms.ToTensor()])

    
    if name=="mnist":
        train = datasets.MNIST(root = './data/',train = True,download = True,transform=transform)
        test = datasets.MNIST(root = './data/',train = False,download = True,transform= transform)
        
    elif name=="fashion":
        train = datasets.FashionMNIST(root = './data/',train = True,download = True, transform= transform)
        test = datasets.FashionMNIST(root = './data/',train = False,download = True,transform = transform) 
    else:
        raise ValueError('Unknown dataset, available datasets: "mnist", "fashion"')
        
    train_loader = DataLoader(train,batch_size=bs, shuffle=True)
    test_loader = DataLoader(test,batch_size=bs, shuffle=True)
    
    if plot:
        real_batch, y = next(iter(train_loader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Image Example: Fashion MNIST (10 classes)")
        plt.imshow(np.transpose(make_grid(real_batch[:64], padding=2, normalize=True),(1,2,0)))
        plt.show()
        
    return train_loader, test_loader



def show(img):
    npimg = img.cpu().numpy()
    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')


def plot_recovery(images, out,m, name):
    fig, ax = plt.subplots(1+len(m), 10, figsize=(15,5))
    for j in range(10):
        ax[0,j].imshow(images[j*2].view(28,28).cpu().numpy(), cmap ='gray')
        ax[0,j].set_xticks([])
        ax[0,j].set_yticks([])
    for k in range((len(m))):
        for j in range(10):
            ax[k+1,j].imshow(out[k][j*2].reshape(28,28), cmap ='gray')
            ax[k+1,j].set_xticks([])
            ax[k+1,j].set_yticks([])
    fig.suptitle('''Top: Original cloth images
    2nd row: Reconstruction with {} measurements
    3nd row: Reconstruction with {} measurements
    4nd row: Reconstruction with {} measurements
     with {}'''.format(m[0],m[1],m[2],name), y = 1.1)
    plt.show()


