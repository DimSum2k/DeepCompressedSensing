from utils import show

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid


class VAE(nn.Module):
    def __init__(self,z_dim):
        super(VAE, self).__init__()

        # encoder (regression network)
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc31 = nn.Linear(500, z_dim)
        self.fc32 = nn.Linear(500, z_dim)
        # decoder (generator)
        self.fc4 = nn.Linear(z_dim, 500)
        self.fc5 = nn.Linear(500, 500)
        self.fc6 = nn.Linear(500, 784)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)

    def decode(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h)) 

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAEcnn(nn.Module):
    def __init__(self,z_dim):
        super(VAEcnn, self).__init__()

        # encoder (regression network)
        self.conv1 = nn.Conv2d(1, 64, 4, stride=2,padding=2)
        self.conv2 = nn.Conv2d(64, 64, 4, stride=2,padding=2)
        self.conv3 = nn.Conv2d(64, 64, 4, stride=1,padding=1)
        self.fc11 = nn.Linear(64*7*7, z_dim)
        self.fc12 = nn.Linear(64*7*7, z_dim)
        # decoder (generator)
        self.fc4 = nn.Linear(z_dim, 49)
        self.fc5 = nn.Linear(49, 49)
        self.deconv1 = nn.ConvTranspose2d(1, 64, 4, stride=1)
        self.deconv2 = nn.ConvTranspose2d(64, 64, 4)
        self.deconv3 = nn.ConvTranspose2d(64, 64, 4)
        self.fc6 = nn.Linear(64*16*16, 28*28)



    def encode(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.view(-1,64*7*7)
        return self.fc11(h), self.fc12(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1,1,28,28))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def decode(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        h = h.view(-1,1,7,7)
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        h = h.view(-1,64*16*16)
        return torch.sigmoid(self.fc6(h)).view(-1,28*28)

        




        



def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train_epoch(model,optimizer,device, epoch, trainset):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(trainset):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx %250 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainset.dataset),
                100. * batch_idx / len(trainset),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(trainset.dataset)))



def test_epoch(model, device, epoch, testset, name):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(testset):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(testset.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/' + name + '/reconstruction_' + str(epoch) + '.png', nrow=n)
                
    test_loss /= len(testset.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))



def train_vae(zdim, nb_epochs, device, trainset, testset=None, name='', cnn=False):
    
    assert name
    try:
        os.mkdir(os.path.join("results",name))
    except OSError:
        print ("Creation of the directory %s failed" % os.path.join("results",name))
    else:
        print ("Successfully created the directory %s " % os.path.join("results",name))

    if cnn:
        model = VAEcnn(z_dim=zdim).to(device)
    else:
        model = VAE(z_dim=zdim).to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    
    for epoch in range(1, nb_epochs + 1):
        
        train_epoch(model,optimizer,device, epoch,trainset)  
        test_epoch(model,device, epoch, testset, name)

        with torch.no_grad():
            sample = torch.randn(64, zdim).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/' + name + '/sample_' + str(epoch) + '.png')        

    torch.save(model.state_dict(), "save/" + name + '.pth')
        
    # Visualisation of the results
    with torch.no_grad():
        z = torch.randn(64, zdim).to(device)  
        sample = model.decode(z).to(device)  
        save_image(sample.view(64, 1, 28, 28), 'results/' + name + '/generation' + '.png')
        show(make_grid(sample.view(64, 1, 28, 28), padding=0))  