import torch.nn.functional as F
import torch.optim as optim
from cs import sense
import torch.nn as nn


class Generator(torch.nn.Module):
    def __init__(self, zdim):
        super(Generator, self).__init__()
        n_features = 100
        n_out = 784
        self.fc1 = nn.Linear(zdim, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 784)

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x),negative_slope=0.2)
        h = F.leaky_relu(self.fc2(h),negative_slope=0.2)
        return self.fc3(h)
 
    def decode(self, x):
        h = F.leaky_relu(self.fc1(x),negative_slope=0.2)
        h = F.leaky_relu(self.fc2(h),negative_slope=0.2)
        return self.fc3(h)




model = Generator(100).to(device)
model = model.eval()



#Deep Compresse sensing Algorithm 1:
#Init
m=50
n=784
epochs=67
zdim=100

normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0/m]))
A = normal.sample((m,n)).squeeze().to(device)
model = Generator(100).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9,0.999))

for e in range(epochs):
  print("epoch /",e)
  for b,(x_batch, _) in enumerate(train_loader):
    if b%50==0:
      print(b,len(train_loader))
    model.zero_grad()
    lossG=0
    lossF=0
    list_z_i_best=[]
    for i,x_i in enumerate(x_batch):

      m_i = torch.mm(A,x_i.view(28*28,1).cuda())
      z_i=torch.randn(1,zdim,device=device)

      #inner loop
      z_i_best=sense(model,device,A,m_i,zdim,,0.,n_restart=1,n_inner=3)#no regul, no restart ,3 latent steps
      m_i_hat=torch.mm(A,model.decode(z_i_best).T)
      #in order to sample one fake
      list_z_i_best.append(z_i_best)

      lossG+=torch.pow(torch.norm(m_i - m_i_hat), 2)

    #computing F loss

  
    #1 image sample among the images of the current batch #2 generation from G, 2 images randomly selected
    random_choice1 = np.random.randint(len(x_batch))
    real_sample=x_batch[random_choice1].view(28*28,1).cuda()
    
    # one post latent optimization
    random_choice_bis = np.random.randint(len(x_batch))
    z_1=list_z_i_best[random_choice_bis]
    #one before
    z_2=torch.randn(1,20,device=device)
    fake_sample_1 = model.decode(z_1).view(28*28,1).cuda()
    fake_sample_2 = model.decode(z_2).view(28*28,1).cuda()
    
    #triple lossF
    t1=torch.pow(torch.norm(torch.mm(A,(real_sample-fake_sample_1)))-torch.norm(real_sample-fake_sample_1),2)
    t2=torch.pow(torch.norm(torch.mm(A,(real_sample-fake_sample_2)))-torch.norm(real_sample-fake_sample_2),2)
    t3=torch.pow(torch.norm(torch.mm(A,(fake_sample_1-fake_sample_2)))-torch.norm(fake_sample_1-fake_sample_2),2)
    lossG=t1+t2+t3

    loss=lossG+lossF
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "DCS1.pth")
