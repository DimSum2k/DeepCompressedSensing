import numpy as np 
import torch
import torch.optim as optim


def sense(model, device, A, y, zdim, regularization,n_restart=10,n_inner=1000):
    
    assert regularization>=0
    
    z_best = torch.randn(1, zdim).to(device)
    z_best.requires_grad_(False)
    
    # 10 restart as in paper
    for _ in range(n_restart): 
        z = torch.randn(1, zdim).to(device)
        z.requires_grad_(True)
        
        optimizer = optim.Adam([z], lr = 0.01)
        
        for _ in range(n_inner):  
            AG_z = torch.mm(A,model.decode(z).view(-1,1)) 
            loss = torch.pow(torch.norm(AG_z - y), 2) + regularization*torch.pow(torch.norm(z),2)
            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()
                            
        sample = model.decode(z).to(device)
                            
        with torch.no_grad():
            if loss < torch.pow(torch.norm(torch.mm(A,model.decode(z_best).view(-1,1).to(device))-y),2):
                z_best = z  
                                
    return z_best


def generate(model,device, m, n, zdim, x_star, regularization):
    # measurement matrix
    normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0/m]))
    A = normal.sample((m,n)).squeeze().to(device)
    A.requires_grad_(False)
    
    # noise
    normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([0.0])) # loc=std
    noise = normal.sample((m,)).to(device)
    noise.requires_grad_(False)
    
    # mesurements
    y = (torch.mm(A,x_star) + noise).to(device)
    y.requires_grad_(False)
    
    #recovery
    z = sense(model,device,A, y, zdim,regularization)
    
    return model.decode(z).view(28, 28).detach().cpu().numpy()


def score(true_im, reconstruction):
    return np.mean((true_im - reconstruction)**2)

def get_score(predictions,signals):

    assert len(predictions)==len(signals)

    s=0
    #for pred, (true,_) in zip(predictions,signals):
    for pred, true in zip(predictions,signals):
        s+= score(true.numpy().flatten(),pred.flatten())
        #s+= score(true.flatten(),pred.flatten())
    return s/len(predictions)  



def sense_sparse(model, device, A, y, zdim, n_restart=10,n_inner=1000):
    
    spare_gen_weight = 0.01
    
    z_best = torch.randn(1, zdim).to(device)
    z_best.requires_grad_(False)
    
    v_best = torch.randn(1, 784).to(device)
    v_best.requires_grad_(False)
    
    for _ in range(n_restart): 
        z = torch.randn(1, zdim).to(device)
        z.requires_grad_(True)
        
        v = torch.randn(1, 784).to(device)
        v.requires_grad_(True)
        
        optimizer = optim.Adam([z,v], lr = 0.01)
        
        for _ in range(n_inner):  
            AG_zv = torch.mm(A,model.decode(z).view(-1,1) + v)  # A(G(z)+v)
            loss = torch.pow(torch.norm(AG_zv - y), 2) + spare_gen_weight*torch.norm(v,1)
            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()
                            
        sample = model.decode(z).to(device)
                            
        with torch.no_grad():
            if loss < torch.pow(torch.norm(torch.mm(A,model.decode(z_best).view(-1,1) + v) - y), 2) + spare_gen_weight*torch.norm(v,1):
                z_best = z  
                                
    return z_best



def generate_sparse(model,device, m, n, zdim, x_star):
    # measurement matrix
    normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0/m]))
    A = normal.sample((m,n)).squeeze().to(device)
    A.requires_grad_(False)
    
    # noise - torch.tensor([0.1])
    normal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([0.0])) 
    noise = normal.sample((m,)).to(device)
    noise.requires_grad_(False)
    
    y = (torch.mm(A,x_star) + noise).to(device)
    y.requires_grad_(False)
    
    z = sense_sparse(model,device, A, y, zdim)
    
    return model.decode(z).view(28, 28).detach().cpu().numpy()