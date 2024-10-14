import torch 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = "../Data/"
N = 100     # Nb of tokens in each piece