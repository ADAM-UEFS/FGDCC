import torch


import sys
 
# setting path
sys.path.append('../')

from src.KMeans import KMeansModule


def main():
    batch_size = 64
    input_dim = 256
    nb_classes = 1000

    kmeans = KMeansModule(nb_classes = nb_classes, k=5)

    x = torch.randn(batch_size, input_dim, dtype=torch.float32)  # Batch of 64 samples, ensure faiss float32    
    y = torch.randint(low=0, high=(input_dim-1), size=[batch_size])    
    print(x[0].size())
    print(y.size())

    D,I = kmeans.assign(x,y)
    print(D.size())
    print(I.size())
    
main()

