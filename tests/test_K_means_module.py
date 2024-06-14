import torch


import sys
 
# setting path
sys.path.append('../')

from src.KMeans import KMeansModule


# TODO (1): Adapt to handle a range of classes e.g., K=[2,3,4,5] and test.
# (2) Build a test case to integrate to the hierarchical classification module.
# (3) GPU testing?. 


'''
    TODO(s):
        () - Setup the whole pipeline with classification module
        () - Adapt k-means module to handle model selection
        () - 
        
'''
def main():
    batch_size = 64
    input_dim = 256
    nb_classes = 1000

    kmeans = KMeansModule(nb_classes = nb_classes, k_range=[2,3,4,5]) # ok
    
    x = torch.randn(batch_size, input_dim, dtype=torch.float32)  # Batch of 64 samples, ensure faiss float32    
    y = torch.randint(low=0, high=(input_dim-1), size=[batch_size])    
    print(x[0].size())
    print(y.size())

    D,I = kmeans.assign(x,y)
    print(D)
    print(I)
    print(D.size())
    print(I.size())

main()

