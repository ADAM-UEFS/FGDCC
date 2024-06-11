import torch

import sys
 
# setting path
sys.path.append('../')

from src.helper import HierarchicalClassifier

def main():
    batch_size = 64
    input_dim = 1280
    nb_classes = 1000

    x = torch.randn(batch_size, input_dim)  # Batch of 64 samples
    print('x Size: ',x.size())
    model = HierarchicalClassifier(input_dim=1280, num_parents=nb_classes,  num_children_per_parent=[2,3,4,5])
    parent_logits, children_logits = model(x)
    print(parent_logits.size())
    print(j.size() for j in children_logits)

main()

