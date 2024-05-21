

import torch
import torch.nn as nn


# -- Sketch
class DeeperClusterV2(nn.Module):
    def __init__(self, vit_encoder=nn.Module):
        super().__init__()
        self.vit_encoder = vit_encoder
        self.autoencoder = AutoEncoder()
        self.subclass_prediction_layer = nn.Linear(self.vit_encoder.embed_dim, 5) # TODO: confirm.


    # -- TODO: 
    # 1 -- Feature extraction with the ViT encoder.
    #
    # 2 -- Reduce feature dimensionality with the autoencoder.
    # - 2.1 - Input the autoencoder bottleneck features into K-means. 
    # - 2.2 - Compute reconstruction loss (L2 norm) between masked input and its reconstruction.
    # - 2.3 - Add Clustering penalty term to the reconstruction loss (K-means distances). 
    # - 2.4 - Backpropagate.
    # 
    # 3 -- Classification: 
    # - 3.1 - Compute Cluster assignments.
    # - 3.2 - Predict Parent and sub-class (K-means assignment).
    # - 3.3 - Compute loss with respect to the predictions.  
    # - 3.4 - Backpropagate. 

    def forward(self, x):
        inp = self.vit_encoder.forward_features(x)
        reconstructed_input, bottleneck_output = self.autoencoder(inp)
        
# TODO: Replace leaky with gelus.
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1024, 768),
            nn.GELU(),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(256, 512),
            nn.GELU(),
            torch.nn.Linear(512, 768),
            nn.GELU(),
            torch.nn.Linear(768, 1024), 
            torch.nn.Sigmoid() # Verify this
        )

        self.drop_rate = 0.25
        self.dropout = nn.Dropout(self.drop_rate)

    def forward(self, x):
        x = self.dropout(x) # Mask input    
        bottleneck_output = self.encoder(x)
        reconstructed_input = self.decoder(bottleneck_output)
        return reconstructed_input, bottleneck_output


