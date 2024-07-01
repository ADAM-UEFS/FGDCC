
import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.GELU(),
            nn.Linear(1024, 768),
            nn.GELU(),
            nn.Linear(768, 384),
            nn.GELU(),
            #nn.Linear(512, 256),
            #nn.GELU(),
        )

        self.decoder = torch.nn.Sequential(
            #torch.nn.Linear(256, 512),
            #nn.GELU(),
            torch.nn.Linear(384, 768),
            nn.GELU(),
            torch.nn.Linear(768, 1024),
            nn.GELU(),
            torch.nn.Linear(1024, 1280), 
        )

        self.drop_rate = 0.25
        self.dropout = nn.Dropout(self.drop_rate)

        # TODO: we could use "smooth L1" loss, and therefore ignore the activation.
        self.out = torch.nn.Sigmoid() # TODO: review if this is necessary depending on the loss.
    
    def forward(self, x):

        x = self.dropout(x) # Mask input    
        bottleneck_output = self.encoder(x)
        bottleneck_output = F.layer_norm(bottleneck_output, (bottleneck_output.size(-1),))  # normalize over feature-dim 

        reconstructed_input = self.decoder(bottleneck_output)
        reconstructed_input = F.layer_norm(reconstructed_input, (reconstructed_input.size(-1),))  # normalize over feature-dim 

        return reconstructed_input, bottleneck_output


def vanilla_autoencoder():
    return AutoEncoder()
