import torch

import faiss
import faiss.contrib.torch_utils



'''
    Things to consider...
    Two stages are going to have to be considered in this approach: (1) - Dataset iteration and (2) - End of epoch.
    --
        (1) - During the dataset iteration we will assign images to the nearest centroid to the cluster
        correspondent to the class that an image belongs to. 
        
        This also mean that we'd probably lose batch speedup due to performing assignment according to each image label
        on the batch.

        (2) - On the end of an epoch we will perform the update step until convergence.

            To do this we will therefore have to cache the features compressed by the autoencoder. 
            There is also concernings on how to implement this in a distributed way, and other things that
            we should have to test in order to check if this could work somehow.

            For example, there may be problems with initializing an array of faiss.Kmeans objects, considering the info below: 

                - "All GPU indexes are built with a StandardGpuResources object (which is an implementation of the abstract class GpuResources). 
                The resource object contains needed resources for each GPU in use, including an allocation of temporary scratch space 
                (by default, about 2 GB on a 12 GB GPU), cuBLAS handles and CUDA streams." 

                See: https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU?fbclid=IwZXh0bgNhZW0CMTAAAR3Sfc4jQp3RZEUKUdAOsXcfc0OlhlLlKl2kxPQJG-pc8B71FPL7YBHqQRg_aem_AYSDKlDdlVDpSNmVp-hKCajFVRnw_n2S-OiYzkGBvRxNHJJi-PsdO4CKw0EoF4ou5fmk1WzM1-AIbw9ZrOFSMT8V
                        
    --
        - Questions:
            - How often do we call the update step In terms of frequency of epochs? 
            - Should we use the same matrix across all epochs or restart the training on the basis of an x number of epochs?      

        - TODO (Ideas/Ablation):
            - Implement other stopping criteria (e.g., inertia).
            - Keep track of the average no. of iterations to converge and log this.
            - Keep track of the classes that are presenting empty clusters (log as well). 
            - How frequently should we perform E on the k-means? 
                I sense that we could use a scheduler to increase the amount of epochs before performing E
                e.g., start with 1 and double on every 10 epochs;
                e.g., start with 1 for 10 epochs then double it on every 5. 


'''
class KMeansModule:
    
    def __init__(self, nb_classes, dimensionality=256, n_iter=10, k=5, max_iter=300):

        # Create the K-means object
        self.k = k
        self.d = dimensionality
        self.n_iter = n_iter
        self.n_kmeans = [faiss.Kmeans(d=dimensionality, k=k, niter=1, gpu=True, verbose=True) for _ in nb_classes]         


    def assign(self, x, y):
        #D_batch = torch.empty()
        #I_batch = torch.empty()
        for i in range(len(x)):
            # Train K-means model for one iteration to initialize centroids
            self.n_kmeans[y[i]].train(x[i])
            # Assign vectors to the nearest cluster centroid
            D, I = self.n_kmeans[y[i]].index.search(x[i], 1)

            D_batch = torch.cat((D_batch, D))
            I_batch = torch.cat((I_batch, I))
        return D_batch, I_batch

    def update(self, features):
        return 0


    # TODO: ensure that xb have been previously sent to a device 
    # What happens when we send a tensor to a device? []
    def iterative_kmeans(self, xb, device):
                
        for _ in range(self.n_iter - 1):  # n_iter-1 because we already did one iteration
            # Assign vectors to the nearest cluster centroid
            D, I = self.n_kmeans.index.search(xb, 1) # TODO: what is the shape of I[]
             
            # Initialize tensors to store new centroids and counts
            new_centroids = torch.zeros((self.k, self.d), dtype=torch.float32, device=device)
            counts = torch.zeros(self.k, dtype=torch.int64, device=device)

            # TODO: check if this assumes that xb is a batch
            # Sum up all points in each cluster
            for i in range(len(xb)):
                cluster_id = I[i][0] 
                new_centroids[cluster_id] += xb[i] # torch.from_numpy(xb[i]).to(device)
                counts[cluster_id] += 1

            # Compute the mean for each cluster
            for j in range(self.k):
                if counts[j] > 0:
                    new_centroids[j] /= counts[j]

            # Convert PyTorch tensor to NumPy array
            new_centroids_np = new_centroids.cpu().numpy()

            # Update the centroids in the FAISS index
            self.n_kmeans.centroids = new_centroids_np
            self.n_kmeans.index.reset()
            self.n_kmeans.index.add(new_centroids_np)
        
        # TODO: Verify the shape of D, and whether we should return the final value or its mean across the iterations. 
        return self.n_kmeans, D
