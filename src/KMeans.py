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

    # TODO (1): Adapt to handle a range of classes e.g., K=[2,3,4,5] and test.
    # (2) Build a test case to integrate to the hierarchical classification module.
    # (3) GPU testing?. 

    def __init__(self, nb_classes, dimensionality=256, n_iter=10, k=5, max_iter=300):

        # Create the K-means object
        self.k = k
        self.d = dimensionality
        self.n_iter = n_iter
        self.n_kmeans = [faiss.Kmeans(d=dimensionality, k=k, niter=1, verbose=True, min_points_per_centroid = 1 ) for _ in range(nb_classes)]         

    '''
        Assigns a single data point to the set of clusters correspondent to the y target.
        As faiss prevents us from assigning a single data point, this approach ensures 
        the data meets the necessary size requirements and prevent us from
        taking the risk of modifying the library's code. 

        What is the problem within the assignment step?

        The pipeline proposal works as following: 
            (1) - ViT encoding (feature extraction)
            (2) - Non-linear dimensionality reduction (Autoencoding)
            (3) - Compute assignments to obtain the subclass labels 
            (4) - Compute the autoencoder reconstruction loss 
            (5) - Compute the classification losses. 

        Considering this we have the following problem: 
        In order to compute the cluster assignments we have to have the initialized centroids at first, that is because k-means generally uses the own data points
        to provide the locations for the centroids. This is a problem because we need at least n >= k data points to initialize the centroids and our data comes 
        in batches of size 32 < n < 128. 

        Centroid initialization is going to be a problem in the first epoch because we wouldn't have the features cached yet, after that we can use the previous epoch's
        cache to initialize the centroids appropriately. Despite that, initialization is a problem for the first epochs despite of everything. That's because we are 
        using the reduced dimension features provided by an autoencoder that is going to be simultaneously trained to reduce the dimensionality of the features 
        provided by a ViT encoder that haven't properly learned good features yet. Therefore in the first few epochs, the initialization is expected to be very poor.

        Therefore the proposed solution is:

        For a pair (x, y)

            If first epoch do:
                - If the corresponding centroids has been initialized do:
                    - Compute the assignment and concatenate over the batch dimension.
                - Else:
                    - Initialize centroids from x with random perturbations. TODO: here we can compute the std for each dimension and draw a random amount of stds to multiply and add to each dimension. 
            Else          
                - Initialize the centroids for all the K-means from the features cached from the previous epoch.
                
            TODO: keep two caches in memory, for the current and last epochs.             
        
        Other than that we would still could have residual problems due to this initialization process.
        Because of that, another regularization mechanism that we could do to circumvent this is to reset the K-means
        centroids after every N epochs. (Another ablation parameter).  

        This adds to the regularization strategies that we are building to avoid trivial solutions i.e., representational collapse and ablation parameters. 

        (1) - Replace emtpy centroids by non empty ones with a perturbation.
        (2) - Train K-means after every T epochs e.g., 2 or 3 (this gives the encoders some space to refine their features). 
        (3) - Reset the K-means centroids after every N epochs. I

    '''
    # TODO: 
    # If first epoch:
    #   If centroid is initialized just compute assignments.
    #   If not initialized do initialization:
    #       Compute the stds for each dimension, draw a random number to multiply the std with, replace.
    # 
    # PS: This have to be performed for each sample in the batch because it could be the case that a previous sample in the batch
    # already happened to have been used to initialize the centroid. TODO: Attention.
    #
    # Receive as input a cache parameter (check if not empty). TODO: structure the cache dictionary. 
    # Adjust the process to handle multiple K-means [2,3,4,5, ...]
    def assign(self, x, y):
        #D_batch = torch.empty()
        #I_batch = torch.empty()
        print(x[0].size())
        print(y.size())

        # replace empty centroid by a non empty one with a perturbation
        #centroids[k] = centroids[m]
        #for j in range(args.dim_pca):
        #    sign = (j % 2) * 2 - 1;
        #    centroids[k, j] += sign * 1e-7;
        #    centroids[m, j] -= sign * 1e-7;

               
        # Workaround to handle faiss centroid initialization.           
        def augment(x): 
            augmented_data = x.repeat((self.k, 1)) # add random noise?
            return augmented_data 
        
        for i in range(len(x)):
            batch_x = x[i].unsqueeze(0) # Expand dims

            # Create additional synthetic points to meet the minimum requirement for the number of clusters.             
            augmented_data = augment(batch_x)
            
            # Train K-means model for one iteration to initialize centroids
            self.n_kmeans[y[i]].train(augmented_data)

            # Assign vectors to the nearest cluster centroid
            D, I = self.n_kmeans[y[i]].index.search(batch_x, 1)            
            print(D.size(), D[1])
            print(I.size(), I[1])
            #D_batch = torch.cat((D_batch, D))
            #I_batch = torch.cat((I_batch, I))
            
        return D_batch, I_batch # FIXME.

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
