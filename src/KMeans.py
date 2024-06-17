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

    # TODO: nb_classes has to be the class idx_map. 
    # TODO: Verify what pytorch uses. 
    def __init__(self, nb_classes, dimensionality=256, n_iter=10, k_range=[2,3,4,5], resources=None, config=None, max_iter=300):

        self.resources = resources
        self.config = config        
        
        # Create the K-means object
        self.k_range = k_range
        self.d = dimensionality
        self.n_iter = n_iter
        
        if len(k_range) == 1:
            self.n_kmeans = [faiss.Kmeans(d=dimensionality, k=k_range[0], niter=1, verbose=True, min_points_per_centroid = 1 ) for _ in range(nb_classes)]   
        else:
            self.n_kmeans = []   
            for _ in range(nb_classes):
                self.n_kmeans.append([faiss.Kmeans(d=dimensionality, k=k, niter=1, verbose=True, min_points_per_centroid = 1) for k in k_range])
            
#            for cls in range(nb_classes):
#                for k in range(len(k_range)):
                    #faiss.GpuIndexFlatL2(self.resources, self.d, self.config)
#                    index_flat = self.n_kmeans[cls][k].index
#                    gpu_index_flat = faiss.index_cpu_to_gpu(resources, config.device, index_flat)
#                    self.n_kmeans[cls][k].index = gpu_index_flat
                                                            

        
    '''
        Assigns a single data point to the set of clusters correspondent to the y target.
        As faiss prevents us from assigning a single data point, this approach ensures 
        the data meets the necessary size requirements and prevent us from
        taking the risk of modifying the library's code. 

        The problem with the assignment step is that in order to compute the cluster assignments we have to have the initialized centroids at first, that is 
        because k-means generally uses the own data points to provide the locations for the centroids. This is a problem because we need at least n >= k 
        data points to initialize the centroids and our data comes in batches of size 32 < n < 128. 

        Centroid initialization is going to be a problem in the first epoch because we wouldn't have the features cached yet, after that we can use the previous epoch's
        cache to initialize the centroids appropriately. Despite that, initialization is a problem for the first epochs despite of everything. That's because we are 
        using the reduced dimension features provided by an autoencoder that is going to be simultaneously trained to reduce the dimensionality of the features 
        provided by a ViT encoder that haven't properly learned good features yet. Therefore in the first few epochs, the initialization is expected to be very poor.
        
        Other than that we would still could have residual problems due to this initialization process.
        Because of that, another regularization mechanism that we could do to circumvent this is to reset the K-means
        centroids after every N epochs. (Another ablation parameter).  

        This adds to the regularization strategies that we are building to avoid trivial solutions i.e., representational collapse and ablation parameters. 

        (1) - Replace emtpy centroids by non empty ones with a perturbation.
        (2) - Train K-means after every T epochs e.g., 2 or 3 (this gives the encoders some space to refine their features). 
        (3) - Reset the K-means centroids after every N epochs (couldn't be deterministic)

    '''
    def assign(self, x, y, resources, rank, device, cached_features=None):

        def augment(x, n_samples):
            # Workaround to handle faiss centroid initialization with a single sample.
            # We built upon Mathilde Caron's idea of adding perturbations to the data, but we do it randomly instead.
            augmented_data = x.repeat((n_samples, 1))
            for i in range((n_samples)):
                sign = (torch.randint(0, 3, size=(self.d,)) - 1)
                sign = sign.to(device=device, dtype=torch.float32)
                eps = torch.tensor(1e-7, dtype=torch.float32, device=device)   
                augmented_data[i] += sign * eps                
            return augmented_data 

        def initialize_centroids(batch_x, class_id):
            # If first epoch, augment the datapoint then initialize
            if cached_features is None:
                batch_x = augment(batch_x, self.k_range[len(self.k_range)-1]) # Create additional synthetic points to meet the minimum requirement for the number of clusters.             
            else:
                image_list = cached_features[class_id] # Otherwise use the features cached from the previous epoch                
                batch_x = torch.stack(image_list)
            for k in range(len(self.k_range)):
                self.n_kmeans[class_id][k].train(batch_x.detach().cpu()) # Then train K-means model for one iteration to initialize centroids
                # Replace the regular index by a gpu one
                index_flat = self.n_kmeans[class_id][k].index
                gpu_index_flat = faiss.index_cpu_to_gpu(resources, rank, index_flat)
                self.n_kmeans[class_id][k].index = gpu_index_flat

        D_batch = []
        I_batch = []        
        for i in range(len(x)):
            # Expand dims
            batch_x = x[i].unsqueeze(0)
            # Initialize the centroids if it haven't already been initialized
            if self.n_kmeans[y[i]][0].centroids is None:
                initialize_centroids(batch_x, y[i])
            # Assign the vectors to the nearest centroid
            D_k, I_k = [], []
            for k in range(len(self.k_range)):
                D, I = self.n_kmeans[y[i]][k].index.search(x[i].unsqueeze(0), 1)
                D_k.append(D[0])
                I_k.append(I[0])            
            D_batch.append(torch.stack(D_k))
            I_batch.append(torch.stack(I_k))
        D_batch = torch.stack((D_batch))
        I_batch = torch.stack((I_batch))
        return D_batch, I_batch


    def update(self, cached_features):
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
