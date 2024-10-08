import torch
import torch.distributed as dist

import faiss
import faiss.contrib.torch_utils

import numpy as np
np.random.seed(0) # TODO: modify

from src.utils.logging import (
    CSVLogger,
    AverageMeter)


'''
        - TODO (Ideas/Ablation):
            - Keep track of the average no. of iterations to converge and log this.
            - Keep track of the classes that are presenting empty clusters (log as well). 
            - How frequently should we perform E on the k-means? 
                I sense that we could use a scheduler to increase the amount of epochs before performing E
                e.g., start with 1 and double on every 10 epochs;
                e.g., start with 1 for 10 epochs then double it on every 5. 
'''
class KMeansModule:

    def __init__(self, nb_classes, dimensionality=256, n_iter=300, tol=1e-4, k_range=[2,3,4,5], resources=None, config=None):
        
        self.resources = resources
        self.config = config

        self.k_range = k_range
        self.d = dimensionality
        self.max_iter = n_iter
        self.tol = tol
        
        # Create the K-means object
        if len(k_range) == 1:
            self.n_kmeans = [faiss.Kmeans(d=dimensionality, k=k_range[0], niter=1, verbose=True, min_points_per_centroid = 1 ) for _ in range(nb_classes)]   
        else:
            self.n_kmeans = []   
            for _ in range(nb_classes):
                self.n_kmeans.append([faiss.Kmeans(d=dimensionality, k=k, niter=1, verbose=False, min_points_per_centroid = 1) for k in k_range])                                                            

    def my_index_cpu_to_gpu_multiple(self, resources, index, co=None, gpu_nos=None):
        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()
        if gpu_nos is None: 
            gpu_nos = range(len(resources))
        for i, res in zip(gpu_nos, resources):
            vdev.push_back(i)
            vres.push_back(res)
        index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)
        index.referenced_objects = resources
        return index
    
    def CosineClusterIndex(self, xb, yb, cached_features):
        return 0

    def initialize_centroids(self, batch_x, class_id, resources, rank, device, config, cached_features):
        def augment(x, n_samples): # Built as a helper function for development TODO: remove
            # Workaround to handle faiss centroid initialization with a single sample.
            # We built upon Mathilde Caron's idea of adding perturbations to the data, but we do it randomly instead.
            augmented_data = x.repeat((n_samples, 1))
            for i in range((n_samples)):
                sign = (torch.randint(0, 3, size=(self.d,)) - 1)
                sign = sign.to(device=device, dtype=torch.float32)
                eps = torch.tensor(1e-7, dtype=torch.float32, device=device)   
                augmented_data[i] += sign * eps                
            return augmented_data 
        
        if cached_features is None:
            print('Augment shouldnt be used, exitting...')
            exit(0)
            batch_x = augment(batch_x, self.k_range[len(self.k_range)-1]) # Create additional synthetic points to meet the minimum requirement for the number of clusters.             
        else:
            image_list = cached_features[class_id] # Otherwise use the features cached from the previous epoch                
            batch_x = torch.stack(image_list)
        for k in range(len(self.k_range)):
            self.n_kmeans[class_id][k].train(batch_x.detach().cpu()) # Then train K-means model for one iteration to initialize centroids

            #gpu_index_flat = faiss.GpuIndexFlatL2(resources, self.d, config)
            #gpu_index_flat = faiss.index_cpu_to_gpu_multiple(resources, devices=[rank],index=index_flat)
            #gpu_index_flat = faiss.GpuIndexFlatL2(resources[rank], self.d, config[rank])
            #gpu_index_flat = self.my_index_cpu_to_gpu_multiple(resources, index_flat, gpu_nos=[0,1,2,3,4,5,6,7])
            
            # Replace the regular index by a gpu one            
            index_flat = self.n_kmeans[class_id][k].index
            gpu_index_flat = faiss.index_cpu_to_gpu(self.resources, rank, index_flat)
            self.n_kmeans[class_id][k].index = gpu_index_flat
            
    
    def init(self, resources, rank, device, config, cached_features):
        # Initialize the centroids for each class
        for key in cached_features.keys():
            self.initialize_centroids(batch_x=None,
                                      class_id=key,
                                      resources=resources,
                                      rank=rank,
                                      device=device,
                                      config=config,
                                      cached_features=cached_features)             
    '''
        Assigns a single data point to the set of clusters correspondent to the y target.

        Centroid initialization is going to be a problem in the first epoch because we wouldn't have the features cached yet, after that we can use the previous epoch's
        cache to initialize the centroids appropriately. Despite that, initialization is a problem for the first epochs despite of everything. That's because we are 
        using the reduced dimension features provided by an autoencoder that is going to be simultaneously trained to reduce the dimensionality of the features 
        provided by a ViT encoder that haven't properly learned good features yet. Therefore in the first few epochs, the initialization is expected to be very poor.
        
        Other than that we would still could have residual problems due to this initialization process.
        Because of that, another regularization mechanism that we could do to circumvent this is to reset the K-means
        centroids after every N epochs. (Another ablation parameter).  

    '''
    def assign(self, x, y, resources, rank, device, cached_features=None):
        D_batch = []
        I_batch = []        
        for i in range(len(x)):
            # Expand dims
            batch_x = x[i].unsqueeze(0)
            # Initialize the centroids if it haven't already been initialized
            if self.n_kmeans[y[i]][0].centroids is None:
                # FIXME this won't work inside the training loop because inside it batch_x is not on cpu but since this isn't expected to be called should be safe for now
                self.initialize_centroids(batch_x, y[i].item(), resources, rank, device, cached_features)                
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

    def update(self, cached_features, device, empty_clusters_per_epoch, empty_clusters_per_k_per_epoch):
        means = [[] for k in self.k_range]
        for key in cached_features.keys():
            xb = torch.stack(cached_features[key]) # Form an image batch
            if xb.get_device() == -1:
                xb = xb.to(device, dtype=torch.float32)
            _, batch_k_means_loss = self.iterative_kmeans(xb, key, device, empty_clusters_per_epoch, empty_clusters_per_k_per_epoch) # TODO: sum and average across dataset length
            # log_loss.update(batch_k_means_loss.mean())
            # For each "batch" append the losses (average distances) for each K value.
            for k in range(len(self.k_range)):
                means[k].append(batch_k_means_loss[k])
        # Compute the average loss for each value of K in k_range across all data points 
        losses = []
        for k in range(len(self.k_range)):
            stack = torch.stack(means[k])
            losses.append(stack.mean())
        return losses, empty_clusters_per_epoch, empty_clusters_per_k_per_epoch

    def iterative_kmeans(self, xb, class_index, device, empty_clusters_per_epoch, empty_clusters_per_k_per_epoch):
        empty_clusters = []
        D_per_K_value = torch.zeros(len(self.k_range)) # e.g., [2, 3, 4, 5]
        for k in range(len(self.k_range)):
            K = self.k_range[k]
            previous_inertia = 0
            non_empty = []
            # n_iter-1 because we already did one iteration    
            for itr in range(self.max_iter - 1):  
                # Compute the assignments
                D, I = self.n_kmeans[class_index][k].index.search(xb, 1)
            
                inertia = torch.sum(D)
                if not previous_inertia == 0 and abs(previous_inertia - inertia) < self.tol:
                    D_per_K_value[k] = torch.mean(D)
                    break
                previous_inertia = inertia

                new_centroids = []
                counts = []
                
                # Initialize tensors to store new centroids and counts
                new_centroids = torch.zeros((K, self.d), dtype=torch.float32, device=device)
                counts = torch.zeros(K, dtype=torch.int64, device=device)

                # Sum up all points in each cluster
                for i in range(len(xb)):
                    cluster_id = I[i][0].item()
                    new_centroids[cluster_id] += xb[i] 
                    counts[cluster_id] += 1

                # Compute the mean for each cluster
                for j in range(K):
                    if counts[j] > 0:
                        new_centroids[j] /= counts[j]
                        non_empty.append((k,j))
                    else:
                        sign = (torch.randint(0, 3, size=(self.d,), device=device) - 1)
                        eps = torch.tensor(1e-7, dtype=torch.float32, device=device)   
                        op = sign * eps 
                        if len(non_empty) > 0:                        
                            if len(non_empty) > 1:
                                idx = np.random.randint(0, len(non_empty)) 
                            else:
                                idx = 0
                            cluster_id = non_empty[idx][1] # choose a random cluster from the set of non empty clusters
                            non_empty_cluster = new_centroids[cluster_id]
                            new_centroids[j] = torch.clone(non_empty_cluster)
                            # Replace empty centroid by a non empty one with a perturbation
                            new_centroids[j] += op 
                            non_empty_cluster -= op
                            non_empty.append((k,j))
                            empty_clusters.append(K)
                        else:
                            # If the first centroid is empty, select one randomly hoping that it is not empty. 
                            if ((j+1) <  (K-1)):
                                idx = np.random.randint(j+1, K)
                            else:
                                idx = j+1
                            non_empty_cluster = self.n_kmeans[class_index][k].centroids[idx]
                            new_centroids[j] = torch.clone(non_empty_cluster)
                            # Replace empty centroid by a possibly non-empty one with a perturbation
                            new_centroids[j] += op 
                            non_empty_cluster -= op
                            non_empty.append((k,j))
                            empty_clusters.append(K)                        
                # Update the centroids in the FAISS index
                self.n_kmeans[class_index][k].centroids = new_centroids
                self.n_kmeans[class_index][k].index.reset()
                self.n_kmeans[class_index][k].index.add(new_centroids)    
                D_per_K_value[k] = torch.mean(D)
        if len(empty_clusters) > 0:
            empty_clusters_per_epoch.update(len(empty_clusters))
            for entry in empty_clusters:
                empty_clusters_per_k_per_epoch[(entry - 2)].update(1)
            # TODO: remove
            #print('Empty Clusters per k value for class id:', class_index)
            #print(empty_clusters)
            #print('################# ################# #################')
        return self.n_kmeans, D_per_K_value