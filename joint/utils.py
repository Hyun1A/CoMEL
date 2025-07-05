import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, accuracy_score
import torch
import torch.nn.functional as F

def seed_torch(seed=2021):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False   

@torch.no_grad()
def ema_update(model,targ_model,mm=0.9999):
    r"""Performs a momentum update of the target network's weights.
    Args:
        mm (float): Momentum used in moving average update.
    """
    assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm

    for param_q, param_k in zip(model.parameters(), targ_model.parameters()):
        param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm) # mm*k +(1-mm)*q

def patch_shuffle(x,group=0,g_idx=None,return_g_idx=False):
    b,p,n = x.size()
    ps = torch.tensor(list(range(p)))

    # padding
    H, W = int(np.ceil(np.sqrt(p))), int(np.ceil(np.sqrt(p)))
    if group > H or group<= 0:
        return group_shuffle(x,group)
    _n = -H % group
    H, W = H+_n, W+_n
    add_length = H * W - p
    # print(add_length)
    ps = torch.cat([ps,torch.tensor([-1 for i in range(add_length)])])
    # patchify
    ps = ps.reshape(shape=(group,H//group,group,W//group))
    ps = torch.einsum('hpwq->hwpq',ps)
    ps = ps.reshape(shape=(group**2,H//group,W//group))
    # shuffle
    if g_idx is None:
        g_idx = torch.randperm(ps.size(0))
    ps = ps[g_idx]
    # unpatchify
    ps = ps.reshape(shape=(group,group,H//group,W//group))
    ps = torch.einsum('hwpq->hpwq',ps)
    ps = ps.reshape(shape=(H,W))
    idx = ps[ps>=0].view(p)
    
    if return_g_idx:
        return x[:,idx.long()],g_idx
    else:
        return x[:,idx.long()]

def group_shuffle(x,group=0):
    b,p,n = x.size()
    ps = torch.tensor(list(range(p)))
    if group > 0 and group < p:
        _pad = -p % group
        ps = torch.cat([ps,torch.tensor([-1 for i in range(_pad)])])
        ps = ps.view(group,-1)
        g_idx = torch.randperm(ps.size(0))
        ps = ps[g_idx]
        idx = ps[ps>=0].view(p)
    else:
        idx = torch.randperm(p)
    return x[:,idx.long()]


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    labels = np.array(dataset.slide_label)
    label_uni = set(dataset.slide_label)
    weight_per_class = [N/len(labels[labels==c]) for c in label_uni]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.slide_label[idx]
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)

def five_scores(bag_labels, bag_predictions,sub_typing=False):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    # threshold_optimal=0.5
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label>=threshold_optimal] = 1
    this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    avg = 'macro' if sub_typing else 'binary'
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average=avg)
    accuracy = accuracy_score(bag_labels, bag_predictions)
    return accuracy, auc_value, precision, recall, fscore

def five_scores_ensemble(bag_labels, bag_predictions,sub_typing=False):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    # threshold_optimal=0.5
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label>=threshold_optimal] = 1
    this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    avg = 'macro' if sub_typing else 'binary'
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average=avg)
    accuracy = accuracy_score(bag_labels, bag_predictions)
    return accuracy, auc_value, precision, recall, fscore

def five_scores_and_seg_scores(bag_labels, bag_probs, bag_labels_patch, bag_probs_patch, sub_typing=False, return_pred_patch=False, num_classes=2, labels=[0,1]):
        
    labels.sort()
    n_classes = labels[-1]+1

    ##################################
    ##### for bag classification #####

    bag_probs = np.array(bag_probs)
    bag_labels_one_hot = F.one_hot(torch.cat(bag_labels,dim=0), num_classes=n_classes).numpy()

    auc_value = roc_auc_score(bag_labels_one_hot, bag_probs, multi_class='ovr', average='macro')
    bag_predictions = bag_probs.argmax(axis=1)
    bag_predictions_one_hot = F.one_hot(torch.tensor(bag_predictions).to(torch.long), num_classes=n_classes).numpy()
    avg= 'macro'
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels_one_hot, bag_predictions_one_hot, average=avg)
    accuracy = accuracy_score(bag_labels, bag_predictions)

    ##### for bag classification #####
    ##################################


    ###################################
    ##### for inst classification #####

    ##### auc_inst, acc_inst, fscore_inst

    bag_labels_patch = [label_patch % 2 for label_patch in bag_labels_patch]
    bag_labels_patch_cat = torch.cat(bag_labels_patch, dim=1)[0,:,0].numpy()
    bag_probs_patch_cat = torch.cat(bag_probs_patch,dim=0).numpy()

    fpr_patch, tpr_patch, threshold_patch = roc_curve(bag_labels_patch_cat, bag_probs_patch_cat[:,1], pos_label=1)
    fpr_optimal_patch, tpr_optimal_patch, threshold_optimal_patch = optimal_thresh(fpr_patch, tpr_patch, threshold_patch)

    auc_patch = roc_auc_score(bag_labels_patch_cat, bag_probs_patch_cat[:,1])


    tmp_preds = bag_probs_patch_cat[:,1]
    tmp_preds[tmp_preds>=threshold_optimal_patch] = 1
    tmp_preds[tmp_preds<threshold_optimal_patch] = 0
    bag_predictions_patch = tmp_preds

    avg = 'binary'
    _, _, fscore_patch, _ = precision_recall_fscore_support(bag_labels_patch_cat, bag_predictions_patch, average=avg)
    acc_patch = accuracy_score(bag_labels_patch_cat, bag_predictions_patch)

    bag_predictions_patch = []
    bag_predictions_patch_one_hot = []
    for i in range(len(bag_probs_patch)):
        pred_patch = (bag_probs_patch[i][:,1] > threshold_optimal_patch).long()
        bag_predictions_patch.append(pred_patch)
        pred_patch_one_hot = F.one_hot(pred_patch, num_classes=n_classes).numpy()
        bag_predictions_patch_one_hot.append(pred_patch_one_hot)

    count_tumor_slide, mioU, dice = compute_seg_scores(bag_labels, bag_predictions, bag_labels_patch, bag_predictions_patch)

    ##### for inst classification #####
    ###################################


    if return_pred_patch:
        return accuracy, auc_value, precision, recall, fscore, count_tumor_slide, \
                mioU, dice, acc_patch, auc_patch, fscore_patch, threshold_optimal_patch.item(), bag_predictions_patch
    else:
        return accuracy, auc_value, precision, recall, fscore, count_tumor_slide, \
                mioU, dice, acc_patch, auc_patch, fscore_patch, threshold_optimal_patch.item()



def compute_seg_scores(bag_labels, bag_predictions, bag_labels_patch, bag_predictions_patch, labels=[0,1]):
        
    count_tumor_slide = 0
    tumor_slide_idx = []
    tumor_slide_true_area = []
    tumor_slide_pred_area = []
    tumor_slide_overlap_area = []
    tumor_slide_miou = []
    tumor_slide_dice = []
        
    labels.sort()
    base_label = labels[-1]


    for idx, (bag_label, bag_prediction, bag_label_patch, bag_prediction_patch) in enumerate(zip(bag_labels, bag_predictions, bag_labels_patch, bag_predictions_patch)):
                
        if bag_label == 0:
            continue
        
        count_tumor_slide += 1
        tumor_slide_idx.append(idx)
        
        #########################################
        #### compute IoU and Dice for a WSI #####            
        
        true_tumor_region = (bag_label_patch==base_label).squeeze(0).squeeze(-1).float()       
        pred_tumor_region = (bag_prediction_patch==base_label).float()         
        
        overlap_region = (torch.all(torch.cat([true_tumor_region.unsqueeze(1)==1, \
                                                pred_tumor_region.unsqueeze(1)==1], dim=1), dim=1)).float()

        true_tumor_area = true_tumor_region.sum()
        pred_tumor_area = pred_tumor_region.sum()
        overlap_area = overlap_region.sum()

        tumor_slide_true_area.append(true_tumor_area)
        tumor_slide_pred_area.append(pred_tumor_area)
        tumor_slide_overlap_area.append(overlap_area)

        
        sample_iou = overlap_area / (true_tumor_area + pred_tumor_area - overlap_area+1e-5)            
        sample_dice = 2*overlap_area / (true_tumor_area + pred_tumor_area+1e-5)
        
        tumor_slide_miou.append(sample_iou)
        tumor_slide_dice.append(sample_dice)
        #### compute IoU and Dice for a WSI #####            
        #########################################
        
    miou = torch.tensor(tumor_slide_miou).mean().numpy()
    dice = torch.tensor(tumor_slide_dice).mean().numpy()
                
    return count_tumor_slide, miou, dice





def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False,save_best_model_stage=0.):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.save_best_model_stage = save_best_model_stage

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):
        
        score = -val_loss if epoch >= self.save_best_model_stage else 0.

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def state_dict(self):
        return {
            'patience': self.patience,
            'stop_epoch': self.stop_epoch,
            'verbose': self.verbose,
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'val_loss_min': self.val_loss_min
        }
    def load_state_dict(self,dict):
        self.patience = dict['patience']
        self.stop_epoch = dict['stop_epoch']
        self.verbose = dict['verbose']
        self.counter = dict['counter']
        self.best_score = dict['best_score']
        self.early_stop = dict['early_stop']
        self.val_loss_min = dict['val_loss_min']

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def balanced_kmeans_torch(X, n_clusters, n_samples_per_cluster, max_iter=10):
    """
    Balanced K-Means clustering using PyTorch tensors.
    
    Each cluster will have exactly n_samples_per_cluster points.
    
    Parameters:
        X (torch.Tensor): Input data of shape (N, D) where N = n_clusters * n_samples_per_cluster.
        n_clusters (int): Number of clusters.
        n_samples_per_cluster (int): Fixed number of samples per cluster.
        max_iter (int): Maximum number of iterations.
    
    Returns:
        assignments (torch.Tensor): Tensor of shape (N,) with cluster assignments (0 ~ n_clusters-1).
        centroids (torch.Tensor): Tensor of shape (n_clusters, D) with final cluster centers.
    """
    N, D = X.shape
    if N != n_clusters * n_samples_per_cluster:
        raise ValueError("Total number of samples must be exactly n_clusters * n_samples_per_cluster")
    
    device = X.device

    # 1. Initialize centroids by randomly picking n_clusters samples from X.
    rand_indices = torch.randperm(N, device=device)[:n_clusters]
    centroids = X[rand_indices].clone()  # shape: (n_clusters, D)
    
    assignments = torch.full((N,), -1, dtype=torch.long, device=device)
    
    for it in range(max_iter):
        # 2. Compute distances between each point and each centroid.
        distances = torch.cdist(X, centroids, p=2)
        
        # 3. Greedy assignment:
        flat_distances = distances.view(-1)
        sorted_costs, sorted_indices = torch.sort(flat_distances)
        
        sample_indices = sorted_indices // n_clusters  # integer division
        cluster_indices = sorted_indices % n_clusters
        
        assigned = torch.zeros(N, dtype=torch.bool, device=device)
        cluster_counts = torch.zeros(n_clusters, dtype=torch.long, device=device)
        new_assignments = torch.full((N,), -1, dtype=torch.long, device=device)

        for idx in range(sorted_indices.numel()):
            i = int(sample_indices[idx].item())
            k = int(cluster_indices[idx].item())
            if (not assigned[i]) and (cluster_counts[k] < n_samples_per_cluster):
                new_assignments[i] = k
                assigned[i] = True
                cluster_counts[k] += 1
                if assigned.all():
                    break
        
        if torch.equal(new_assignments, assignments):
            print(f"Converged at iteration {it}")
            break
        
        assignments = new_assignments
        
        new_centroids = torch.zeros_like(centroids)
        for k in range(n_clusters):
            idxs = (assignments == k).nonzero(as_tuple=True)[0]
            if idxs.numel() > 0:
                new_centroids[k] = X[idxs].mean(dim=0)
            else:
                new_centroids[k] = X[torch.randint(0, N, (1,), device=device)]
        centroids = new_centroids.clone()
        
    return assignments, centroids
