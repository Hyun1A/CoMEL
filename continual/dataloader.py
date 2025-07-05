import os
import csv
import torch
import random
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
import h5py
import math
import itertools, bisect


def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

def get_patient_label(csv_file):
    patients_list=[]
    labels_list=[]
    label_file = readCSV(csv_file)
    for i in range(0, len(label_file)):
        patients_list.append(label_file[i][0])
        labels_list.append(label_file[i][1])
    a=Counter(labels_list)
    print("patient_len:{} label_len:{}".format(len(patients_list), len(labels_list)))
    print("all_counter:{}".format(dict(a)))
    return np.array(patients_list,dtype=object), np.array(labels_list,dtype=object)

def data_split(full_list, ratio, shuffle=True,label=None,label_balance_val=True):
    """
    dataset split: split the full_list randomly into two sublist (val-set and train-set) based on the ratio
    :param full_list: 
    :param ratio:     
    :param shuffle:  
    """
    # select the val-set based on the label ratio
    if label_balance_val and label is not None:
        _label = label[full_list]
        _label_uni = np.unique(_label)
        sublist_1 = []
        sublist_2 = []

        for _l in _label_uni:
            _list = full_list[_label == _l]
            n_total = len(_list)
            offset = int(n_total * ratio)
            if shuffle:
                random.shuffle(_list)
            sublist_1.extend(_list[:offset])
            sublist_2.extend(_list[offset:])
    else:
        n_total = len(full_list)
        offset = int(n_total * ratio)
        if n_total == 0 or offset < 1:
            return [], full_list
        if shuffle:
            random.shuffle(full_list)
        val_set = full_list[:offset]
        train_set = full_list[offset:]

    return val_set, train_set


def get_kflod(k, patients_array, labels_array,val_ratio=False,label_balance_val=True):
    if k > 1:
        skf = StratifiedKFold(n_splits=k)
    else:
        raise NotImplementedError
    train_patients_list = []
    train_labels_list = []
    test_patients_list = []
    test_labels_list = []
    val_patients_list = []
    val_labels_list = []
    for train_index, test_index in skf.split(patients_array, labels_array):
        if val_ratio != 0.:
            val_index,train_index = data_split(train_index,val_ratio,True,labels_array,label_balance_val)
            x_val, y_val = patients_array[val_index], labels_array[val_index]
        else:
            x_val, y_val = [],[]
        x_train, x_test = patients_array[train_index], patients_array[test_index]
        y_train, y_test = labels_array[train_index], labels_array[test_index]

        train_patients_list.append(x_train)
        train_labels_list.append(y_train)
        test_patients_list.append(x_test)
        test_labels_list.append(y_test)
        val_patients_list.append(x_val)
        val_labels_list.append(y_val)
        
    # print("get_kflod.type:{}".format(type(np.array(train_patients_list))))
    return np.array(train_patients_list,dtype=object), np.array(train_labels_list,dtype=object), np.array(test_patients_list,dtype=object), np.array(test_labels_list,dtype=object),np.array(val_patients_list,dtype=object), np.array(val_labels_list,dtype=object)

def get_tcga_parser(root,cls_name,mini=False):
        x = []
        y = []

        for idx,_cls in enumerate(cls_name):
            _dir = 'mini_pt' if mini else 'pt_files'
            _files = os.listdir(os.path.join(root,_cls,'features',_dir))
            _files = [os.path.join(os.path.join(root,_cls,'features',_dir,_files[i])) for i in range(len(_files))]
            x.extend(_files)
            y.extend([idx for i in range(len(_files))])
            
        return np.array(x).flatten(),np.array(y).flatten()

class TCGADataset(Dataset):
    
    def __init__(self, file_name=None, file_label=None,max_patch=-1,root=None,persistence=True,keep_same_psize=0,is_train=False, use_h5=False, patch_labels=False, return_coords=False):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(TCGADataset, self).__init__()

        self.file_name = file_name
        self.slide_label = file_label
        self.slide_label = [int(_l) for _l in self.slide_label]

        self.class_list = list(set(self.slide_label))
        self.class_list.sort()

        self.st_cls = self.class_list[0]

        self.size = len(self.file_name)
        self.root = root
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train

        if persistence:
            self.feats = [ torch.load(os.path.join(root,'pt', _f+'.pt')) for _f in file_name ]

        self.use_h5 = use_h5
        self.patch_labels = patch_labels
        self.return_coords = return_coords


    def __len__(self):
        return self.size


    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """

        if self.persistence:
            features = self.feats[idx]
        else:
            
            if not self.use_h5:
                if not self.patch_labels:
                    if not self.return_coords:
                        dir_path = os.path.join(self.root,"pt")
                        file_path = os.path.join(dir_path, self.file_name[idx]+'.pt')
                        features = torch.load(file_path)
                        label = int(self.slide_label[idx])
                        return features , label                

                    else:
                        dir_path = os.path.join(self.root,"pt_w_coords")
                        file_path = os.path.join(dir_path, self.file_name[idx]+'.pt')
                        data = torch.load(file_path)
                        features = data["features"]
                        coords = data["coords"]
                        label = int(self.slide_label[idx])
                        return features , label, coords, self.file_name[idx]

                else:
                    if not self.return_coords:
                        dir_path = os.path.join(self.root,"pt_w_patch_labels")
                        file_path = os.path.join(dir_path, self.file_name[idx]+'.pt')
                        data = torch.load(file_path)
                        features = data["features"]
                        patch_labels = data["patch_labels"] + self.st_cls
                        label = int(self.slide_label[idx])
                        return features , label, patch_labels            

                    else:
                        dir_path = os.path.join(self.root,"pt_w_patch_labels_coords")
                        file_path = os.path.join(dir_path, self.file_name[idx]+'.pt')
                        data = torch.load(file_path)
                        features = data["features"]
                        patch_labels = data["patch_labels"] + self.st_cls
                        coords = data["coords"]
                        label = int(self.slide_label[idx])
                        return features , label, patch_labels, coords, self.file_name[idx]


            else:
                dir_path = os.path.join(self.root,"h5")
                file_path = os.path.join(dir_path, self.file_name[idx]+'.h5')
                data = h5py.File(file_path)
                features = torch.tensor(data['features'])
                patch_labels = torch.tensor(data['patch_labels']) + self.st_cls
                label = int(self.slide_label[idx])
                return features , label, patch_labels




class C16Dataset(Dataset):
    def __init__(self, file_name, file_label,root,persistence=False,keep_same_psize=0,is_train=False, use_h5=False, patch_labels=False, return_coords=False):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(C16Dataset, self).__init__()
        self.file_name = file_name
        self.slide_label = file_label
        self.slide_label = [int(_l) for _l in self.slide_label]

        self.class_list = list(set(self.slide_label))
        self.class_list.sort()

        self.st_cls = self.class_list[0]

        self.size = len(self.file_name)
        self.root = root
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train

        if persistence:
            self.feats = [ torch.load(os.path.join(root,'pt', _f+'.pt')) for _f in file_name ]

        self.use_h5 = use_h5
        self.patch_labels = patch_labels
        self.return_coords = return_coords

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        if self.persistence:
            features = self.feats[idx]
        else:
            
            if not self.use_h5:
                if not self.patch_labels:
                    if not self.return_coords:
                        dir_path = os.path.join(self.root,"pt")
                        file_path = os.path.join(dir_path, self.file_name[idx]+'.pt')
                        features = torch.load(file_path)
                        label = int(self.slide_label[idx])
                        return features, label                

                    else:
                        dir_path = os.path.join(self.root,"pt_w_coords")
                        file_path = os.path.join(dir_path, self.file_name[idx]+'.pt')
                        data = torch.load(file_path)
                        features = data["features"]
                        coords = data["coords"]
                        label = int(self.slide_label[idx])



                        if hasattr(self, "logits"):
                            return features , label, coords, self.file_name[idx], idx
                        else:
                            return features , label, coords, self.file_name[idx]


                else:
                    if not self.return_coords:
                        dir_path = os.path.join(self.root,"pt_w_patch_labels")
                        file_path = os.path.join(dir_path, self.file_name[idx]+'.pt')
                        data = torch.load(file_path)
                        features = data["features"]
                        patch_labels = data["patch_labels"] + self.st_cls
                        label = int(self.slide_label[idx])
                        return features , label, patch_labels            

                    else:
                        dir_path = os.path.join(self.root,"pt_w_patch_labels_coords")
                        file_path = os.path.join(dir_path, self.file_name[idx]+'.pt')
                        data = torch.load(file_path)
                        features = data["features"]
                        patch_labels = data["patch_labels"] + self.st_cls
                        coords = data["coords"]
                        label = int(self.slide_label[idx])

                        if hasattr(self, "logits"):
                            return features , label, patch_labels, coords, self.file_name[idx], idx
                        else:
                            return features , label, patch_labels, coords, self.file_name[idx]

            else:
                dir_path = os.path.join(self.root,"h5")
                file_path = os.path.join(dir_path, self.file_name[idx]+'.h5')
                data = h5py.File(file_path)
                features = torch.tensor(data['features'])
                patch_labels = torch.tensor(data['patch_labels']) + self.st_cls
                label = int(self.slide_label[idx])
                return features , label, patch_labels
            






class C16Dataset_Noise_Aug(Dataset):
    def __init__(self, file_name, file_label,root,persistence=False,keep_same_psize=0,is_train=False, use_h5=False, patch_labels=False, return_coords=False, noise_scale=0.05):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(C16Dataset_Noise_Aug, self).__init__()
        self.file_name = file_name
        self.slide_label = file_label
        self.slide_label = [int(_l) for _l in self.slide_label]

        self.class_list = list(set(self.slide_label))
        self.class_list.sort()

        self.st_cls = self.class_list[0]

        self.size = len(self.file_name)
        self.root = root
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train

        if persistence:
            self.feats = [ torch.load(os.path.join(root,'pt', _f+'.pt')) for _f in file_name ]

        self.use_h5 = use_h5
        self.patch_labels = patch_labels
        self.return_coords = return_coords

        self.noise_scale = noise_scale




    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        if self.persistence:
            features = self.feats[idx]
        else:
            
            if not self.use_h5:
                if not self.patch_labels:
                    if not self.return_coords:
                        dir_path = os.path.join(self.root,"pt")
                        file_path = os.path.join(dir_path, self.file_name[idx]+'.pt')
                        
                        features = torch.load(file_path)
                        features_noisy = features + \
                                self.noise_scale*features.norm(dim=-1,keepdim=True) * torch.normal(0, 1, size=features.shape)  #########################
                        features = torch.cat([features.unsqueeze(0), features_noisy.unsqueeze(0)], dim=0)

                        label = int(self.slide_label[idx])
                        
                        return features, label                

                    else:
                        dir_path = os.path.join(self.root,"pt_w_coords")
                        file_path = os.path.join(dir_path, self.file_name[idx]+'.pt')
                        data = torch.load(file_path)
                        features = data["features"]
                        features_noisy = features + \
                                self.noise_scale*features.norm(dim=-1,keepdim=True) * torch.normal(0, 1, size=features.shape)  #########################
                        features = torch.cat([features.unsqueeze(0), features_noisy.unsqueeze(0)], dim=0)

                        coords = data["coords"]
                        label = int(self.slide_label[idx])
                        return features , label, coords, self.file_name[idx]

                else:
                    if not self.return_coords:
                        dir_path = os.path.join(self.root,"pt_w_patch_labels")
                        file_path = os.path.join(dir_path, self.file_name[idx]+'.pt')
                        data = torch.load(file_path)
                        features = data["features"]
                        features_noisy = features + \
                                self.noise_scale*features.norm(dim=-1,keepdim=True) * torch.normal(0, 1, size=features.shape)  #########################
                        features = torch.cat([features.unsqueeze(0), features_noisy.unsqueeze(0)], dim=0)

                        patch_labels = data["patch_labels"] + self.st_cls
                        label = int(self.slide_label[idx])
                        return features , label, patch_labels            

                    else:
                        dir_path = os.path.join(self.root,"pt_w_patch_labels_coords")
                        file_path = os.path.join(dir_path, self.file_name[idx]+'.pt')
                        data = torch.load(file_path)
                        features = data["features"]
                        features_noisy = features + \
                                self.noise_scale*features.norm(dim=-1,keepdim=True) * torch.normal(0, 1, size=features.shape)  #########################
                        features = torch.cat([features.unsqueeze(0), features_noisy.unsqueeze(0)], dim=0)

                        patch_labels = data["patch_labels"] + self.st_cls
                        coords = data["coords"]
                        label = int(self.slide_label[idx])
                        return features , label, patch_labels, coords, self.file_name[idx]


            else:
                dir_path = os.path.join(self.root,"h5")
                file_path = os.path.join(dir_path, self.file_name[idx]+'.h5')
                data = h5py.File(file_path)
                features = torch.tensor(data['features'])
                features_noisy = features + \
                        self.noise_scale*features.norm(dim=-1,keepdim=True) * torch.normal(0, 1, size=features.shape)  #########################
                features = torch.cat([features.unsqueeze(0), features_noisy.unsqueeze(0)], dim=0)


                patch_labels = torch.tensor(data['patch_labels']) + self.st_cls
                label = int(self.slide_label[idx])
                return features , label, patch_labels
            









class C16Dataset_Joint(Dataset):
    def __init__(self, single_datasets):
        self.single_datasets = single_datasets
        self.len_per_data = [len(dataset) for dataset in self.single_datasets]
        self.len_datasets = np.array(self.len_per_data).sum().item()

        self.interval_datasets = [0] + list(itertools.accumulate([len(dataset) for dataset in self.single_datasets]))

    def __len__(self):
        return self.len_datasets
    
    def __getitem__(self, idx):
        pos = bisect.bisect_right(self.interval_datasets, idx) - 1

        return self.single_datasets[pos].__getitem__(idx-self.interval_datasets[pos])


class TCGADataset_Joint(Dataset):
    def __init__(self, single_datasets):
        self.single_datasets = single_datasets
        self.len_per_data = [len(dataset) for dataset in self.single_datasets]
        self.len_datasets = np.array(self.len_per_data).sum().item()

        self.interval_datasets = [0] + list(itertools.accumulate([len(dataset) for dataset in self.single_datasets]))

    def __len__(self):
        return self.len_datasets
    
    def __getitem__(self, idx):
        pos = bisect.bisect_right(self.interval_datasets, idx) - 1

        return self.single_datasets[pos].__getitem__(idx-self.interval_datasets[pos])
