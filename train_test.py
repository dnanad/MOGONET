""" Training and testing of the model
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import torch
import torch.nn.functional as F
from models import init_model_dict, init_optim
from utils import (
    one_hot_tensor,
    cal_sample_weight,
    gen_adj_mat_tensor,
    gen_test_adj_mat_tensor,
    cal_adj_mat_parameter,
)

cuda = True if torch.cuda.is_available() else False  # use GPU if available


def prepare_trte_data(data_folder, view_list):
    """Prepare training and testing data

    input:
    ----------
    data_folder : str
        folder of the dataset
    view_list : list of int
        list of view indices

    output:
    ----------
    data_train_list : list of torch tensor
        training data for each view
    data_all_list: list of torch tensor
        all data for each view
    idx_dict: dict
        indices for training and testing
    labels: numpy array
        labels for all data
    """
    num_view = len(view_list)  # number of types of views (data/omics)
    labels_tr = np.loadtxt(
        os.path.join(data_folder, "labels_tr.csv"), delimiter=","
    )  # load training labels
    labels_te = np.loadtxt(
        os.path.join(data_folder, "labels_te.csv"), delimiter=","
    )  # load testing labels
    labels_tr = labels_tr.astype(int)  # convert to int
    labels_te = labels_te.astype(int)  # convert to int
    data_tr_list = []
    data_te_list = []
    for i in view_list:  # load training and testing data for each view
        data_tr_list.append(
            np.loadtxt(os.path.join(data_folder, str(i) + "_tr.csv"), delimiter=",")
        )
        data_te_list.append(
            np.loadtxt(os.path.join(data_folder, str(i) + "_te.csv"), delimiter=",")
        )
    num_tr = data_tr_list[0].shape[0]  # number of training samples
    num_te = data_te_list[0].shape[0]  # number of testing samples
    data_mat_list = []
    for i in range(num_view):  # concatenate training and testing data for each view
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):  # convert to torch tensor
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))  # indices for training samples
    idx_dict["te"] = list(
        range(num_tr, (num_tr + num_te))
    )  # indices for testing samples
    data_train_list = []
    data_all_list = []
    for i in range(
        len(data_tensor_list)
    ):  # split training and testing data for each view
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        # concatenate training and testing data for each view
        data_all_list.append(
            torch.cat(
                (
                    data_tensor_list[i][idx_dict["tr"]].clone(),
                    data_tensor_list[i][idx_dict["te"]].clone(),
                ),
                0,
            )
        )
    labels = np.concatenate(
        (labels_tr, labels_te)
    )  # concatenate training and testing labels

    return data_train_list, data_all_list, idx_dict, labels


def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    """Generate adjacency matrices for training and testing

    input:
    ----------
    data_tr_list: list of torch tensor
        training data for each view
    data_trte_list : list of torch tensor
        all data for each view
    trte_idx: dict
        indices for training and testing
    adj_parameter : dict
        parameters for generating adjacency matrices

    output:
    ----------
    adj_train_list : list of torch tensor
        adjacency matrices for training
    adj_test_list : list of torch tensor
        adjacency matrices for testing
    """
    adj_metric = "cosine"  # cosine distance
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):  # generate adjacency matrices for each view
        adj_parameter_adaptive = cal_adj_mat_parameter(  # calculate parameters for generating adjacency matrices
            adj_parameter, data_tr_list[i], adj_metric
        )
        adj_train_list.append(  # generate adjacency matrices for training
            gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric)
        )
        adj_test_list.append(  # generate adjacency matrices for testing
            gen_test_adj_mat_tensor(
                data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric
            )
        )

    return adj_train_list, adj_test_list


def train_epoch(
    data_list,
    adj_list,
    label,
    one_hot_label,
    sample_weight,
    model_dict,
    optim_dict,
    train_VCDN=True,
):
    """Train one epoch

    input:
    ----------
    data_list : list of torch tensor
        training data for each view
    adj_list : list of torch tensor
        adjacency matrices for each view
    label : torch tensor
        labels for training data
    one_hot_label : torch tensor
        one-hot labels for training data
    sample_weight : torch tensor
        sample weights for training data
    model_dict : dict
        models for each view
    optim_dict : dict
        optimizers for each view
    train_VCDN : bool
        whether to train VCDN

    output:
    ----------
    loss_dict : dict
        losses for each view
    """
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction="none")  # cross entropy loss
    for m in model_dict:  # set models to train mode
        model_dict[m].train()
    num_view = len(data_list)
    for i in range(num_view):  # train each view
        optim_dict["C{:}".format(i + 1)].zero_grad()  # set gradients to zero
        ci_loss = 0  # initialize loss
        ci = model_dict["C{:}".format(i + 1)](  # get predicted labels
            model_dict["E{:}".format(i + 1)](
                data_list[i], adj_list[i]
            )  # get embeddings
        )
        ci_loss = torch.mean(
            torch.mul(criterion(ci, label), sample_weight)
        )  # calculate loss
        ci_loss.backward()  # backpropagation
        optim_dict["C{:}".format(i + 1)].step()  # update parameters
        loss_dict["C{:}".format(i + 1)] = (
            ci_loss.detach().cpu().numpy().item()
        )  # save loss
    if train_VCDN and num_view >= 2:  # train VCDN
        optim_dict["C"].zero_grad()  # set gradients to zero
        c_loss = 0  # initialize loss
        ci_list = []
        for i in range(num_view):  # get predicted labels for each view
            ci_list.append(
                model_dict["C{:}".format(i + 1)](  # get predicted labels
                    model_dict["E{:}".format(i + 1)](
                        data_list[i], adj_list[i]
                    )  # get embeddings
                )
            )
        c = model_dict["C"](ci_list)  # get predicted labels
        c_loss = torch.mean(
            torch.mul(criterion(c, label), sample_weight)
        )  # calculate loss
        c_loss.backward()  # backpropagation
        optim_dict["C"].step()  # update parameters
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()  # save loss

    return loss_dict


def test_epoch(data_list, adj_list, te_idx, model_dict):
    """Test one epoch

    input:
    ----------
    data_list : list of torch tensor
        testing data for each view
    adj_list : list of torch tensor
        adjacency matrices for each view
    te_idx : torch tensor
        indices for testing
    model_dict : dict
        models for each view

    output:
    ----------
    prob : numpy array
        predicted probabilities
    """
    for m in model_dict:  # set models to evaluation mode
        model_dict[m].eval()
    num_view = len(data_list)  # get number of views
    ci_list = []
    for i in range(num_view):  # get predicted labels for each view
        ci_list.append(
            model_dict["C{:}".format(i + 1)](  # get predicted labels
                model_dict["E{:}".format(i + 1)](
                    data_list[i], adj_list[i]
                )  # get embeddings
            )
        )
    if num_view >= 2:  # get predicted labels for VCDN
        c = model_dict["C"](ci_list)  # get predicted labels
    else:
        c = ci_list[0]  # get predicted labels
    c = c[te_idx, :]  # get predicted labels for testing
    prob = F.softmax(c, dim=1).data.cpu().numpy()  # get predicted probabilities

    return prob


def train_test(
    data_folder,
    view_list,
    num_class,
    adj_parameter,
    dim_he_list,
    lr_e_pretrain,
    lr_e,
    lr_c,
    num_epoch_pretrain,
    num_epoch,
    test_interval,
):
    """Train and test VCDN

    input:
    ----------
    data_folder : str
        name of dataset
    view_list : list of str
        names of views
    num_class : int
        number of classes
    adj_parameter : float
        parameter for adjacency matrix
    dim_he_list : list of int
        dimensions of hidden layers for encoder
    lr_e_pretrain : float
        learning rate for pretraining encoder
    lr_e : float
        learning rate for encoder
    lr_c : float
        learning rate for classifier
    num_epoch_pretrain : int
        number of epochs for pretraining encoder
    num_epoch : int
        number of epochs for training VCDN
    test_interval : int
        interval for testing

    output:
    ----------
    prob : numpy array
        predicted probabilities
    """
    num_view = len(view_list)  # get number of views
    dim_hvcdn = pow(num_class, num_view)  # get dimension of hidden layer for VCDN
    # if data_folder == "TEST_DATA":
    #     adj_parameter = 2
    #     dim_he_list = [600, 600, 400]
    # if data_folder == "ROSMAP":
    #     adj_parameter = 2
    #     dim_he_list = [200, 200, 100]
    # if data_folder == "BRCA":
    #     adj_parameter = 10
    #     dim_he_list = [400, 400, 200]
    (
        data_tr_list,
        data_trte_list,
        trte_idx,
        labels_trte,
    ) = prepare_trte_data(  # prepare training and testing data
        data_folder, view_list
    )
    labels_tr_tensor = torch.LongTensor(
        labels_trte[trte_idx["tr"]]
    )  # get training labels
    onehot_labels_tr_tensor = one_hot_tensor(
        labels_tr_tensor, num_class
    )  # get one-hot training labels
    sample_weight_tr = cal_sample_weight(
        labels_trte[trte_idx["tr"]], num_class
    )  # get sample weights
    sample_weight_tr = torch.FloatTensor(
        sample_weight_tr
    )  # convert sample weights to torch tensor
    if cuda:  # move tensors to GPU
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
    (
        adj_tr_list,
        adj_te_list,
    ) = gen_trte_adj_mat(  # generate adjacency matrices for training and testing
        data_tr_list, data_trte_list, trte_idx, adj_parameter
    )
    dim_list = [
        x.shape[1] for x in data_tr_list
    ]  # get dimensions of features for each view
    model_dict = init_model_dict(
        num_view, num_class, dim_list, dim_he_list, dim_hvcdn
    )  # initialize models
    for m in model_dict:  # move models to GPU
        if cuda:
            model_dict[m].cuda()

    print("\nPretrain GCNs...")  # pretrain GCNs
    optim_dict = init_optim(
        num_view, model_dict, lr_e_pretrain, lr_c
    )  # initialize optimizers
    for epoch in range(num_epoch_pretrain):  # pretrain GCNs
        train_epoch(  # train one epoch
            data_tr_list,
            adj_tr_list,
            labels_tr_tensor,
            onehot_labels_tr_tensor,
            sample_weight_tr,
            model_dict,
            optim_dict,
            train_VCDN=False,
        )
    print("\nTraining...")  # train VCDN
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)  # initialize optimizers
    for epoch in range(num_epoch + 1):  # train VCDN
        train_epoch(  # train one epoch
            data_tr_list,
            adj_tr_list,
            labels_tr_tensor,
            onehot_labels_tr_tensor,
            sample_weight_tr,
            model_dict,
            optim_dict,
        )
        if epoch % test_interval == 0:  # test
            te_prob = test_epoch(  # test one epoch
                data_trte_list, adj_te_list, trte_idx["te"], model_dict
            )
            print("\nTest: Epoch {:d}".format(epoch))  # print test results
            if num_class == 2:
                print(
                    "Test ACC: {:.3f}".format(
                        accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                    )
                )
                print(
                    "Test F1: {:.3f}".format(
                        f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                    )
                )
                print(
                    "Test AUC: {:.3f}".format(
                        roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:, 1])
                    )
                )
                if epoch == num_epoch:
                    print(
                        "Total number of samples in test set: {:d}".format(
                            len(trte_idx["te"])
                        )
                    )
                    print(
                        "Confusion Matrix: \n",
                        confusion_matrix(
                            labels_trte[trte_idx["te"]], te_prob.argmax(1)
                        ),
                    )
            else:
                print(
                    "Test ACC: {:.3f}".format(
                        accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                    )
                )
                print(
                    "Test F1 weighted: {:.3f}".format(
                        f1_score(
                            labels_trte[trte_idx["te"]],
                            te_prob.argmax(1),
                            average="weighted",
                        )
                    )
                )
                print(
                    "Test F1 macro: {:.3f}".format(
                        f1_score(
                            labels_trte[trte_idx["te"]],
                            te_prob.argmax(1),
                            average="macro",
                        )
                    )
                )
