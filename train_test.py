""" Training and testing of the model
"""
from math import inf
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import graphviz
from sklearn.tree import export_graphviz
from xgboost import plot_tree as xgb_plot_tree

import torch
import torch.nn.functional as F
from models import init_model_dict, init_optim
from utils import (
    one_hot_tensor,
    cal_sample_weight,
    gen_adj_mat_tensor,
    gen_test_adj_mat_tensor,
    cal_adj_mat_parameter,
    result_plot,
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

    Parameters:
    ----------
    data_tr_list: list of torch tensor
        training data for each view
    data_trte_list : list of torch tensor
        all data for each view
    trte_idx: dict
        indices for training and testing
    adj_parameter : dict
        parameters for generating adjacency matrices

    Returns:
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

    Attributes:
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

    Returns:
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

    Attributes:
    ----------
    data_list : list of torch tensor
        testing data for each view
    adj_list : list of torch tensor
        adjacency matrices for each view
    te_idx : torch tensor
        indices for testing
    model_dict : dict
        models for each view

    Returns:
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
            # get predicted labels
            model_dict["C{:}".format(i + 1)](
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
    latest_model,
):
    """Train and test
    Attributes:
    ----------
    folder_path : str
        path of the folder where the data is stored
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

    Returns:
    ----------
    model_dict : dict
        models for each view
    epoch_loss_dict : dict
        losses for each view
    test_info : dict
        test results
    """
    num_view = len(view_list)  # get number of views
    dim_hvcdn = pow(num_class, num_view)  # get dimension of hidden layer for VCDN
    # prepare training and testing data
    (
        data_tr_list,
        data_trte_list,
        trte_idx,
        labels_trte,
    ) = prepare_trte_data(data_folder, view_list)

    # get training labels
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])

    # get one-hot training labels
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)

    # get sample weights
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)

    # convert sample weights to torch tensor
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    if cuda:  # move tensors to GPU
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()

    # generate adjacency matrices for training and testing
    (
        adj_tr_list,
        adj_te_list,
    ) = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    dim_list = [
        x.shape[1] for x in data_tr_list
    ]  # get dimensions of features for each view

    # initialize models
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)

    # move models to GPU
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()

    # pretrain GCNs
    print("\nPretrain GCNs...")

    # initialize optimizers
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)

    for epoch in range(num_epoch_pretrain):
        if epoch % test_interval == 0:
            print("Preptarin epoch {:}".format(epoch))
        # pretrain one epoch
        train_epoch(
            data_tr_list,
            adj_tr_list,
            labels_tr_tensor,
            onehot_labels_tr_tensor,
            sample_weight_tr,
            model_dict,
            optim_dict,
            train_VCDN=False,
        )

    # traning the GCNs and the classifier
    print("\nTraining...")
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)  # initialize optimizers
    epoch_loss_dict = {}
    test_info = {}
    if num_class == 2:
        for epoch in range(num_epoch + 1):
            loss_dict = train_epoch(  # train one epoch
                data_tr_list,
                adj_tr_list,
                labels_tr_tensor,
                onehot_labels_tr_tensor,
                sample_weight_tr,
                model_dict,
                optim_dict,
            )
            epoch_loss_dict[epoch] = loss_dict  # save loss
            info = {}
            scores_df = pd.DataFrame()
            if epoch % test_interval == 0:  # test
                info["epoch"] = epoch
                te_prob = test_epoch(  # test one epoch
                    data_trte_list, adj_te_list, trte_idx["te"], model_dict
                )
                # info["prob"] = te_prob
                print("\nTest: Epoch {:d}".format(epoch))  # print test results

                acc = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                f1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                roc = roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:, 1])
                info["acc"] = acc
                info["f1"] = f1
                info["roc"] = roc
                print("Test ACC: {:.3f}".format(acc))
                print("Test F1: {:.3f}".format(f1))
                print("Test AUC: {:.3f}".format(roc))
            if epoch == num_epoch:
                acc_final = accuracy_score(
                    labels_trte[trte_idx["te"]], te_prob.argmax(1)
                )
                f1_final = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                roc_final = roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:, 1])
                print(
                    "Total number of samples in test set: {:d}".format(
                        len(trte_idx["te"])
                    )
                )
                print(
                    "Confusion Matrix: \n",
                    confusion_matrix(labels_trte[trte_idx["te"]], te_prob.argmax(1)),
                )
                result_plot(
                    labels_trte[trte_idx["te"]], te_prob, latest_model=latest_model
                )
                # create the scores in a dataframe
                scores_df["ACC"] = [acc_final]
                scores_df["F1"] = [f1_final]
                scores_df["AUC"] = [roc_final]
                scores_df["CM"] = [
                    confusion_matrix(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                ]

    else:
        for epoch in range(num_epoch + 1):
            loss_dict = train_epoch(  # train one epoch
                data_tr_list,
                adj_tr_list,
                labels_tr_tensor,
                onehot_labels_tr_tensor,
                sample_weight_tr,
                model_dict,
                optim_dict,
            )
            epoch_loss_dict[epoch] = loss_dict  # save loss
            info = {}
            scores_df = pd.DataFrame()
            if epoch % test_interval == 0:  # test
                info["epoch"] = epoch
                te_prob = test_epoch(  # test one epoch
                    data_trte_list, adj_te_list, trte_idx["te"], model_dict
                )
                # info["prob"] = te_prob
                print("\nTest: Epoch {:d}".format(epoch))  # print test results

                acc = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                f1_weighted = f1_score(
                    labels_trte[trte_idx["te"]], te_prob.argmax(1), average="weighted"
                )
                f1_macro = f1_score(
                    labels_trte[trte_idx["te"]],
                    te_prob.argmax(1),
                    average="macro",
                )
                info["acc"] = acc
                info["f1_weighted"] = f1_weighted
                info["f1_macro"] = f1_macro
                print(
                    "Test ACC: {:.3f}".format(
                        accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                    )
                )
                print("Test F1 weighted: {:.3f}".format(f1_weighted))
                print("Test F1 macro: {:.3f}".format(f1_macro))

            if epoch == num_epoch:
                # create the scores in a dataframe
                scores_df["ACC"] = [acc]
                scores_df["F1_weighted"] = [f1_weighted]
                scores_df["F1_macro"] = [f1_macro]
        test_info[epoch] = info
    print("\nTraining finished!")

    return model_dict, epoch_loss_dict, test_info, scores_df


def calc_score(y, yhat):
    """Calculate accuracy, F1 score, AUC, and confusion matrix.

    Attributes:
    y: list
        true labels
    yhat: list
        predicted labels

    Returns:
    acc: float
        accuracy
    f1: float
        F1 score
    auc: float
        AUC
    cm: array
        confusion matrix
    """
    acc = accuracy_score(y, yhat)
    f1 = f1_score(y, yhat)
    auc = roc_auc_score(y, yhat)
    cm = confusion_matrix(y, yhat)

    return (acc, f1, auc, cm)


def fit_predict_evaluate(
    options,
    pl,
    X_train,
    y_train,
    X_test,
    y_test,
    GSparameters,
    Savepath,
    rootfigname,
    threshold,
    cm_normalised,
):
    ####################### TODO: add correct columns to DF_train and DF_test
    # y_train = DF_train[""]
    # X_train = DF_train.drop("", axis=1)

    # y_test = DF_test[""]
    # X_test = DF_test.drop("", axis=1)

    if options["mode"] == "test":
        print(f"Performing test run with default hyperparameters")

        # Fit - Predict
        print("Fitting the model...")
        pl.fit(X_train, y_train)

    elif options["mode"] == "gridsearch":
        if GSparameters is None:
            raise (
                ValueError(
                    "GSparameters is not defined (models.py) for the chosen model."
                )
            )
        print("Performing a gridsearch")
        grid = GridSearchCV(pl, GSparameters, cv=5, n_jobs=-1).fit(X_train, y_train)
        best_params = grid.best_params_
        print(f" - best model parameters:{best_params}")
        # Store the optimum model in best_pipe
        best_pipe = grid.best_estimator_
        pl = best_pipe

    # Evaluate
    print("Predicting and evaluating...")
    y_test_predict = pl.predict(X_test)
    y_proba = pl.predict_proba(X_test)

    yyhat_dict = {"test": {"y": y_test, "yhat": y_test_predict, "yproba": y_proba}}

    scores = get_skill_scores(yyhat_dict)

    skill_vis(yyhat_dict, Savepath, rootfigname, cm_normalised=cm_normalised)
    # model_vis(options, pl["model"], Savepath)

    return scores


def get_skill_scores(yyhat_dict):
    scores = {}
    for key in yyhat_dict.keys():
        y = yyhat_dict[key]["y"]
        yhat = yyhat_dict[key]["yhat"]

        # calculate scores
        acc, f1, auc, cm = calc_score(y, yhat)
        scores[key] = {"ACC": acc, "F1": f1, "AUC": auc, "CM": cm}

        print(
            f"%s scores: ACC = %.3f, F1 = %.3f, AUC = %.3f, (n:%d)"
            % (key, acc, f1, auc, len(y))
        )

        print("Confusion Matrix: \n", cm)

    return scores


def skill_vis(
    yyhat_dict,
    Savepath,
    rootfigname,
    cm_normalised=False,
):
    import scikitplot as skplt

    for i, key in enumerate(yyhat_dict.keys()):
        y = yyhat_dict[key]["y"].to_numpy().flatten()
        yhat = yyhat_dict[key]["yhat"]
        yproba = yyhat_dict[key]["yproba"]

        # save each figure seprpately
        skplt.metrics.plot_confusion_matrix(
            y, yhat, normalize=cm_normalised, title="Confusion Matrix", figsize=(8, 8)
        )
        # save plot
        image_name = rootfigname + "_" + str(i) + "_cm.png"  #
        figpath = os.path.join(Savepath, image_name)
        if not os.path.exists(Savepath):
            os.makedirs(Savepath)
        # save the above given path
        plt.savefig(figpath)

        skplt.metrics.plot_roc(y, yproba, title="ROC Plot")
        # save plot
        image_name = rootfigname + "_" + str(i) + "_roc.png"  #
        figpath = os.path.join(Savepath, image_name)
        # save the above given path
        plt.savefig(figpath)

        skplt.metrics.plot_precision_recall(y, yproba, title="PR Curve")
        # save plot
        image_name = rootfigname + "_" + str(i) + "_pr.png"  #
        figpath = os.path.join(Savepath, image_name)
        # save the above given path
        plt.savefig(figpath)

        fig, ax = plt.subplots(1, 2)
        skplt.metrics.plot_cumulative_gain(
            y, yproba, title="Cumulative Gains Chart", ax=ax[0]
        )
        skplt.metrics.plot_lift_curve(y, yproba, ax=ax[1], title="Lift Curve")
        # save plot
        image_name = rootfigname + "_" + str(i) + "_gain_lift.png"  #
        figpath = os.path.join(Savepath, image_name)
        # save the above given path
        fig.savefig(figpath)

    print(f"Saved figure: {figpath}")


# plt.figure(figsize=(15,7))
# grid = plt.GridSpec(2, 3, wspace=0.3, hspace=0.4)

# for i in range(6):

#     col, row = i%3,i//3
#     ax = plt.subplot(grid[row,col])
#     ax.title.set_color('blue')

#     model = classifiers[i]
#     skplt.metrics.plot_roc(y_test, model.predict_proba(X_test), ax=ax, title=type(cls).__name__)

# plt.show()

# fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
# metrics.auc(fpr, tpr)
