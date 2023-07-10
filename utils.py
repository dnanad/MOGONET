import os
import numpy as np
import torch
import torch.nn.functional as F

import pandas as pd
from sklearn.model_selection import train_test_split


cuda = True if torch.cuda.is_available() else False  # use GPU if available


def cal_sample_weight(labels, num_class, use_sample_weight=True):
    """Calculate sample weights for each sample in the dataset.

    Parameters
    ----------
    labels : numpy array
        Labels of the dataset.
    num_class : int
        Number of classes.
    use_sample_weight : bool, optional
        Whether to use sample weights. The default is True.

    Returns
    -------
    sample_weight : numpy array
        Sample weights for each sample in the dataset.
    """
    if not use_sample_weight:  # if not use sample weights, return uniform weights
        return np.ones(len(labels)) / len(labels)
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels == i)
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels == i)[0]] = count[i] / np.sum(count)

    return sample_weight


def one_hot_tensor(y, num_dim):
    """Convert a tensor of labels to one-hot representation.

    Parameters
    ----------
    y : torch tensor
        Labels of the dataset.
    num_dim : int
        Number of classes.

    Returns
    -------
    y_onehot : torch tensor
        One-hot representation of the labels.
    """
    y_onehot = torch.zeros(y.shape[0], num_dim)  # initialize one-hot tensor
    y_onehot.scatter_(1, y.view(-1, 1), 1)  # fill in the one-hot tensor

    return y_onehot


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    """Calculate cosine distance between two tensors.

    Parameters
    ----------
    x1 : torch tensor
        First tensor.
    x2 : torch tensor, optional
        Second tensor. The default is None.
    eps : float, optional
        Epsilon is the minimum value to avoid division by zero. The default is 1e-8.

    Returns
    -------
    torch tensor
        Cosine distance between two tensors.
    """
    x2 = x1 if x2 is None else x2  # if x2 is not provided, use x1 as x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)  # calculate the norm of x1
    w2 = (
        w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    )  # if x2 is not provided, use x1 as x2
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def to_sparse(x):
    """Convert a tensor to sparse tensor.

    Parameters
    ----------
    x : torch tensor
        Input tensor.

    Returns
    -------
    sparse_tensortype
        Sparse tensor.
    """
    x_typename = torch.typename(x).split(".")[-1]  # get the type of the input tensor
    sparse_tensortype = getattr(torch.sparse, x_typename)  # get the sparse tensor type
    indices = torch.nonzero(x)  # get the indices of non-zero elements
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(
            *x.shape
        )  # return a sparse tensor with all elements zeros
    indices = indices.t()  # transpose the indices
    values = x[
        tuple(indices[i] for i in range(indices.shape[0]))
    ]  # get the values of non-zero elements
    return sparse_tensortype(indices, values, x.size())  # return a sparse tensor


def cal_adj_mat_parameter(edge_per_node, data, metric="cosine"):
    """Calculate the parameter for the adjacency matrix.

    Parameters
    ----------
    edge_per_node : int
        Number of edges per node.
    data : numpy array
        Data matrix.
    metric : str, optional
        Metric to calculate the distance between two data points. The default is "cosine".

    Returns
    -------
    float
        Parameter for the adjacency matrix.
    """
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    parameter = torch.sort(
        dist.reshape(
            -1,
        )
    ).values[edge_per_node * data.shape[0]]
    return np.ndarray.item(parameter.data.cpu().numpy())
    # return np.asscalar(parameter.data.cpu().numpy())


def graph_from_dist_tensor(dist, parameter, self_dist=True):
    """Generate a graph from a pairwise distance matrix.

    Parameters
    ----------
    dist : torch tensor
        Pairwise distance matrix.
    parameter : float
        Parameter for the adjacency matrix.
    self_dist : bool, optional
        Whether the input is a self-distance matrix. The default is True.

    Returns
    -------
    g : torch tensor
        Adjacency matrix.
    """
    if self_dist:
        assert (
            dist.shape[0] == dist.shape[1]
        ), "Input is not pairwise dist matrix"  # check if the input is a self-distance matrix
    g = (dist <= parameter).float()  # generate the adjacency matrix
    if self_dist:  # if the input is a self-distance matrix
        diag_idx = np.diag_indices(g.shape[0])  # get the indices of diagonal elements
        g[diag_idx[0], diag_idx[1]] = 0  # set the diagonal elements to zero

    return g


def gen_adj_mat_tensor(data, parameter, metric="cosine"):
    """Generate an adjacency matrix from a data matrix.

    Parameters
    ----------
    data : torch tensor
        Data matrix.
    parameter : float
        Parameter for the adjacency matrix.
    metric : str, optional
        Metric to calculate the distance between two data points. The default is "cosine".

    Returns
    -------
    adj : torch tensor
        Adjacency matrix.
    """
    assert (
        metric == "cosine"
    ), "Only cosine distance implemented"  # check if the metric is cosine distance
    dist = cosine_distance_torch(data, data)  # calculate the pairwise distance matrix
    g = graph_from_dist_tensor(
        dist, parameter, self_dist=True
    )  # generate the adjacency matrix
    if metric == "cosine":  # if the metric is cosine distance
        adj = 1 - dist  # calculate the adjacency matrix
    else:
        raise NotImplementedError  # raise error if the metric is not cosine distance
    adj = adj * g  # set the non-adjacent elements to zero
    adj_T = adj.transpose(
        0, 1
    )  # transpose the adjacency matrix to get the symmetric adjacency matrix
    I = torch.eye(adj.shape[0])  # generate an identity matrix
    if cuda:  # if cuda is available
        I = I.cuda()  # move the identity matrix to cuda
    adj = (
        adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()
    )  # set the non-adjacent elements to zero
    adj = F.normalize(adj + I, p=1)  # normalize the adjacency matrix
    adj = to_sparse(adj)  # convert the adjacency matrix to sparse matrix

    return adj


def gen_test_adj_mat_tensor(data, trte_idx, parameter, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    adj = torch.zeros((data.shape[0], data.shape[0]))
    if cuda:
        adj = adj.cuda()
    num_tr = len(trte_idx["tr"])

    dist_tr2te = cosine_distance_torch(data[trte_idx["tr"]], data[trte_idx["te"]])
    g_tr2te = graph_from_dist_tensor(dist_tr2te, parameter, self_dist=False)
    if metric == "cosine":
        adj[:num_tr, num_tr:] = 1 - dist_tr2te
    else:
        raise NotImplementedError
    adj[:num_tr, num_tr:] = adj[:num_tr, num_tr:] * g_tr2te

    dist_te2tr = cosine_distance_torch(data[trte_idx["te"]], data[trte_idx["tr"]])
    g_te2tr = graph_from_dist_tensor(dist_te2tr, parameter, self_dist=False)
    if metric == "cosine":
        adj[num_tr:, :num_tr] = 1 - dist_te2tr
    else:
        raise NotImplementedError
    adj[num_tr:, :num_tr] = adj[num_tr:, :num_tr] * g_te2tr  # retain selected edges

    adj_T = adj.transpose(0, 1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)

    return adj


def save_model_dict(folder, model_dict):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for module in model_dict:
        torch.save(
            model_dict[module].state_dict(), os.path.join(folder, module + ".pth")
        )


def load_model_dict(folder, model_dict):
    for module in model_dict:
        if os.path.exists(os.path.join(folder, module + ".pth")):
            print("Module {:} loaded!".format(module))
            model_dict[module].load_state_dict(
                torch.load(
                    os.path.join(folder, module + ".pth"),
                    map_location="cuda:{:}".format(torch.cuda.current_device()),
                )
            )
        else:
            print("WARNING: Module {:} from model_dict is not loaded!".format(module))
        if cuda:
            model_dict[module].cuda()
    return model_dict


# data preprocessing


def import_process_datafile(
    raw_data_file_path, columns_to_drop, index_column, labels_dict
):
    df = pd.read_csv(raw_data_file_path, sep=",")
    df = df.drop(columns=columns_to_drop)  # drop gene_num column
    df.set_index(index_column, inplace=False)  # set gene_name as index
    df_T = df.T
    df_T.columns = df_T.iloc[0:1].values[0]  # set the first row as column names
    df_T = df_T.iloc[1:, :]  # drop the first row

    # added new column, using a dictionary labels_dict
    df_T["disease"] = df_T.index.map(labels_dict)

    return df_T


def get_label_dict(labels_path):
    df_labels = pd.read_csv(labels_path)
    df_sample_names_labels = df_labels[["sample_name", "disease"]]
    df_sample_names_labels.set_index("sample_name", inplace=True)
    label_dict = df_sample_names_labels.disease.to_dict()

    return label_dict


def save_feat_name(i, df, data_folder_path):
    # save the feature names as `i_featname.csv`
    df_feat = pd.DataFrame(
        df.columns.drop(
            ["disease"],
        )
    )
    file_name = str(i) + "_featname.csv"
    file_path = os.path.join(data_folder_path, file_name)
    df_feat.to_csv(file_path, header=None, index=None)

    return


def train_test_save(i, df, test_size, data_folder_path):
    train, test = train_test_split(df, test_size=test_size)

    label_train_path = os.path.join(data_folder_path, "labels_tr.csv")
    label_train = train["disease"]
    label_train.to_csv(label_train_path, header=None, index=None)
    print(
        "Labels for training set saved as `labels_tr.csv` in the folder:",
        data_folder_path,
    )
    train_file_name = str(i) + "_tr.csv"
    train_path = os.path.join(data_folder_path, train_file_name)
    train = train.drop(columns=["disease"])
    train.to_csv(train_path, header=None, index=None)
    print("Training set saved as `", train_file_name, "in the folder:", train_path)

    label_test_path = os.path.join(data_folder_path, "labels_te.csv")
    label_test = test["disease"]
    label_test.to_csv(label_test_path, header=None, index=None)
    print("Label for test set saved as `labels_te.csv` in the folder:", label_test_path)

    test_file_name = str(i) + "_te.csv"
    test_path = os.path.join(data_folder_path, test_file_name)
    test = test.drop(columns=["disease"])
    test.to_csv(test_path, header=None, index=None)
    print("Test set saved as`", test_file_name, "in the folder:", test_path)

    return


def dataset_summary(folder_name):
    print("Dataset:", folder_name)
    df_train = pd.read_csv("./" + folder_name + "/1_tr.csv", header=None)
    df_test = pd.read_csv("./" + folder_name + "/1_te.csv", header=None)
    print(
        "Number of features:",
        pd.read_csv("./" + folder_name + "/1_featname.csv", header=None).shape[0],
    )
    print("Total Number of samples:", df_train.shape[0] + df_test.shape[0])
    print(
        "Number of labels:",
        len(
            pd.read_csv("./" + folder_name + "/labels_te.csv", header=None)[0].unique()
        ),
    )
    print("-------------------------------------------------------------------")
    print(
        "Training set dimension:",
        df_train.shape,
    )
    print(
        "Number of samples for TRAINING:",
        df_train.shape[0],
    )
    print(
        "Number of labels for TRAINING:",
        pd.read_csv("./" + folder_name + "/labels_tr.csv", header=None).shape[0],
    )
    print("-------------------------------------------------------------------")
    print(
        "Test set dimension:",
        df_test.shape,
    )
    print(
        "Number of samples for TESTING:",
        df_test.shape[0],
    )
    print(
        "Number of labels for TESTING:",
        pd.read_csv("./" + folder_name + "/labels_te.csv", header=None).shape[0],
    )


def find_numFolders_maxNumFolders(input):
    intlistfolders = []
    list_subfolders_with_paths = [f.name for f in os.scandir(input) if f.is_dir()]
    for x in list_subfolders_with_paths:
        try:
            intval = int(x)
            # print(intval)
            intlistfolders.append(intval)
        except:
            pass
    return intlistfolders, max(intlistfolders)


# pipline for cml

# import packages
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler


def get_pipelines(options, DF, model):
    pl_preprocessor = build_preprocessor_pipeline(DF, options["features_n"])

    # Build the entire pipeline
    pl = Pipeline(steps=[("preproc", pl_preprocessor), ("model", model)])

    return pl


def build_preprocessor_pipeline(DF):
    # numeric column names
    num_cols = DF.select_dtypes(exclude=["object"]).columns.tolist()

    # pipeline for numerical columns
    num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    pl_impute_encode = ColumnTransformer([("num", num_pipe, num_cols)])

    # full preprocessor pipeline
    pl_preprocessor = Pipeline([("impute_encode", pl_impute_encode)])

    return pl_preprocessor


from datetime import datetime


def get_expname_datetime(options):
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%Y%m%d-%H%M%S")
    expname = (
        dt_string
        + "_"
        + options["name"]
        + "_"
        + options["model"]
        + "_"
        + options["mode"]
    )
    print("exp. name =" + expname)
    return expname
