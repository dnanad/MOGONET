import os
import numpy as np
import torch
import torch.nn.functional as F

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
            #            print("Module {:} loaded!".format(module))
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
