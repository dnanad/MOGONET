import os
import numpy as np
import torch
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt
import pickle


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from datetime import datetime

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
                    os.path.join(folder, module + ".pth")
                    # map_location="cuda:{:}".format(torch.cuda.current_device()),
                )
            )
        else:
            print("WARNING: Module {:} from model_dict is not loaded!".format(module))
        if cuda:
            model_dict[module].cuda()
    return model_dict


# data preprocessing


def import_process_datafile(raw_data_file_path, columns_to_drop, index_column):
    df = pd.read_csv(raw_data_file_path, sep=",")
    df = df.drop(columns=columns_to_drop)  # drop gene_num column
    df.set_index(index_column, inplace=False)  # set gene_name as index
    df_T = df.T
    df_T.columns = df_T.iloc[0:1].values[0]  # set the first row as column names
    df_T = df_T.iloc[1:, :]  # drop the first row

    return df_T


def get_label_dict(labels_path):
    df_labels = pd.read_csv(labels_path)
    df_sample_names_labels = df_labels[["sample_name", "disease"]]
    df_sample_names_labels.set_index("sample_name", inplace=True)
    label_dict = df_sample_names_labels.disease.to_dict()

    return label_dict


def save_feat_name(j, df, data_folder_path, i, CV=False):
    # save the feature names as `j_featname.csv`
    df_feat = pd.DataFrame(
        df.columns.values.tolist(),
        columns=["feature_name"],  # , index=range(len(df.columns)
    )
    if CV:
        folder_name = "CV"
        folder_path = os.path.join(data_folder_path, folder_name)
        if not os.path.exists(folder_path):  # if the "CV" folder does not exist
            os.makedirs(folder_path)  # create the folder
        cv_folder = os.path.join(folder_path, "CV_" + str(i))
        if not os.path.exists(cv_folder):  # if the "CV_i" folder does not exist
            os.makedirs(cv_folder)  # create the folder
        file_name = str(j) + "_featname.csv"
        file_path = os.path.join(cv_folder, file_name)
        df_feat.to_csv(file_path, header=None, index=None)

    else:
        folder_name = "NO_CV"
        folder_path = os.path.join(data_folder_path, folder_name)
        if not os.path.exists(folder_path):  # if the "NO_CV" folder does not exist
            os.makedirs(folder_path)  # create the folder

        file_name = str(j) + "_featname.csv"
        file_path = os.path.join(folder_path, file_name)
        df_feat.to_csv(file_path, header=None, index=None)

    return


# train_test_split function
def train_test_split(common_sample_ids, test_size, sample_folder, n_splits, CV=False):
    if CV:
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_splits, shuffle=False)
        kf.get_n_splits(common_sample_ids)
        train_test_folder = os.path.join(sample_folder, "CV_train_test")
        if not os.path.exists(train_test_folder):
            os.makedirs(train_test_folder)
        common_sample_ids = list(common_sample_ids)
        # save train and test as pickle files
        for i, (train_index, test_index) in enumerate(kf.split(common_sample_ids)):
            cv_folder = os.path.join(train_test_folder, "CV_" + str(i))
            train = [common_sample_ids[i] for i in train_index]
            test = [common_sample_ids[i] for i in test_index]
            if not os.path.exists(cv_folder):
                os.makedirs(cv_folder)
            with open(
                os.path.join(cv_folder, "train_" + str(i) + ".pickle"), "wb"
            ) as handle:
                pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(
                os.path.join(cv_folder, "test_" + str(i) + ".pickle"), "wb"
            ) as handle:
                pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(
            list(common_sample_ids), test_size=test_size, random_state=42
        )
        train_test_folder = os.path.join(sample_folder, "NO_CV_train_test")
        if not os.path.exists(train_test_folder):
            os.makedirs(train_test_folder)
        # save train and test as pickle files
        with open(os.path.join(train_test_folder, "train.pickle"), "wb") as handle:
            pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(train_test_folder, "test.pickle"), "wb") as handle:
            pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return train_test_folder


def train_test_save(j, df, train, test, data_folder_path, i, CV):
    # split the data into train and test using the sample ids lists in train, test lists
    train = df[df.index.isin(train)]
    test = df[df.index.isin(test)]

    if CV:
        folder_name = "CV"
        folder_path = os.path.join(data_folder_path, folder_name)
        if not os.path.exists(folder_path):  # if the "CV" folder does not exist
            os.makedirs(folder_path)  # create the folder
        cv_folder = os.path.join(folder_path, "CV_" + str(i))
        if not os.path.exists(cv_folder):
            os.makedirs(cv_folder)
        train_file_name = str(j) + "_tr.csv"
        train_path = os.path.join(cv_folder, train_file_name)
        train.to_csv(train_path, header=None, index=None)
        print("Training set saved as `", train_file_name, "in the folder:", folder_path)

        test_file_name = str(j) + "_te.csv"
        test_path = os.path.join(cv_folder, test_file_name)
        test.to_csv(test_path, header=None, index=None)
        print("Test set saved as`", test_file_name, "in the folder:", folder_path)

    else:
        folder_name = "NO_CV"
        folder_path = os.path.join(data_folder_path, folder_name)
        if not os.path.exists(folder_path):  # if the "NO_CV" folder does not exist
            os.makedirs(folder_path)  # create the folder

        train_file_name = str(j) + "_tr.csv"
        train_path = os.path.join(folder_path, train_file_name)
        train.to_csv(train_path, header=None, index=None)
        print("Training set saved as `", train_file_name, "in the folder:", folder_path)

        test_file_name = str(j) + "_te.csv"
        test_path = os.path.join(folder_path, test_file_name)
        test.to_csv(test_path, header=None, index=None)
        print("Test set saved as`", test_file_name, "in the folder:", folder_path)

    return


# save labels from the labels dict
def save_labels(labels_dict, train, test, data_folder_path, i, CV=False):
    labels = pd.DataFrame.from_dict(labels_dict, orient="index")

    if CV:
        folder_name = "CV"
        folder_path = os.path.join(data_folder_path, folder_name)
        if not os.path.exists(folder_path):  # if the "CV" folder does not exist
            os.makedirs(folder_path)  # create the folder
        cv_folder = os.path.join(folder_path, "CV_" + str(i))
        if not os.path.exists(cv_folder):
            os.makedirs(cv_folder)
        label_train_path = os.path.join(cv_folder, "labels_tr.csv")
        label_test_path = os.path.join(cv_folder, "labels_te.csv")

        label_train = labels[labels.index.isin(train)]
        label_train.to_csv(label_train_path, header=None, index=None)
        print(
            "Labels for training set saved as `labels_tr.csv` in the folder:", cv_folder
        )
        label_test = labels[labels.index.isin(test)]
        label_test.to_csv(label_test_path, header=None, index=None)
        print("Label for test set saved as `labels_te.csv` in the folder:", cv_folder)

    else:
        folder_name = "NO_CV"
        folder_path = os.path.join(data_folder_path, folder_name)
        if not os.path.exists(folder_path):  # if the "NO_CV" folder does not exist
            os.makedirs(folder_path)  # create the folder

        label_train_path = os.path.join(folder_path, "labels_tr.csv")
        label_test_path = os.path.join(folder_path, "labels_te.csv")

        label_train = labels[labels.index.isin(train)]
        label_train.to_csv(label_train_path, header=None, index=None)
        print(
            "Labels for training set saved as `labels_tr.csv` in the folder:",
            label_train_path,
        )
        label_test = labels[labels.index.isin(test)]
        label_test.to_csv(label_test_path, header=None, index=None)
        print(
            "Label for test set saved as `labels_te.csv` in the folder:",
            label_test_path,
        )

    return


def save_sample_ids(sample_ids_dict, sample_folder):
    if not os.path.exists(sample_folder):  # if the sample folder does not exist
        os.makedirs(sample_folder)  # create the sample folder

    sample_ids_dict_save = os.path.join(sample_folder, "sample_ids_dict.pickle")

    if not os.path.exists(sample_ids_dict_save):
        # save sample_ids_dict dictionary as pickle file
        with open(
            sample_ids_dict_save, "wb"
        ) as handle:  # path to the sample_ids_dict pickle file
            pickle.dump(
                sample_ids_dict, handle, protocol=pickle.HIGHEST_PROTOCOL
            )  # save the sample_ids_dict pickle file

    return print("Sample ids saved as pickle file at: ", sample_folder)


def find_common_sample_ids(sample_ids_dict, sample_folder):
    common_samples = os.path.join(sample_folder, "common_sample_ids.pickle")
    if not os.path.exists(common_samples):
        common_sample_ids = set.intersection(*map(set, sample_ids_dict.values()))
        with open(
            common_samples, "wb"
        ) as handle:  # path to the common_sample_ids pickle file
            pickle.dump(
                common_sample_ids, handle, protocol=pickle.HIGHEST_PROTOCOL
            )  # save the common_sample_ids pickle file
    return


def omicwise_filtering(i, data_folder_path, common_sample_ids):
    """Filter the omic data based on the common sample ids"""
    omic_path = os.path.join(data_folder_path, str(i))
    processed_data_file_name = (
        str(i) + "_processed_data.csv"
    )  # name of the processed data file
    df = pd.read_csv(
        os.path.join(omic_path, processed_data_file_name), index_col=0
    )  # read the processed data file
    df = df[df.index.isin(common_sample_ids)]  # keep the common samples
    common_processed_data_file_name = (
        str(i) + "_common_processed_data.csv"
    )  # name of the common processed data file
    df.to_csv(
        os.path.join(omic_path, common_processed_data_file_name)
    )  # save the common processed data file

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


# select file from a folder which doest start with a number from 0-9
def select_file_from_folder(folder_path):
    file_list = os.listdir(folder_path)
    # if the file starts with a number from 0-9, remove it from the list
    for file in file_list:
        if file.startswith(tuple(map(str, range(10)))):
            file_list.remove(file)
    return file_list[0]


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


def get_pipelines(options, DF, model):
    pl_preprocessor = build_preprocessor_pipeline(DF)

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


def get_expname_datetime(options):
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%Y%m%d-%H%M%S")
    expname = options["model"] + "_" + options["mode"] + "_" + dt_string
    print("exp. name =" + expname)
    return expname


# print epoch loss
def print_epoch_loss(epoch, train_loss, test_loss):
    print("Epoch: %d, train loss: %f, test loss: %f" % (epoch, train_loss, test_loss))
    return


# key as x axis and values which are dictionaries as y axis
def plot_epoch_loss(epoch_loss_dict, fig_path):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import os

    # convert epoch loss dictionary to dataframe
    df = pd.DataFrame.from_dict(epoch_loss_dict)
    df = df.transpose()
    df = df.reset_index()
    df = df.rename(columns={"index": "epoch"})
    df = df.melt("epoch", var_name="cols", value_name="vals")
    # plot epoch loss
    plt.figure()
    ax = sns.lineplot(x="epoch", y="vals", hue="cols", data=df)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per epoch")
    plt.savefig(fig_path)
    plt.show()
    return


def result_plot(
    labels,
    prob,
    latest_model,
    cm_normalised=False,
):
    if not os.path.exists(latest_model):
        os.makedirs(latest_model)
    model_name = os.path.basename(os.path.normpath(latest_model))
    import scikitplot as skplt

    y = labels
    # yhat = yyhat_dict[key]["yhat"]
    yproba = prob
    # save each figure seprpately
    skplt.metrics.plot_confusion_matrix(
        y,
        yproba.argmax(1),
        normalize=cm_normalised,
        title="Confusion Matrix: " + model_name,
        figsize=(8, 8),
    )
    # save plot
    image_name = "mogonet_" + model_name + "_cm.png"  #
    figpath = os.path.join(latest_model, image_name)
    # save the above given path
    plt.savefig(figpath)

    skplt.metrics.plot_roc(y, yproba, title="ROC Plot: " + model_name)
    # save plot
    image_name = "mogonet_" + model_name + "_roc.png"  #
    figpath = os.path.join(latest_model, image_name)
    # save the above given path
    plt.savefig(figpath)

    skplt.metrics.plot_precision_recall(y, yproba, title="PR Curve: " + model_name)
    # save plot
    image_name = "mogonet_" + model_name + "_pr.png"  #
    figpath = os.path.join(latest_model, image_name)
    # save the above given path
    plt.savefig(figpath)

    skplt.metrics.plot_cumulative_gain(
        y, yproba, title="Cumulative Gains Chart: " + model_name
    )
    # save plot
    image_name = "mogonet_" + model_name + "_gain.png"  #
    figpath = os.path.join(latest_model, image_name)
    # save the above given path
    plt.savefig(figpath)

    skplt.metrics.plot_lift_curve(y, yproba, title="Lift Curve: " + model_name)
    # save plot
    image_name = "mogonet_" + model_name + "_lift.png"  #
    figpath = os.path.join(latest_model, image_name)
    # save the above given path
    plt.savefig(figpath)

    print(f"Saved figure: {figpath}")
