""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import os


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class GraphConvolution(nn.Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907

    Attributes
    ----------
    in_features : int
        number of input features
    out_features : int
        number of output features
    bias : bool
        whether to use bias or not

    Methods
    -------
    forward(x, adj)
        Forward pass of the GCN
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN_E(nn.Module):
    """GCN Encoder

    Attributes
    ----------
    in_dim : int
        number of input features
    hgcn_dim : list
        list of hidden dimensions of the GCN
    dropout : float
        dropout rate

    Methods
    -------
    forward(x, adj)
        Forward pass of the GCN Encoder
    """

    def __init__(self, in_dim, hgcn_dim, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hgcn_dim[0])
        self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
        self.gc3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        x = F.leaky_relu(x, 0.25)

        return x


class Classifier_1(nn.Module):
    """Classifier for 1 omics

    Attributes
    ----------
    in_dim : int
        number of input features
    out_dim : int
        number of output features

    Methods
    -------
    forward(x)
        Forward pass of the classifier
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class VCDN(nn.Module):
    """View Correlation Discovery Network

    Attributes
    ----------
    num_view : int
        number of views/data sources/omics
    num_cls : int
        number of classes
    hvcdn_dim : int
        hidden dimension of the VCDN

    Methods
    -------
    forward(in_list)
        Forward pass of the VCDN
    """

    def __init__(self, num_view, num_cls, hvcdn_dim):
        super().__init__()
        self.num_cls = num_cls
        self.model = nn.Sequential(
            nn.Linear(pow(num_cls, num_view), hvcdn_dim),
            nn.LeakyReLU(0.25),
            nn.Linear(hvcdn_dim, num_cls),
        )
        self.model.apply(xavier_init)

    def forward(self, in_list):
        """Forward pass of the VCDN

        Parameters
        ----------
        in_list : list
            list of input tensors

        Returns
        -------
        output : tensor
            output tensor
        """
        num_view = len(in_list)
        for i in range(num_view):
            in_list[i] = torch.sigmoid(in_list[i])
        x = torch.reshape(
            torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)),
            (-1, pow(self.num_cls, 2), 1),
        )
        for i in range(2, num_view):
            x = torch.reshape(
                torch.matmul(x, in_list[i].unsqueeze(1)),
                (-1, pow(self.num_cls, i + 1), 1),
            )
        vcdn_feat = torch.reshape(x, (-1, pow(self.num_cls, num_view)))
        output = self.model(vcdn_feat)

        return output


def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc, gcn_dopout=0.5):
    """Initialize the model dictionary

    Attributes
    ----------
    num_view : int
        number of views/data sources/omics
    num_class : int
        number of classes
    dim_list : list
        list of input dimensions of each view
    dim_he_list : list
        list of hidden dimensions of each view
    dim_hc : int
        hidden dimension of the classifier
    gcn_dopout : float, optional
        dropout rate of GCN, by default 0.5

    Returns
    -------
    model_dict : dict
        model dictionary
    """
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i + 1)] = GCN_E(
            dim_list[i], dim_he_list, gcn_dopout
        )  # GCN_E: Graph Convolutional Network Encoder
        model_dict["C{:}".format(i + 1)] = Classifier_1(
            dim_he_list[-1], num_class
        )  # Classifier_1: Classifier
    if num_view >= 2:
        model_dict["C"] = VCDN(
            num_view, num_class, dim_hc
        )  # VCDN: View Correlation Discovery Network
    return model_dict


def init_optim(num_view, model_dict, lr_e=1e-4, lr_c=1e-4):
    """Initialize the optimizer dictionary

    Attributes
    ----------
    num_view : int
        number of views/data sources/omics
    model_dict : dict
        model dictionary
    lr_e : float, optional
        learning rate of encoder, by default 1e-4
    lr_c : float, optional
        learning rate of classifier, by default 1e-4

    Returns
    -------
    optim_dict : dict
        optimizer dictionary
    """
    optim_dict = {}
    for i in range(num_view):
        optim_dict[
            "C{:}".format(i + 1)
        ] = torch.optim.Adam(  # Adam: A Method for Stochastic Optimization
            list(model_dict["E{:}".format(i + 1)].parameters())  # E: Encoder
            + list(model_dict["C{:}".format(i + 1)].parameters()),  # C: Classifier
            lr=lr_e,
        )
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(
            model_dict["C"].parameters(), lr=lr_c
        )  # C: Classifier
    return optim_dict


##############################################################################################################
###Classical Machine Learning Models###
##############################################################################################################
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


def construct_model(opt):
    if opt == "SVC":
        model = SVC(C=1.0, kernel="rbf", gamma="auto", probability=True)
        GSparameters = {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["rbf", "linear", "poly"],
            "model__gamma": ["auto", "scale"],
        }
    elif opt == "DTC":
        model = DecisionTreeClassifier(
            random_state=420,
            min_impurity_decrease=0.01,
            max_depth=3,
            min_samples_split=20,
        )
        GSparameters = {
            "model__min_impurity_decrease": [0.01, 0.1, 1],
            "model__max_depth": [3, 5, 7],
            "model__min_samples_split": [20, 50, 100],
        }
    elif opt == "RFC":
        model = RandomForestClassifier(
            random_state=420,
            min_impurity_decrease=0.01,
            max_depth=3,
            min_samples_split=20,
        )
        GSparameters = {
            "model__min_impurity_decrease": [0.01, 0.1, 1],
            "model__max_depth": [3, 5, 7],
            "model__min_samples_split": [20, 50, 100],
        }
    elif opt == "XGBC":
        model = XGBClassifier(
            random_state=420, learning_rate=0.1, n_estimators=100, max_depth=3
        )
        GSparameters = {
            "model__learning_rate": [0.01, 0.1, 1],
            "model__n_estimators": [50, 100, 150],
            "model__max_depth": [3, 5, 7],
        }

    else:
        raise (ValueError(f"model:{opt} option not defined"))

    return model, GSparameters
