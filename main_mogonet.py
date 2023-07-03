""" Example for MOGONET classification
"""
from train_test import train_test

if __name__ == "__main__":
    data_folder = "TEST_DATA"  # "ROSMAP"
    view_list = [1]  # [1,2,3]
    num_epoch_pretrain = 100
    num_epoch = 500
    test_interval = 50
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3

    if data_folder == "TEST_DATA":
        num_class = 2
        adj_parameter = 2
        dim_he_list = [400, 400, 200]
    if data_folder == "ROSMAP":
        num_class = 2
        adj_parameter = 2
        dim_he_list = [200, 200, 100]
    if data_folder == "BRCA":
        num_class = 5
        adj_parameter = 10
        dim_he_list = [400, 400, 200]

    train_test(
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
    )
