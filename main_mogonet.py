""" Example for MOGONET classification
"""
from train_test import train_test
from utils import save_model_dict, find_numFolders_maxNumFolders, plot_epoch_loss
import os
from datetime import datetime

if __name__ == "__main__":
    data_folder = "TEST_DATA"  # "PE_cfRNA"  # "TEST_DATA"  # "ROSMAP"
    model_folder = os.path.join(data_folder, "models")
    # os.makedirs(model_folder) if not os.path.exists(model_folder) else print(
    #     "Overwriting contents in existing dir"
    # )
    rootpath = os.path.dirname(os.path.realpath(__file__))
    data_folder_path = os.path.join(rootpath, data_folder)
    model_folder_path = os.path.join(rootpath, model_folder)

    num_epoch_pretrain = 50
    num_epoch = 250
    test_interval = 50
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3

    exp_epoch = str(num_epoch_pretrain) + "_" + str(num_epoch)
    model_describe = os.path.join(model_folder_path, exp_epoch)
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d-%H%M%S")
    latest_model = model_describe + "_" + dt_string
    fig_file_name = exp_epoch + "_" + dt_string + "_epoch_loss.png"
    fig_path = os.path.join(latest_model, fig_file_name)

    view_list, _ = find_numFolders_maxNumFolders(data_folder_path)
    # view_list = [1]
    num_view = len(view_list)

    if data_folder == "PE_cfRNA":
        num_class = 2
        adj_parameter = 2
        dim_he_list = [400, 400, 200]
    if data_folder == "TEST_DATA":
        num_class = 2  # number of classes
        adj_parameter = 2  # number of neighbors for each node
        dim_he_list = [400, 400, 200]  # hidden dimensions for each view
    if data_folder == "ROSMAP":
        num_class = 2
        adj_parameter = 2
        dim_he_list = [200, 200, 100]
    if data_folder == "BRCA":
        num_class = 5
        adj_parameter = 10
        dim_he_list = [400, 400, 200]

    model_dict, epoch_loss_dict = train_test(
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

    # save the model

    save_model_dict(latest_model, model_dict)
    plot_epoch_loss(epoch_loss_dict, fig_path)
