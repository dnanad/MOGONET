""" Example for MOGONET classification
"""
from train_test import train_test
from utils import save_model_dict, find_numFolders_maxNumFolders, plot_epoch_loss
import os
from datetime import datetime
import pickle
import pandas as pd

if __name__ == "__main__":
    data_folder = "discovery_cohort"  # "PE_cfRNA"  # "discovery_cohort"  # "PB2_cfRNA"  # "PE_cfRNA_pre"  # "TEST_DATA"  # "PE_cfRNA"  # "TEST_DATA"  # "PE_cfRNA"  # "TEST_DATA"  # "ROSMAP"
    CV = True
    n_splits = 5
    num_epoch_pretrain = 500
    num_epoch = 2500
    test_interval = 100
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3

    rootpath = os.path.dirname(os.path.realpath(__file__))
    data_folder_path = os.path.join(rootpath, data_folder)

    if CV:
        folder = os.path.join(data_folder_path, "CV")
        folder_path = os.path.join(rootpath, folder)
        model_folder = os.path.join(folder, "models")
        model_folder_path = os.path.join(rootpath, model_folder)
    else:
        folder = os.path.join(data_folder_path, "NO_CV")
        folder_path = os.path.join(rootpath, folder)
        model_folder = os.path.join(folder, "models")
        model_folder_path = os.path.join(rootpath, model_folder)

    exp_epoch = str(num_epoch_pretrain) + "_" + str(num_epoch)
    exp = os.path.join(model_folder_path, exp_epoch)

    now = datetime.now()
    dt_string = now.strftime("%Y%m%d-%H%M%S")
    latest_model = os.path.join(exp, dt_string)

    fig_file_name = exp_epoch + "_" + dt_string + "_epoch_loss.png"
    fig_path = os.path.join(latest_model, fig_file_name)

    view_list, _ = find_numFolders_maxNumFolders(data_folder_path)
    num_view = len(view_list)

    if data_folder == "disc_coho1":
        num_class = 2
        adj_parameter = 2
        dim_he_list = [400, 400, 200]
    if data_folder == "disc_coho3":
        num_class = 2
        adj_parameter = 2
        dim_he_list = [400, 400, 200]
    if data_folder == "disc_coho":
        num_class = 2
        adj_parameter = 2
        dim_he_list = [400, 400, 200]
    if data_folder == "discovery_cohort":
        num_class = 2
        adj_parameter = 2
        dim_he_list = [400, 400, 200]
    if data_folder == "PB2_cfRNA":
        num_class = 2
        adj_parameter = 2
        dim_he_list = [400, 400, 200]
    if data_folder == "PB2_meta":
        num_class = 2
        adj_parameter = 2
        dim_he_list = [400, 400, 200]
    if data_folder == "PE_cfRNA_pre":
        num_class = 2
        adj_parameter = 2
        dim_he_list = [400, 400, 200]
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

    if CV:
        scores_df_main = pd.DataFrame()
        for i in range(n_splits):
            cv_folder = os.path.join(folder_path, "CV_" + str(i))
            cv_model_path = os.path.join(latest_model, "CV_" + str(i))
            model_dict, epoch_loss_dict, test_info, scores_df = train_test(
                cv_folder,
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
                cv_model_path,
            )
            scores_df_main = pd.concat([scores_df_main, scores_df], ignore_index=True)

            if not os.path.exists(cv_model_path):
                os.makedirs(cv_model_path)
            # save_model
            save_model_dict(cv_model_path, model_dict)
            # plot_epoch_loss(epoch_loss_dict, fig_path)
            # pickle dump the dict test_info at the same location as the model
            with open(os.path.join(cv_model_path, "test_info.pickle"), "wb") as handle:
                pickle.dump(test_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
        scores_df_main["model"] = "mogonet"
        # save the scores
        score_filename = "mogonet_" + exp_epoch + "_" + dt_string + "_detail_scores.csv"
        scores_df_main.to_csv(os.path.join(latest_model, score_filename))
    else:
        model_dict, epoch_loss_dict, test_info = train_test(
            folder_path,
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
        )

        # save the model
        # print(test_info)

        save_model_dict(latest_model, model_dict)
        plot_epoch_loss(epoch_loss_dict, fig_path)
        # pickle dump the dict test_info at the same location as the model
        with open(os.path.join(latest_model, "test_info.pickle"), "wb") as handle:
            pickle.dump(test_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
