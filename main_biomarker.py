""" Example for biomarker identification
"""
import os
import copy
from feat_importance import cal_feat_imp, summarize_imp_feat
from utils import find_numFolders_maxNumFolders

if __name__ == "__main__":
    data_folder = "PE_cfRNA"  # "PE_cfRNA_pre"  # "TEST_DATA_copy"  # "ROSMAP" #"BRCA
    model_folder = os.path.join(data_folder, "models")
    exp_name = "500_2500_20230809-144552"
    rootpath = os.path.dirname(os.path.realpath(__file__))
    data_folder_path = os.path.join(rootpath, data_folder)
    model_folder_path = os.path.join(rootpath, model_folder)
    exp_path = os.path.join(model_folder_path, exp_name)

    view_list, _ = find_numFolders_maxNumFolders(data_folder_path)
    # view_list = [1]
    num_view = len(view_list)
    topn = 30

    if data_folder == "PE_cfRNA_pre":
        num_class = 2
        adj_parameter = 2
        dim_he_list = [400, 400, 200]
    if data_folder == "PE_cfRNA":
        num_class = 2
        adj_parameter = 2
        dim_he_list = [400, 400, 200]
    if data_folder == "TEST_DATA_copy":
        num_class = 2  # number of classes
        adj_parameter = 2  # number of neighbors for each node
        dim_he_list = [400, 400, 200]  # hidden dimensions for each view
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

    featimp_list_list = []
    # for rep in range(5):
    #     featimp_list = cal_feat_imp(
    #         data_folder, os.path.join(model_folder, str(rep + 1)), view_list, num_class
    #     )
    #     featimp_list_list.append(copy.deepcopy(featimp_list))
    featimp_list = cal_feat_imp(
        data_folder, exp_path, view_list, num_class, adj_parameter, dim_he_list
    )
    featimp_list_list.append(copy.deepcopy(featimp_list))

    summarize_imp_feat(featimp_list_list, exp_path, topn=topn)
