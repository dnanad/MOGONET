# agrregate all the scores in one df
import pandas as pd
import os
from termcolor import colored


def main(mogonet_model):
    # load the scores for mogonet for the mogonet_model
    data_folder = "discovery_cohort"  # "PE_cfRNA"
    CV = True
    n_splits = 5
    num_epoch_pretrain = 500
    num_epoch = 2500

    rootpath = os.path.dirname(os.path.realpath(__file__))
    data_folder_path = os.path.join(rootpath, data_folder)

    if CV:
        cv_folder_path = os.path.join(data_folder_path, "CV")
        model_folder = os.path.join(data_folder_path, "CV", "models")
        model_folder_path = os.path.join(rootpath, model_folder)
    else:
        nocv_folder_path = os.path.join(data_folder_path, "NO_CV")
        model_folder = os.path.join(data_folder_path, "NO_CV", "models")
        model_folder_path = os.path.join(rootpath, model_folder)

    # exact_model for mogonet
    numerical_identity = (
        str(num_epoch_pretrain) + "_" + str(num_epoch) + "_" + mogonet_model
    )

    exact_model = os.path.join(
        model_folder_path, str(num_epoch_pretrain) + "_" + str(num_epoch), mogonet_model
    )
    mogonet_score_file = "mogonet_" + numerical_identity + "_detail_scores.csv"
    score_path = os.path.join(exact_model, mogonet_score_file)
    # print(colored(score_path, "green"))
    scores_df_exact = pd.read_csv(score_path, index_col=0)
    # print(scores_df_exact)

    # load all the scores from the cml folders
    cml_scores_df = pd.DataFrame()
    for folder in os.listdir(model_folder_path):
        if folder.startswith("CML"):
            # go to each folder one by one
            folder_path = os.path.join(model_folder_path, folder)
            # open each foler at folder path and load the csv
            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder)
                for file in os.listdir(subfolder_path):
                    if file.endswith(".csv"):
                        score_path_cml = os.path.join(subfolder_path, file)
                        scores_df_cml = pd.read_csv(score_path_cml, index_col=0)
                        cml_scores_df = pd.concat(
                            [cml_scores_df, scores_df_cml], ignore_index=True
                        )
    # merge the two df
    final_scores = pd.concat([cml_scores_df, scores_df_exact], ignore_index=True)
    # save the final_scores at cv_folder_path

    final_scores.to_csv(os.path.join(cv_folder_path, "final_scores.csv"))


if __name__ == "__main__":
    main(mogonet_model="20230915-021156")
    # "20230914-223605")
