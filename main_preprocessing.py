"""For preprocessing the data
"""
from genericpath import exists
import pandas as pd
import os
import pickle
from utils import (
    import_process_datafile,
    get_label_dict,
    save_feat_name,
    train_test_save,
    dataset_summary,
    find_numFolders_maxNumFolders,
    train_test_split,
    save_labels,
    save_sample_ids,
    find_common_sample_ids,
    omicwise_filtering,
)

if __name__ == "__main__":
    data_folder = (
        "Test_data"  # "PE_cfRNA"  # "Test_data"  # "PE_cfRNA_pre"  # "Test_data"
    )
    labels = "sraruntable_final.csv"
    index_column = "gene_name"
    columns_to_drop = ["gene_num"]

    CV = True  # False  # True #Cross validation
    n_splits = 5
    test_size = 0.2

    # creta a path to the data folder
    rootpath = os.path.dirname(os.path.realpath(__file__))
    data_folder_path = os.path.join(rootpath, data_folder)

    # create a path to the labels
    labels_path = os.path.join(data_folder_path, "labels", labels)

    # find the number of omics and the maximum number of folders
    omics_list, _ = find_numFolders_maxNumFolders(data_folder_path)

    # dictionary to store the sample ids
    sample_ids_dict = {}
    for i in omics_list:
        omic_path = os.path.join(data_folder_path, str(i))  # path to the omic folder
        processed_data_file_name = str(i) + "_processed_data.csv"
        processed_file_path = os.path.join(omic_path, processed_data_file_name)
        check_if_it_is_needed_to_process = os.path.exists(processed_file_path)
        if not check_if_it_is_needed_to_process:
            for f in os.scandir(
                omic_path
            ):  # iterate through the files in the omic folder
                if f.is_file():  # if the file is a file
                    raw_data_file = f.name  # get the name of the file
                raw_data_file_path = os.path.join(
                    omic_path, raw_data_file
                )  # path to the raw data file
                df = import_process_datafile(  # import and process the data
                    raw_data_file_path, columns_to_drop, index_column
                )
                # save the processed data

                df.to_csv(processed_file_path)  # save the processed data file
                sample_ids_dict[i] = df.index  # save the sample ids

    sample_folder = os.path.join(
        data_folder_path, "samples"
    )  # path to the sample folder

    # save sample ids
    save_sample_ids(sample_ids_dict, sample_folder)

    if not os.path.exists(sample_folder):
        # compare the sample ids and keep the common ones
        find_common_sample_ids(sample_ids_dict, sample_folder)

    # import pickle
    with open(os.path.join(sample_folder, "common_sample_ids.pickle"), "rb") as handle:
        common_sample_ids = pickle.load(handle)

    for j in omics_list:
        omicwise_filtering(j, data_folder_path, common_sample_ids)

    if CV:
        print("Cross validation")
        train_test_folder = train_test_split(
            common_sample_ids=common_sample_ids,
            test_size=test_size,
            sample_folder=sample_folder,
            n_splits=n_splits,
            CV=CV,
        )
        for i in range(n_splits):
            print("Split: ", i)
            cv_folder = os.path.join(train_test_folder, "CV_" + str(i))
            train = pd.read_pickle(
                os.path.join(cv_folder, "train_" + str(i) + ".pickle")
            )
            test = pd.read_pickle(os.path.join(cv_folder, "test_" + str(i) + ".pickle"))
            for j in omics_list:
                omic_path = os.path.join(data_folder_path, str(j))
                common_processed_data_file_name = str(j) + "_common_processed_data.csv"
                df = pd.read_csv(
                    os.path.join(omic_path, common_processed_data_file_name),
                    index_col=0,
                )
                save_feat_name(
                    j=j, df=df, data_folder_path=data_folder_path, i=i, CV=CV
                )
                train_test_save(
                    j=j,
                    df=df,
                    train=train,
                    test=test,
                    data_folder_path=data_folder_path,
                    i=i,
                    CV=CV,
                )
            labels_dict = get_label_dict(labels_path)
            save_labels(
                labels_dict=labels_dict,
                train=train,
                test=test,
                data_folder_path=data_folder_path,
                i=i,
                CV=CV,
            )
    else:
        CV = False
        print("No cross validation")
        # split  and save the common_samples into train and test
        train_test_folder = train_test_split(
            common_sample_ids=common_sample_ids,
            test_size=test_size,
            sample_folder=sample_folder,
            n_splits=None,
        )

        # open saved pickel files
        train = pd.read_pickle(os.path.join(train_test_folder, "train.pickle"))
        test = pd.read_pickle(os.path.join(train_test_folder, "test.pickle"))

        # omic wise save features and train test split
        for j in omics_list:
            omic_path = os.path.join(data_folder_path, str(j))
            common_processed_data_file_name = str(j) + "_common_processed_data.csv"
            df = pd.read_csv(
                os.path.join(omic_path, common_processed_data_file_name), index_col=0
            )

            # save features
            save_feat_name(
                j=j, df=df, data_folder_path=data_folder_path, i=None, CV=CV
            )  # save the feature names

            # train test split
            train_test_save(
                j=j,
                df=df,
                train=train,
                test=test,
                data_folder_path=data_folder_path,
                i=None,
                CV=CV,
            )  # save the train and test data

        # labels
        labels_dict = get_label_dict(labels_path)
        save_labels(
            labels_dict=labels_dict,
            train=train,
            test=test,
            data_folder_path=data_folder_path,
            i=None,
            CV=CV,
        )

    print("Preprocessing completed!")
