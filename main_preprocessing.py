"""For preprocessing the data
"""
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
)

if __name__ == "__main__":
    data_folder = (
        "Test_data"  # "PE_cfRNA"  # "Test_data"  # "PE_cfRNA_pre"  # "Test_data"
    )
    labels = "sraruntable_final.csv"
    index_column = "gene_name"
    columns_to_drop = ["gene_num"]
    test_size = 0.2

    # creta a path to the data folder
    rootpath = os.path.dirname(os.path.realpath(__file__))
    data_folder_path = os.path.join(rootpath, data_folder)

    # create a path to the labels
    labels_path = os.path.join(data_folder_path, "labels", labels)

    # find the number of omics and the maximum number of folders
    omics_list, omincs_number = find_numFolders_maxNumFolders(data_folder_path)

    # dictionary to store the sample ids
    sample_ids_dict = {}
    for i in omics_list:
        omic_path = os.path.join(data_folder_path, str(i))  # path to the omic folder
        for f in os.scandir(omic_path):  # iterate through the files in the omic folder
            if f.is_file():  # if the file is a file
                raw_data_file = f.name  # get the name of the file
            raw_data_file_path = os.path.join(
                omic_path, raw_data_file
            )  # path to the raw data file
            df = import_process_datafile(  # import and process the data
                raw_data_file_path, columns_to_drop, index_column
            )
            # save the processed data
            processed_data_file_name = (
                str(i) + "_processed_data.csv"
            )  # name of the processed data file
            df.to_csv(
                os.path.join(omic_path, processed_data_file_name)
            )  # save the processed data file
            sample_ids_dict[i] = df.index  # save the sample ids

    # save sample ids
    sample_folder = os.path.join(
        data_folder_path, "samples"
    )  # path to the sample folder
    if not os.path.exists(sample_folder):  # if the sample folder does not exist
        os.makedirs(sample_folder)  # create the sample folder
    # save sample_ids_dict dictionary as pickle file
    with open(
        os.path.join(sample_folder, "sample_ids_dict.pickle"), "wb"
    ) as handle:  # path to the sample_ids_dict pickle file
        pickle.dump(
            sample_ids_dict, handle, protocol=pickle.HIGHEST_PROTOCOL
        )  # save the sample_ids_dict pickle file

    # compare the sample ids and keep the common ones
    common_sample_ids = set.intersection(
        *map(set, sample_ids_dict.values())
    )  # get the common sample ids
    with open(
        os.path.join(sample_folder, "common_sample_ids.pickle"), "wb"
    ) as handle:  # path to the common_sample_ids pickle file
        pickle.dump(
            common_sample_ids, handle, protocol=pickle.HIGHEST_PROTOCOL
        )  # save the common_sample_ids pickle file

    # split the common_samples into train and test
    train, test = train_test_split(common_sample_ids, test_size, sample_folder)

    for i in omics_list:
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

        # save features
        save_feat_name(i, df, data_folder_path)  # save the feature names

        # train test split
        train_test_save(
            i, df, train, test, data_folder_path
        )  # save the train and test data

    # labels
    labels_dict = get_label_dict(labels_path)
    save_labels(labels_dict, train, test, data_folder_path)

    print("Preprocessing completed!")
