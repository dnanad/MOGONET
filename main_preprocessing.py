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
    data_folder = "Test_data"  # "PE_cfRNA"  # "Test_data"
    labels = "sraruntable_final.csv"
    index_column = "gene_num"
    columns_to_drop = ["gene_name"]
    test_size = 0.2

    rootpath = os.path.dirname(os.path.realpath(__file__))
    data_folder_path = os.path.join(rootpath, data_folder)

    labels_path = os.path.join(data_folder_path, "labels", labels)

    omics_list, omincs_number = find_numFolders_maxNumFolders(data_folder_path)

    sample_ids_dict = {}
    for i in omics_list:
        omic_path = os.path.join(data_folder_path, str(i))
        for f in os.scandir(omic_path):
            if f.is_file():
                raw_data_file = f.name
            raw_data_file_path = os.path.join(omic_path, raw_data_file)
            df = import_process_datafile(
                raw_data_file_path, columns_to_drop, index_column
            )
            # save the processed data
            processed_data_file_name = str(i) + "_processed_data.csv"
            df.to_csv(os.path.join(omic_path, processed_data_file_name))
            sample_ids_dict[i] = df.index

    # save sample ids
    sample_folder = os.path.join(data_folder_path, "samples")
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)
    # save sample_ids_dict dictionary as pickle file
    with open(os.path.join(sample_folder, "sample_ids_dict.pickle"), "wb") as handle:
        pickle.dump(sample_ids_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # compare the sample ids and keep the common ones
    common_sample_ids = set.intersection(*map(set, sample_ids_dict.values()))
    with open(os.path.join(sample_folder, "common_sample_ids.pickle"), "wb") as handle:
        pickle.dump(common_sample_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # split the common_samples into train and test
    train, test = train_test_split(common_sample_ids, test_size, sample_folder)

    for i in omics_list:
        omic_path = os.path.join(data_folder_path, str(i))
        processed_data_file_name = str(i) + "_processed_data.csv"
        df = pd.read_csv(os.path.join(omic_path, processed_data_file_name), index_col=0)
        df = df[df.index.isin(common_sample_ids)]
        common_processed_data_file_name = str(i) + "_common_processed_data.csv"
        df.to_csv(os.path.join(omic_path, common_processed_data_file_name))

        # save features
        save_feat_name(i, df, data_folder_path)

        # train test split
        train_test_save(i, df, train, test, data_folder_path)

    # labels
    labels_dict = get_label_dict(labels_path)
    save_labels(labels_dict, train, test, data_folder_path)

    print("Preprocessing completed!")
