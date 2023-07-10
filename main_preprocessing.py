"""For preprocessing the data
"""
import os
from utils import (
    import_process_datafile,
    get_label_dict,
    save_feat_name,
    train_test_save,
    dataset_summary,
    find_numFolders_maxNumFolders,
)

if __name__ == "__main__":
    data_folder = "Test_data"
    labels = "sraruntable_final.csv"
    index_column = "gene_num"
    columns_to_drop = ["gene_name"]
    test_size = 0.2

    rootpath = os.path.dirname(os.path.realpath(__file__))
    data_folder_path = os.path.join(rootpath, data_folder)

    labels_path = os.path.join(data_folder_path, "labels", labels)
    labels_dict = get_label_dict(labels_path)

    omics_list, omincs_number = find_numFolders_maxNumFolders(data_folder_path)

    for i in omics_list:
        print(os.path.join(data_folder_path, str(i)))
        omic_path = os.path.join(data_folder_path, str(i))
        for f in os.scandir(omic_path):
            if f.is_file():
                raw_data_file = f.name
            raw_data_file_path = os.path.join(omic_path, raw_data_file)
            df = import_process_datafile(
                raw_data_file_path, columns_to_drop, index_column, labels_dict
            )
            # save features
            save_feat_name(i, df, data_folder_path)
            # train test split
            train_test_save(i, df, test_size, data_folder_path)

    # dataset_summary(data_folder)

    print("Preprocessing completed!")
