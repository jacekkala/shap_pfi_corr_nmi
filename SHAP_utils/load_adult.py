import pandas as pd

def load_adult_dataset(data_folder):
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]
    
    train_path = f"{data_folder}/adult.data"
    try:
        df_train = pd.read_csv(train_path, header=None, names=columns, na_values=" ?", skipinitialspace=True)
    except FileNotFoundError:
        train_path = f"{data_folder}/adult"
        df_train = pd.read_csv(train_path, header=None, names=columns, na_values=" ?", skipinitialspace=True)

    test_path = f"{data_folder}/adult.test"
    df_test = pd.read_csv(test_path, header=0, names=columns, na_values=" ?", skipinitialspace=True, comment='|')
    df_test['income'] = df_test['income'].str.replace('.', '', regex=False)
    
    df = pd.concat([df_train, df_test], ignore_index=True)
    return df

# my_df = load_adult_dataset(r"PATH_TO_DATA_FOLDER")
# my_df.head()