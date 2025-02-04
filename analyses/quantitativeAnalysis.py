import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

def RMSE(imputed_data, original_data):
    return np.sqrt(np.mean((original_data - imputed_data)**2))

def pearson_corr(imputed_data, original_data):
    Y = original_data
    fake_Y = imputed_data
    fake_Y, Y = fake_Y.reshape(-1), Y.reshape(-1)
    fake_Y_mean, Y_mean = np.mean(fake_Y), np.mean(Y)
    corr = (np.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
            np.sqrt(np.sum((fake_Y - fake_Y_mean) ** 2)) * np.sqrt(np.sum((Y - Y_mean) ** 2)))
    return corr

def l1_distance(imputed_data, original_data):
    return np.mean(np.abs(original_data-imputed_data))


if __name__ == '__main__':

    data_non_path = r"juraket-293t_d40.csv"
    imputed_data_path = "imputation.csv"
    original_data_path = r"juraket-293t.csv"

    data_non = pd.read_csv(data_non_path, sep=',', index_col=0).values
    imputed_data = pd.read_csv(imputed_data_path, sep=',', index_col=0).values
    original_data = pd.read_csv(original_data_path, sep=',', index_col=0).values


    # PCC
    pccs_non = pearson_corr(data_non,original_data)
    pccs = pearson_corr(imputed_data,original_data)
    # RMSE
    rmse_non = RMSE(data_non,original_data)
    rmse = RMSE(imputed_data,original_data)
    # L1
    l1_non = l1_distance(data_non,original_data)
    l1 = l1_distance(imputed_data,original_data)

    print("===============")
    print("PCC before imputation =", pccs_non)
    print("L1 before imputation =", l1_non)
    print("RMSE before imputation =", rmse_non)
    print("===============")
    print("PCC after imputation =", pccs)
    print("L1 after imputation =", l1)
    print("RMSE after imputation =", rmse)
