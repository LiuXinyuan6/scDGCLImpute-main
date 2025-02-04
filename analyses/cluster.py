import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def validate_origin(original_data, true_label, label_num):
    print('start evaluating original_data....')
    # ------------kmeans clustering------------
    estimators = KMeans(n_clusters=label_num, random_state=42)
    est = estimators.fit(original_data)
    kmeans_pred = est.labels_
    nmi_kmeans = format(normalized_mutual_info_score(true_label, kmeans_pred), '.5f')
    ari_kmeans = format(adjusted_rand_score(true_label, kmeans_pred), '.5f')
    print('nmi and ari of original data with kmeans clustering: ',
          str(nmi_kmeans), str(ari_kmeans))

    return  nmi_kmeans,  ari_kmeans


def validate_imputation(imputed_data, true_label, label_num):
    print('start evaluating imputation....')
    # ------------kmeans clustering------------
    estimators = KMeans(n_clusters=label_num, random_state=42)
    est = estimators.fit(imputed_data)
    kmeans_pred = est.labels_
    nmi_kmeans = format(normalized_mutual_info_score(true_label, kmeans_pred), '.5f')
    ari_kmeans = format(adjusted_rand_score(true_label, kmeans_pred), '.5f')
    print('nmi and ari of imputed data with kmeans clustering: ',
          str(nmi_kmeans), str(ari_kmeans))

    return  nmi_kmeans,  ari_kmeans


if __name__ == '__main__':
    label_num = 4
    X_fea = pd.read_csv(r"normalize_d0.csv",index_col=0).T.values
    X_fea_1 = pd.read_csv(r"normalize_imputation.csv",index_col=0).T.values

    labels = pd.read_csv(r"label.csv", index_col=0)
    labels = labels.values.squeeze()
    print(labels.shape)

    nmi_kmeans_ori,  ari_kmeans_ori = validate_origin(X_fea, labels, label_num)
    nmi_kmeans_imp,  ari_kmeans_imp= validate_imputation(X_fea_1, labels, label_num)


    print("===")
    print("K-means imputation before ARI: ", ari_kmeans_ori)
    print("K-means imputation before NMI: ", nmi_kmeans_ori)
    print("---")
    print("K-means imputation after ARI: ", ari_kmeans_imp)
    print("K-means imputation after NMI: ", nmi_kmeans_imp)



