# scDGCLImpute
scDGCLImpute: Enhancing scRNA-seq Data Analysis with Precise Imputation of Missing Values through Deep Contrastive Learning on Graph Representations
# 1. Overview
Single-cell RNA sequencing (scRNA-seq) technology provides unprecedented resolution for elucidating cellular heterogeneity and transcriptomic dynamics within complex tissues. However, scRNA-seq data often suffers from missing values due to technical limitations, such as low RNA capture efficiency and insufficient library depth. These missing values can significantly impact the accuracy of downstream analyses. In this study, we proposed a novel imputation model for missing single-cell RNA data, termed scDGCLImpute (single-cell Deep Graph Contrastive Learning Imputation). scDGCLImpute constructs a K-nearest neighbor graph to represent the relationships between cells. Then, it employs contrastive learning strategies to derive low-dimensional embeddings of cells. These embeddings capture the underlying structure of the data and enable the model to learn the topological relationships among cells. Finally, scDGCLImpute imputes missing values using local linear reconstruction principles. Evaluation of scDGCLImpute using five real scRNA-seq datasets from diverse tissues and one simulated dataset demonstrates its superior performance over existing methods in imputing missing values, improving cell clustering, and enhancing the reconstruction of developmental trajectories. These findings indicate that scDGCLImpute can effectively mitigate the negative impact of missing data on scRNA-seq analyses, leading to more accurate downstream results and facilitating deeper exploration in single-cell transcriptomics.
![主图_英文](https://github.com/user-attachments/assets/d4f7ac37-531b-4970-b8ba-99125b4c962c)


# 2. Datasets
## 2.1 Pbmc:
- https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/pbmc4k

## 2.2 Jurkat-293t:
- https://www.10xgenomics.com/datasets/50-percent-50-percent-jurkat-293-t-cell-mixture-1-standard-1-1-0

## 2.3 Baron:
- Gene Expression Omnibus (GEO) under accession number GSE102827

## 2.4 Usoskin:
- Gene Expression Omnibus (GEO) under accession number GSE59739

## 2.5 E3E7:
- https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-3929

## 2.6 SimData:
- SimData was generated using Splatter. The specific parameters used to generate the simulated dataset are detailed in the table below.

| Name             | Value                        | Description                                      |
|------------------|------------------------------|--------------------------------------------------|
| seed_value       | 10086                        | Random number seed                              |
| nCells           | 3000                         | Total number of cells                           |
| nGenes           | 1500                         | Total number of genes                           |
| group.prob       | [0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.2] | Proportion of each cell type                   |
| de.prob          | [0.045, 0.045, 0.045, 0.045, 0.045, 0.045, 0.045] | Probability of differential gene expression     |
| de.facLoc        | 0.1                           | Localization parameter for differential expression |
| de.facScale      | 0.4                           | Scale parameter for differential expression      |
| dropout_mid      | [0.2, 1, 2, 3, 4]             | Midpoint parameters for the distribution of missing data rates |
| running_times    | [1, 2, 3, 4, 5]               | Number of simulation replicates                 |
| dropout.shape    | -1                            | Shape parameter for the distribution of missing data rates |


# 3. Running Guide
## 3.1 Environment Configuration
```
joblib==1.3.2
matplotlib==3.8.1
numpy==1.26.4
pandas==2.0.3
PyYAML==6.0.1
scanpy==1.10.3
scikit_learn==1.3.2
scipy==1.14.1
setuptools==65.5.0
torch==2.3.0
torch_geometric==2.6.1
```

scDGCLImpute can be used both from the command line and as a Python package. The following instructions will help you quickly get started on your local machine.

## 3.2 Command Line Usage
3.2.1 Installation: Install the latest GitHub version by cloning the repository:
```bash
git clone https://github.com/LiuXinyuan6/scDGCLImpute-main.git
```
3.2.2 Change to the project directory
```bash
cd scDGCLImpute-main
```
3.2.3 Training and Imputation
```bash
python ./train.py --datasets juraket-293t.csv --dropout 0.4
```
Parameter Explanation：
```--datasets``` The dataset name to be imputed, where rows are genes and columns are cells; the name must match the one in the ‘config.yaml’ file.
```--dropout```  Missing data rate, default is 0.
```--gpu_id```   Whether to use GPU for training, default is 0.
```--config```   Specify the configuration file, default is config.yaml.

After training, the model will automatically save the trained model (juraket-293t_model.pt) and the embedding matrix (juraket-293t_embedding.npy). It will also perform imputation on the missing data using the embedding matrix (imputation.csv). These files will be generated in the working directory.

## 3.3 Python Package Usage
```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Usoskin')
    parser.add_argument('--dropout', type=str, default='0')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    else:
        print("No GPU available, running on CPU")
    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    torch.manual_seed(config['seed'])
    random.seed(12345)

    if_training = config['if_training']
    learning_rate = config['learning_rate']
    ae_hidden = config['ae_hidden']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv, 'GATv2Conv': GATv2Conv})[config['base_model']]
    num_layers = config['num_layers']
    pca_num = config['pca_num']
    k_neighbor = config['k_neighbor']
    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']
    data_path = config['data_path']

    normalize_path = data_path + "normalize_d" + args.dropout + ".csv"
    normalize, cells, genes = load_data(normalize_path)

    data = get_adj(normalize, k=k_neighbor, pca= pca_num)
    print(data.edge_index.shape)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    encoder = Encoder(data.num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    print(model)

    is_loadPr = False
    if is_loadPr:
        model = torch.load(data_path + args.dataset + '_model.pt')

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = t()
    prev = start
    model_learning = model_learning(model,data)

    if if_training:
        for epoch in range(1, num_epochs + 1):
            loss = model_learning.train()
            if epoch % 10 == 0:
                now = t()
                print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
                      f'this epoch {now - prev:.4f}, total {now - start:.4f}')
                prev = now

        model_learning.save_model()

    print("=== Impute ===")
    needImputed = pd.read_csv(data_path + args.dataset + "_d" + args.dropout + ".csv", index_col=0).T
    imputed_data, z = model_learning.impute(needImputed, if_training)
    pd.DataFrame(imputed_data.T, index=genes, columns=cells).to_csv(data_path + "imputation.csv")
    np.save(data_path + args.dataset + "_embedding.npy", z)
```
Please ensure that the dataset filename follows the format: {dataset_name}_d{dropout}.csv and normalize_d{dropout}.csv.

# 4. Verifying
## 4.1 Accuracy

```python
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
    print("PCC before imputation = ", pccs_non)
    print("L1 before imputation = ", l1_non)
    print("RMSE before imputation = ", rmse_non)
    print("===============")
    print("PCC after imputation = ", pccs)
    print("L1 after imputation = ", l1)
    print("RMSE after imputation = ", rmse)
```
Output:
```python
    ===============
    PCC before imputation = 0.767
    L1 before imputation = 0.693
    RMSE before imputation = 5.705
    ===============
    PCC after imputation = 0.955
    L1 after imputation = 0.381
    RMSE after imputation = 2.597
```

## 4.2 Clustering

```python
    label_num = 4
    X_fea = pd.read_csv(r"normalize_d0.csv",index_col=0).T.values
    X_fea_1 = pd.read_csv(r"normalize_imputation.csv",index_col=0).T.values

    labels = pd.read_csv(r"label.csv", index_col=0)
    labels = labels.values.squeeze()
    print(labels.shape)

    nmi_kmeans_ori,  ari_kmeans_ori = validate_origin(X_fea, labels, label_num)
    nmi_kmeans_imp,  ari_kmeans_imp= validate_imputation(X_fea_1, labels, label_num)


    print("===")
    print("K-means before imputation ARI: {:.3f}".format(ari_kmeans_ori))
    print("K-means before imputation NMI: {:.3f}".format(nmi_kmeans_ori))
    print("---")
    print("K-means after imputation ARI: {:.3f}".format(ari_kmeans_imp))
    print("K-means after imputation NMI: {:.3f}".format(nmi_kmeans_imp))
```
Output:
```python
    ===
    K-means before imputation ARI: 0.808
    K-means before imputation NMI: 0.778
    ---
    K-means after imputation ARI: 0.951
    K-means after imputation NMI: 0.922
```



