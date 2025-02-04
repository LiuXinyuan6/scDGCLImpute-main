# scDGCLImpute
scDGCLImpute：基于深度图对比表征的scRNA-seq缺失数据精准插补研究<br>
scDGCLImpute: Precise imputation of missing values in scRNA-seq data through deep contrastive learning on graph representations
# 1. 概述
单细胞RNA测序（scRNA-seq）技术为解析复杂组织中的细胞异质性和转录组动态提供了前所未有的分辨率。然而，受限于RNA捕获效率低和文库深度不足等技术因素，scRNA-seq数据常常存在缺失值，这对下游分析的准确性造成了显著影响。为了解决这一问题，本研究提出了一种基于深度图对比表征的单细胞RNA-seq 缺失数据插补模型scDGCLImpute (single-cell Deep Graph Contrastive Learning Imputation)。该模型通过构建细胞间K近邻图，利用对比学习策略学习细胞的低维嵌入表征，并基于这些表征计算细胞间的拓扑关系，最终采用局部线性重建原理对缺失值进行插补。在五个不同组织类型的真实scRNA-seq数据集和一个模拟数据集上的评估结果表明，scDGCLImpute 在恢复缺失值、促进细胞聚类以及重建细胞发育轨迹等方面显著优于现有方法，这表明该模型能够有效缓解缺失数据对scRNA-seq分析的负面影响，从而提升下游分析的准确性，促进单细胞转录组学研究的深入开展。
![image](https://github.com/user-attachments/assets/51a58f25-9b67-4e16-b5e9-a53f31784c6e)

# 2. 数据集
## 2.1 Pbmc数据集:
- 数据类型：健康供体外周血单核细胞测序的数据。
- 获取途径：10x Genomics官方平台 (https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/pbmc4k)

## 2.2 Jurkat-293t数据集:
- 数据类型：Jurkat和293t细胞系混合物的基因表达谱数据。
- 获取途径：10x Genomics数据库 (https://www.10xgenomics.com/datasets/50-percent-50-percent-jurkat-293-t-cell-mixture-1-standard-1-1-0)

## 2.3 Baron数据集:
- 数据类型：人类胰岛单细胞转录组数据。
- 获取途径：GEO数据库 (访问号：GSE84133)

## 2.4 Usoskin数据集:
- 数据类型：小鼠腰椎背根神经节单细胞转录组数据。
- 获取途径：GEO数据库 (访问号：GSE59739)

## 2.5 E3E7数据集:
- 数据类型：人类植入前胚胎发育细胞转录组数据。
- 获取途径：ArrayExpress数据库 (https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-3929)

## 2.6 模拟数据集(SimData):
- 采用Splatter R软件包生成完整的模拟数据集，具体参数设置见文章“**表2 Splatter R参数配置详情**”。

# 3. 运行指南
## 3.1 配置运行环境
```
joblib==1.3.2
matplotlib
numpy
pandas==2.0.3
PyYAML==6.0.1
scanpy==1.10.3
scikit_learn==1.3.2
scipy==1.14.1
setuptools==65.5.0
torch==2.3.0
torch_geometric==2.6.1
```

scDGCLImpute既可以在命令行中使用，也可以作为 Python 软件包使用。 以下说明可帮助您在本地计算机上快速使用它。

## 3.2 命令行方式
3.2.1 安装：通过克隆的方式安装最新的 GitHub 版本:
```bash
git clone https://github.com/LiuXinyuan6/scDGCLImpute.git
```
3.2.2 切换运行目录
```bash
cd scDGCLImpute
```
3.2.3 使用：训练和插补
```bash
python ./train.py --datasets juraket-293t.csv --dropout 0.4
```
参数说明：
```--datasets``` 待插补的数据集名称，其中行为基因列为细胞；**名称需要与‘config.yaml’文件中的名称对应**。
```--dropout```  缺失率，默认为0。
```--gpu_id```   是否使用GPU训练，默认为0。
```--config```   指定配置文件，默认为config.yaml。

当训练完成后，模型会自动保存训练好的模型（juraket-293t_model.pt）和嵌入矩阵（juraket-293t_embedding.npy）。同时会根据嵌入矩阵对缺失数据进行插补（imputation.csv）。以上文件均在运行目录中生成。

## 3.3 Python 软件包方式
```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Usoskin')
    parser.add_argument('--dropout', type=str, default='0') # 缺失率 ？%
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

    # 加载数据集
    normalize_path = data_path + "normalize_d" + args.dropout + ".csv"
    normalize, cells, genes = load_data(normalize_path)

    # 构建图
    data = get_adj(normalize, k=k_neighbor, pca= pca_num)
    print(data.edge_index.shape)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    encoder = Encoder(data.num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    print(model)

    # 加载预训练模型
    is_loadPr = False
    if is_loadPr:
        model = torch.load(data_path + args.dataset + '_model.pt')
        print("使用预训练模型 : " + data_path + args.dataset + '_model.pt')

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = t()
    prev = start
    model_learning = model_learning(model,data)

    if if_training:
        for epoch in range(1, num_epochs + 1):
            loss = model_learning.train()
            # 每隔 10 个 epoch 输出一次
            if epoch % 10 == 0:
                now = t()
                print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
                      f'this epoch {now - prev:.4f}, total {now - start:.4f}')
                prev = now

        model_learning.save_model()

    print("=== Impute ===")
    needImputed = pd.read_csv(data_path + args.dataset + "_d" + args.dropout + ".csv", index_col=0).T
    imputed_data, z = model_learning.impute(needImputed, if_training)
    # 保存插补数据
    pd.DataFrame(imputed_data.T, index=genes, columns=cells).to_csv(data_path + "imputation.csv")
    # 保存嵌入矩阵
    np.save(data_path + args.dataset + "_embedding.npy", z)
```
请确保待插补数据集的文件名命名格式为：{数据集名}_d{dropout}.csv和normalize_d{dropout}.csv。

# 4. 可以通过以下方式验证本文的实验结果
## 4.1 准确性
以jurkat-293t数据集为例：

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
    print("插补前pccs={:.3f}".format(pccs_non))
    print("插补前l1={:.3f}".format(l1_non))
    print("插补前rmse={:.3f}".format(rmse_non))
    print("===============")
    print("插补后pccs={:.3f}".format(pccs))
    print("插补后l1={:.3f}".format(l1))
    print("插补后rmse={:.3f}".format(rmse))
```
输出结果：
```python
    ===============
    插补前pccs=0.767
    插补前l1=0.693
    插补前rmse=5.705
    ===============
    插补后pccs=0.955
    插补后l1=0.381
    插补后rmse=2.597
```

## 4.2 聚类
以Usoskin数据集为例：

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
    print("K-means插补前ARI: {:.3f}".format(ari_kmeans_ori)
    print("K-means插补前NMI: {:.3f}".format(nmi_kmeans_ori)
    print("---")
    print("K-means插补后ARI: {:.3f}".format(ari_kmeans_imp)
    print("K-means插补后NMI: {:.3f}".format(nmi_kmeans_imp)
```
输出结果：
```python
    ===
    K-means插补前ARI: 0.808
    K-means插补前NMI: 0.778
    ---
    K-means插补后ARI: 0.951
    K-means插补后NMI: 0.922
```



