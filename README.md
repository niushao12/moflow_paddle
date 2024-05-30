# moflow_paddle

## 语言

- [中文](README.md)  
- [English](README_EN.md)


参考论文：
Zang, Chengxi, and Fei Wang. "MoFlow: an invertible flow model for generating molecular graphs." In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 617-626. 2020.
```
https://arxiv.org/abs/2006.10137
```
参考代码：
```
https://github.com/cwohk1/moflow_plus
```

## 0. 安装环境:
从项目中克隆代码，执行以下操作：
```
https://github.com/niushao12/moflow_paddle.git moflow_paddle
```
Python版本3.9.0，CUDA 11.2，进入代码安装环境
```
cd moflow_paddle
pip install -r requirements.txt
```

## 1. 数据预处理
从SMILES字符串生成分子图：
```
cd data
python data_preprocess.py --data_name qm9
python data_preprocess.py --data_name zinc250k
```

## 2. 模型训练
#### 训练QM9数据集模型：
```
cd mflow
python train_model.py --data_name qm9  --batch_size 256  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/qm9_64gnn_128-64lin_1-1mask_0d6noise_convlu1  --b_n_flow 10  --b_hidden_ch 128,128  --a_n_flow 27 --a_hidden_gnn 64  --a_hidden_lin 128,64  --mask_row_size_list 1 --mask_row_stride_list 1 --noise_scale 0.6 --b_conv_lu 1  2>&1 | tee qm9_64gnn_128-64lin_1-1mask_0d6noise_convlu1.log
```
#### 训练zinc250k数据集模型：
```
cd mflow
python train_model.py  --data_name zinc250k  --batch_size  256  --max_epochs 200 --gpu 0  --debug True  --save_dir=results/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask   --b_n_flow 10  --b_hidden_ch 512,512  --a_n_flow 38  --a_hidden_gnn 256  --a_hidden_lin  512,64   --mask_row_size_list 1 --mask_row_stride_list 1  --noise_scale 0.6  --b_conv_lu 2  2>&1 | tee zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask.log
```
#### 或者下载并使用我们在以下链接中训练好的模型：
```
链接：https://pan.baidu.com/s/19yz8WOxoNd0b4vnUWL8uNQ 
提取码：bvor 
```

## 3. 模型测试
### 3.1-实验：重构
#### 重构QM9数据集：
```
cd mflow
python generate.py --model_dir results/qm9_64gnn_128-64lin_1-1mask_0d6noise_convlu1 -snapshot model_snapshot_epoch_200 --gpu 0 --data_name qm9  --hyperparams-path moflow-params.json --batch-size 256 --reconstruct  2>&1 | tee qm9_reconstruct_results.txt
```
#### 重构zinc250k数据集：
```
cd mflow
python generate.py --model_dir results/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask  -snapshot model_snapshot_epoch_200 --gpu  0  --data_name zinc250k --hyperparams-path moflow-params.json --batch-size 256  --reconstruct   2>&1 | tee zinc250k_reconstruct_results.txt
```

### 3.2-实验：随机生成
#### 从潜空间中进行随机生成，使用QM9模型
10000个样本 * 5次：
```
python generate.py --model_dir results/qm9_64gnn_128-64lin_1-1mask_0d6noise_convlu1 -snapshot model_snapshot_epoch_200 --gpu 0 --data_name qm9 --hyperparams-path moflow-params.json --batch-size 10000 --temperature 0.85 --delta 0.05 --n_experiments 5 --save_fig false --correct_validity true 2>&1 | tee qm9_random_generation.log
```

#### 从潜空间中进行随机生成，使用zinc250k模型
10000个样本 * 5次：
```
python generate.py --model_dir results/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask  -snapshot model_snapshot_epoch_200 --gpu  0  --data_name zinc250k --hyperparams-path moflow-params.json   --temperature 0.85  --batch-size 10000 --n_experiments 5  --save_fig false --correct_validity true 2>&1 | tee zinc250k_random_generation.log
```

### 3.3-实验：插值生成和可视化
#### 在潜空间中进行插值，使用QM9模型
在两个分子之间进行插值（分子图）：
```
python generate.py --model_dir results/qm9_64gnn_128-64lin_1-1mask_0d6noise_convlu1 -snapshot model_snapshot_epoch_200 --gpu 0 --data_name qm9  --hyperparams-path moflow-params.json --batch-size 1000  --temperature 0.65   --int2point --inter_times 50  --correct_validity true 2>&1 | tee qm9_visualization_int2point.log
```

在分子网格中进行插值（分子图）：
```
python generate.py --model_dir results/qm9_64gnn_128-64lin_1-1mask_0d6noise_convlu1 -snapshot model_snapshot_epoch_200 --gpu 0 --data_name qm9  --hyperparams-path moflow-params.json --batch-size 1000  --temperature 0.65 --delta 5  --intgrid  --inter_times 40  --correct_validity true 2>&1 | tee qm9_visualization_intgrid.log
```

#### 在潜空间中进行插值，使用zinc250k模型
在两个分子之间进行插值（分子图）：
```
python generate.py --model_dir results/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask  -snapshot model_snapshot_epoch_200 --gpu  0  --data_name zinc250k --hyperparams-path moflow-params.json   --batch-size 1000  --temperature 0.65   --int2point --inter_times 50  --correct_validity true 2>&1 | tee zinc250k_visualization_int2point.log
```

在分子网格中进行插值（分子图）：
```
python generate.py --model_dir results/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask  -snapshot model_snapshot_epoch_200 --gpu  0  --data_name zinc250k --hyperparams-path moflow-params.json   --batch-size 1000  --temperature 0.65 --delta 5  --intgrid  --inter_times 40  --correct_validity true 2>&1 | tee zinc250k_visualization_intgrid.log
```
### 3.4-实验：分子优化和约束优化
#### 优化zinc250k的QED属性
#### 训练一个额外的MLP模型，从潜空间到QED属性
```python
python optimize_property.py -snapshot model_snapshot_epoch_200 --hyperparams_path moflow-params.json --batch_size 256 --model_dir results/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask --gpu 0 --max_epochs 3 --weight_decay 1e-3 --data_name zinc250k --hidden 16 --temperature 1.0 --property_name qed 2>&1 | tee training_optimize_zinc250k_qed.log
# 输出：一个用于优化的分子属性预测模型，例如命名为qed_model.pdparams
# 例如，保存qed回归模型到：results/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask/qed_model.pdparams
# 训练并保存模型完成！时间 477.87 秒
# 可以调整：
#         --max_epochs 3
#         --weight_decay 1e-3
#         --hidden 16
# 等等。
```
#### 或者下载并使用我们在以下链接中训练好的模型：
```
链接：https://pan.baidu.com/s/19yz8WOxoNd0b4vnUWL8uNQ 
提取码：bvor
```
#### 优化现有分子以获得优化后的QED得分的新分子
```python
python optimize_property.py -snapshot model_snapshot_epoch_200 --hyperparams_path moflow-params.json --batch_size 256 --model_dir results/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask --gpu 0 --data_name zinc250k --property_name qed --topk 2000 --property_model_path qed_model.pdparams --debug false --topscore 2>&1 | tee zinc250k_top_qed_optimized.log
# 输入：--property_model_path qed_model.pdparams 是回归模型
# 输出：生成优化后的和新颖的分子的排序列表，与qed相关
```

#### 约束优化zinc250k的plogp（或qed）+相似性属性
#### 训练一个额外的MLP模型，从潜空间到plogp属性
```python
python optimize_property.py -snapshot model_snapshot_epoch_200 --hyperparams_path moflow-params.json --batch_size 256 --model_dir results/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask --gpu 0 --max_epochs 3 --weight_decay 1e-2 --data_name zinc250k --hidden 16 --temperature 1.0 --property_name plogp 2>&1 | tee training_optimize_zinc250k_plogp.log
# 输出：一个用于优化的分子属性预测模型，例如命名为plogp_model.pdparams
# 例如，保存plogp回归模型到：results/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask/plogp_model.pdparams
# 训练并保存模型完成！时间 473.74 秒
# 可以调整：
#         --max_epochs 3
#         --weight_decay 1e-2
#         --hidden 16
# 等等。
```
#### 优化现有分子以获得优化后的plogp得分，并受到约束相似性的限制
```python
python optimize_property.py -snapshot model_snapshot_epoch_200 --hyperparams_path moflow-params.json --batch_size 256 --model_dir results/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask --gpu 0 --data_name zinc250k --property_name plogp --topk 800 --property_model_path qed_model.pdparams --consopt --sim_cutoff 0 2>&1 | tee zinc250k_constrain_optimize_plogp.log
# 输入：--property_model_path qed_model.pt或plogp_model.pt是回归模型
#        --sim_cutoff 0（或0.2、0.4等相似性）
#        --topk 800（选择前800个具有较差属性值的分子进行改进）
# 输出：
# 使用qed_model.pt优化plogp与
# 因为qed和plogp有一定的相关性，所以我们在两个优化任务中同时使用qed/plogp模型

更多配置，请参考代码optimize_property.py和论文中的优化章节。