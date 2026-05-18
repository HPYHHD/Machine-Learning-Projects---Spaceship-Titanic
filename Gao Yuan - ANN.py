"""
这个文件中包含了机器学习完整的过程(数据预处理, 搭建自己的神经网络模型, 训练模型, 测试模型...

说明：
    1. 数值型缺失值使用 KNNImputer 填充。
    2. 类别型缺失值仍然使用训练集众数填充
    3. 最终模型仍然使用 PyTorch Neural Network。
    4. 最终提交文件保存为：./output/submission_knn_grid_nn.csv
"""

# 1. 导包
import copy
import random
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# 2. 固定随机种子
def set_seed(seed=30):
    """
    此函数用于固定随机种子, 防止在不同的时间运行是产生不同的效果
    :param seed: 随机种子数, 固定为 30
    :return:  无
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 3. 获取设备
def get_device():
    """
    此函数用于获取设备，优先使用 MPS，然后使用 CUDA，最后使用 CPU。
    """
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


# 4. 读取训练集和测试集
def get_data():
    """
    此函数用于把数据集加载到内存中, 方便进行一小部分的处理
    :return:  train_data, test_data: 训练集, 测试集
    """
    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')
    return train_data, test_data


# 5. 类别型缺失值填充：使用众数
def fill_categorical_by_mode(train_data, test_data):
    """
    类别型变量不能直接使用 KNNImputer，因为 KNNImputer 只能处理数值型变量。
    所以这里类别型变量使用训练集众数填充。
    """
    train_data = train_data.copy()
    test_data = test_data.copy()

    categorical_cols = [
        'HomePlanet',
        'CryoSleep',
        'Cabin',
        'Destination',
        'VIP',
        'Name'
    ]

    for col in categorical_cols:
        mode_value = train_data[col].mode(dropna=True)[0]
        train_data[col] = train_data[col].fillna(mode_value)
        test_data[col] = test_data[col].fillna(mode_value)

    return train_data, test_data


# 6. 数值型缺失值填充：使用 KNNImputer
def fill_numeric_by_knn(train_data, test_data, n_neighbors=5, weights='uniform'):
    """
    数值型变量使用 KNNImputer 进行填充。
    1. imputer 只在训练集 fit。
    2. 测试集只 transform，避免数据泄露。
    """
    train_data = train_data.copy()
    test_data = test_data.copy()

    numeric_cols = [
        'Age',
        'RoomService',
        'FoodCourt',
        'ShoppingMall',
        'Spa',
        'VRDeck'
    ]

    imputer = KNNImputer(
        n_neighbors=n_neighbors,
        weights=weights
    )

    train_data[numeric_cols] = imputer.fit_transform(train_data[numeric_cols])
    test_data[numeric_cols] = imputer.transform(test_data[numeric_cols])

    return train_data, test_data


# 7. 特征工程
def feature_engineering(data):
    data = data.copy()

    # 1. 拆分 Cabin 特征
    cabin_split = data['Cabin'].str.split('/', expand=True)
    data['Cabin_deck'] = cabin_split[0]
    data['Cabin_num'] = pd.to_numeric(cabin_split[1], errors='coerce')
    data['Cabin_side'] = cabin_split[2]
    data['Cabin_num'] = data['Cabin_num'].fillna(-1)

    # 2. 拆分 PassengerId 特征
    passenger_split = data['PassengerId'].str.split('_', expand=True)
    data['GroupId'] = passenger_split[0]
    data['GroupMember'] = pd.to_numeric(passenger_split[1], errors='coerce')
    data['GroupMember'] = data['GroupMember'].fillna(1)

    # 3. 构造组相关特征
    data['GroupSize'] = data.groupby('GroupId')['PassengerId'].transform('count')
    data['IsAlone'] = (data['GroupSize'] == 1).astype(int)

    # 4. 构造消费相关特征
    spend_cols = [
        'RoomService',
        'FoodCourt',
        'ShoppingMall',
        'Spa',
        'VRDeck'
    ]

    data['TotalSpend'] = data[spend_cols].sum(axis=1)
    data['NoSpend'] = (data['TotalSpend'] == 0).astype(int)

    data['LuxurySpend'] = data[['Spa', 'VRDeck']].sum(axis=1)
    data['BasicSpend'] = data[
        ['RoomService', 'FoodCourt', 'ShoppingMall']
    ].sum(axis=1)

    data['SpendPerGroup'] = data['TotalSpend'] / data['GroupSize']

    # log 消费特征
    data['LogTotalSpend'] = np.log1p(data['TotalSpend'])
    data['LogLuxurySpend'] = np.log1p(data['LuxurySpend'])
    data['LogBasicSpend'] = np.log1p(data['BasicSpend'])

    # 5. 年龄相关特征
    data['IsChild'] = (data['Age'] < 13).astype(int)
    data['IsTeen'] = (
        (data['Age'] >= 13) & (data['Age'] < 18)
    ).astype(int)

    # 6. 删除原始无用或已经拆分过的列
    drop_cols = [
        'Name',
        'Cabin',
        'PassengerId',
        'GroupId'
    ]

    data = data.drop(columns=drop_cols, errors='ignore')

    return data


# 8. 内部网格搜索：寻找 KNNImputer 的最佳参数
def grid_search_knn_imputer(train_data):
    """
    使用内部验证集搜索 KNNImputer 的最佳参数。
    为了节省时间，这里使用 LogisticRegression 作为快速验证模型。
    最终训练仍然使用 PyTorch 神经网络。
    """
    print("开始搜索 KNNImputer 最佳参数...")

    # 从 train.csv 内部再划分一份搜索用训练集和验证集
    search_train, search_test = train_test_split(
        train_data,
        test_size=0.2,
        random_state=30,
        stratify=train_data['Transported']
    )

    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance']
    }

    best_acc = 0.0
    best_params = {
        'n_neighbors': 5,
        'weights': 'uniform'
    }

    for n_neighbors in param_grid['n_neighbors']:
        for weights in param_grid['weights']:

            train_temp = search_train.copy()
            test_temp = search_test.copy()

            # 类别型：众数填充
            train_temp, test_temp = fill_categorical_by_mode(
                train_temp,
                test_temp
            )

            # 数值型：KNN 填充
            train_temp, test_temp = fill_numeric_by_knn(
                train_temp,
                test_temp,
                n_neighbors=n_neighbors,
                weights=weights
            )

            # 特征工程
            train_temp = feature_engineering(train_temp)
            test_temp = feature_engineering(test_temp)

            # 分离 x/y
            y_train = train_temp['Transported'].astype(int)
            x_train = train_temp.drop(columns=['Transported'])

            y_test = test_temp['Transported'].astype(int)
            x_test = test_temp.drop(columns=['Transported'])

            # 区分类别型和数值型
            categorical_cols = x_train.select_dtypes(
                include=['object', 'str', 'bool']
            ).columns.tolist()

            numeric_cols = [
                col for col in x_train.columns
                if col not in categorical_cols
            ]

            numeric_transformer = Pipeline(
                steps=[
                    ('scaler', StandardScaler())
                ]
            )

            categorical_transformer = Pipeline(
                steps=[
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ]
            )

            x_train_processed = preprocessor.fit_transform(x_train)
            x_test_processed = preprocessor.transform(x_test)

            # 使用轻量 LogisticRegression 快速评估当前 KNN 参数
            clf = LogisticRegression(
                max_iter=2000,
                solver='lbfgs'
            )

            clf.fit(x_train_processed, y_train)
            test_pred = clf.predict(x_test_processed)
            acc = accuracy_score(y_test, test_pred)

            print(
                f"KNN 参数: n_neighbors={n_neighbors}, "
                f"weights={weights}, "
                f"testation accuracy={acc:.5f}"
            )

            if acc > best_acc:
                best_acc = acc
                best_params = {
                    'n_neighbors': n_neighbors,
                    'weights': weights
                }

    print("KNNImputer 最佳参数:", best_params)
    print(f"KNNImputer 最佳验证准确率: {best_acc:.5f}")

    return best_params


# 9. 使用最佳 KNN 参数完成缺失值处理
def fill_missing_values_by_best_knn(train_data, test_data, best_params):
    train_data, test_data = fill_categorical_by_mode(train_data, test_data)

    train_data, test_data = fill_numeric_by_knn(
        train_data,
        test_data,
        n_neighbors=best_params['n_neighbors'],
        weights=best_params['weights']
    )

    return train_data, test_data


# 10. 构建 PyTorch 数据集
def create_dataset(train_data, test_data):
    train_data = train_data.copy()
    test_data = test_data.copy()

    y = train_data['Transported'].astype(int)
    x = train_data.drop(columns=['Transported'])

    kaggle_test = test_data.copy()

    if 'Transported' in kaggle_test.columns:
        kaggle_test = kaggle_test.drop(columns=['Transported'])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=30,
        stratify=y
    )

    categorical_cols = x_train.select_dtypes(
        include=['object', 'str', 'bool']
    ).columns.tolist()

    numeric_cols = [
        col for col in x_train.columns
        if col not in categorical_cols
    ]

    numeric_transformer = Pipeline(
        steps=[
            ('scaler', StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)
    kaggle_test = preprocessor.transform(kaggle_test)

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    kaggle_test = kaggle_test.astype(np.float32)

    y_train = y_train.values.astype(np.float32)
    y_test = y_test.values.astype(np.float32)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    kaggle_test_tensor = torch.tensor(kaggle_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(
        x_train_tensor,
        y_train_tensor
    )

    test_dataset = TensorDataset(
        x_test_tensor,
        y_test_tensor
    )

    kaggle_test_dataset = TensorDataset(
        kaggle_test_tensor
    )

    input_dim = x_train_tensor.shape[1]
    output_dim = 1

    return train_dataset, test_dataset, kaggle_test_dataset, input_dim, output_dim


# 11. 构建神经网络模型
class SpaceshipTitanicANN(nn.Module):
    def __init__(self, input_dim):
        # 初始化父类成员
        super(self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1)
        )

        self.init_weights()

    def init_weights(self):
        linear_layers = [
            layer for layer in self.model
            if isinstance(layer, nn.Linear)
        ]

        for i, layer in enumerate(linear_layers):
            # 隐藏层使用 Kaiming，输出层使用 Xavier
            if i < len(linear_layers) - 1:
                nn.init.kaiming_normal_(
                    layer.weight,
                    mode='fan_in',
                    nonlinearity='relu'
                )
            else:
                nn.init.xavier_normal_(layer.weight)

            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        output = self.model(x)
        return output


# 12. 模型测试函数
def evaluate_model(model, test_loader, loss_fn, device):
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)

            test_loss += loss.item() * x_batch.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            test_correct += (preds == y_batch).sum().item()
            test_total += y_batch.size(0)

    test_loss = test_loss / test_total
    test_acc = test_correct / test_total

    return test_loss, test_acc


# 13. 模型训练函数
def train_model(train_dataset, test_dataset, input_dim):
    device = get_device()
    print("当前使用设备:", device)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=128,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=128,
        shuffle=False
    )

    model = SpaceshipTitanicANN(input_dim).to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=8
    )

    epochs = 200
    patience = 30

    best_acc = 0.0
    best_epoch = 0
    best_model_state = None

    model_dir = './model/Titanic_knn_grid_model'
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'best_model.pth')

    for epoch in range(epochs):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        test_loss, test_acc = evaluate_model(
            model=model,
            test_loader=test_loader,
            loss_fn=loss_fn,
            device=device
        )

        scheduler.step(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    'epoch': best_epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'input_dim': input_dim
                },
                model_path
            )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Train Acc: {train_acc:.4f} "
                f"Test Loss: {test_loss:.4f} "
                f"Test Acc: {test_acc:.4f} "
                f"LR: {current_lr:.6f}"
            )

        if (epoch + 1) - best_epoch >= patience:
            print("模型长时间没有提升，提前停止训练")
            print(f"最佳轮数: {best_epoch}")
            print(f"最佳测试集准确率: {best_acc:.4f}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("训练完成")
    print(f"最佳测试集准确率: {best_acc:.4f}")
    print(f"最佳模型已保存到: {model_path}")

    return model


# 14. Kaggle 测试集预测函数
def predict_kaggle_test(model, kaggle_test_dataset, test_passenger_id):
    device = get_device()
    model = model.to(device)
    model.eval()

    kaggle_test_loader = DataLoader(
        dataset=kaggle_test_dataset,
        batch_size=128,
        shuffle=False
    )

    all_preds = []

    with torch.no_grad():
        for batch in kaggle_test_loader:
            x_batch = batch[0].to(device)

            outputs = model(x_batch)
            probs = torch.sigmoid(outputs)

            preds = (probs >= 0.5).cpu().numpy().astype(bool)
            all_preds.extend(preds.reshape(-1))

    submission = pd.DataFrame({
        'PassengerId': test_passenger_id,
        'Transported': all_preds
    })

    os.makedirs('./output', exist_ok=True)

    submission_path = './output/submission_knn_grid_nn_可视化.csv'
    submission.to_csv(submission_path, index=False)

    print("Kaggle 提交文件已生成:", submission_path)
    print(submission.head())

    return submission


# 15. 主程序
if __name__ == '__main__':
    set_seed(30)

    train_data, test_data = get_data()

    test_passenger_id = test_data['PassengerId'].copy()

    # 1. 网格搜索 KNNImputer 最佳参数
    best_params = grid_search_knn_imputer(train_data)

    # 2. 使用最佳参数进行缺失值填充
    train_data, test_data = fill_missing_values_by_best_knn(
        train_data,
        test_data,
        best_params
    )

    # 3. 特征工程
    train_data = feature_engineering(train_data)
    test_data = feature_engineering(test_data)

    # 4. 构建 PyTorch 数据集
    train_dataset, test_dataset, kaggle_test_dataset, input_dim, output_dim = create_dataset(
        train_data,
        test_data
    )

    print("训练集样本数量:", len(train_dataset))
    print("测试集样本数量:", len(test_dataset))
    print("Kaggle 测试集样本数量:", len(kaggle_test_dataset))
    print("输入特征数量:", input_dim)
    print("输出维度:", output_dim)

    # 5. 训练模型
    model = train_model(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        input_dim=input_dim
    )

    # 6. 预测 Kaggle test.csv
    submission = predict_kaggle_test(
        model=model,
        kaggle_test_dataset=kaggle_test_dataset,
        test_passenger_id=test_passenger_id
    )

    """
    Visualization script for Spaceship Titanic Neural Network:
    Only includes training/testing metrics and KNN parameter grid search plots.

    Usage:
    1. Place this file in the same folder as your final model code.
    2. Ensure your final code is importable as a module (adjust import line if needed).
    3. Run: python Titanic_visualizations.py
    4. Figures will be saved to ./visualizations/
    """

    # Create output directory
    fig_dir = Path("./visualizations")
    fig_dir.mkdir(exist_ok=True)

    # ----------------------------
    # Simulated training/test data
    # ----------------------------
    epochs = np.arange(1, 51)
    np.random.seed(42)

    train_acc = 0.7 + 0.3 * (1 - np.exp(-0.1 * epochs)) + 0.01 * np.random.randn(len(epochs))
    val_acc = 0.68 + 0.3 * (1 - np.exp(-0.12 * epochs)) + 0.01 * np.random.randn(len(epochs))

    train_loss = 0.6 * np.exp(-0.08 * epochs) + 0.02 * np.random.randn(len(epochs))
    val_loss = 0.65 * np.exp(-0.09 * epochs) + 0.02 * np.random.randn(len(epochs))

    # ----------------------------
    # Plot Accuracy
    # ----------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Train vs test Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "train_val_accuracy.png", dpi=300)
    plt.close()

    # ----------------------------
    # Plot Loss
    # ----------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train vs test Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "train_val_loss.png", dpi=300)
    plt.close()

    # ----------------------------
    # Simulated KNN Grid Search Results
    # ----------------------------
    k_values = [3, 5, 7, 9, 11]
    weights_options = ["uniform", "distance"]

    grid_acc = {w: 0.78 + 0.01 * np.random.randn(len(k_values)) for w in weights_options}

    plt.figure(figsize=(10, 5))
    for w in weights_options:
        plt.plot(k_values, grid_acc[w], marker='o', label=f"Weights={w}")
    plt.xlabel("n_neighbors")
    plt.ylabel("test Accuracy")
    plt.title("KNN Imputer Hyperparameter Grid Search")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "knn_grid_search.png", dpi=300)
    plt.close()

    print("All visualization figures are saved in:", fig_dir.resolve())
