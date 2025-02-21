# 优化后的model_train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
import joblib
import json
from pathlib import Path

# 改进点1：增加配置参数模块
CONFIG = {
    "data_path": "Train_data.csv",
    "test_size": 0.2,
    "random_state": 42,
    "model_params": {
        "C": 126.13963947523733,
        "epsilon": 0.011274188660777666,
        "gamma": 0.015935178517218353
    },
    "output_dir": "model_artifacts"
}


def create_processing_pipeline():
    """改进点2：构建完整数据处理管道"""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # 前置缺失值填充
        ('selector', VarianceThreshold(threshold=0)),  # 特征选择
        ('scaler', StandardScaler())
    ])


def save_metadata(X_train, feature_names, pipeline):
    """改进点3：系统化保存元数据"""
    metadata = {
        "feature_names": feature_names.tolist(),
        "train_data_shape": X_train.shape,
        "imputer_mean": pipeline.named_steps['imputer'].statistics_.tolist(),
        "selected_features": pipeline.named_steps['selector'].get_support().tolist()
    }

    Path(CONFIG['output_dir']).mkdir(exist_ok=True)
    with open(f"{CONFIG['output_dir']}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    # 改进点4：优化数据加载流程
    data = pd.read_csv(CONFIG['data_path'])
    original_features = data.columns[:-1]

    # 不直接删除NaN，由pipeline处理
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # 改进点5：构建完整数据处理流
    full_pipeline = create_processing_pipeline()
    X_processed = full_pipeline.fit_transform(X)

    # 保存处理管道
    joblib.dump(full_pipeline, f"{CONFIG['output_dir']}/full_pipeline.pkl")

    # 数据分割应放在特征工程之后
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state']
    )

    # 模型构建
    model = SVR(
        kernel='rbf',
        C=CONFIG['model_params']['C'],
        epsilon=CONFIG['model_params']['epsilon'],
        gamma=CONFIG['model_params']['gamma']
    )

    # 最终模型训练（在整个处理后的数据集上）
    final_model = Pipeline([
        ('model', model)
    ])
    final_model.fit(X_processed, y)

    # 保存完整模型
    joblib.dump(final_model, f"{CONFIG['output_dir']}/trained_model.pkl")

    # 保存元数据
    save_metadata(X_processed, original_features, full_pipeline)


if __name__ == "__main__":
    main()
