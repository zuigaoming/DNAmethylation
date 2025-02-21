import pandas as pd
import joblib
import json
from pathlib import Path


def load_artifacts():
    artifacts = {
        "pipeline": joblib.load("model_artifacts/full_pipeline.pkl"),
        "model": joblib.load("model_artifacts/trained_model.pkl")
    }

    with open("model_artifacts/metadata.json") as f:
        artifacts["metadata"] = json.load(f)

    return artifacts


def validate_input(data, metadata):
    # 维度验证
    if data.shape[1] != len(metadata["feature_names"]):
        raise ValueError(
            f"特征数量不匹配，预期 {len(metadata['feature_names'])} 个，实际 {data.shape[1]} 个"
        )

    # 特征名验证
    if list(data.columns) != metadata["feature_names"]:
        missing = set(metadata["feature_names"]) - set(data.columns)
        extra = set(data.columns) - set(metadata["feature_names"])
        raise ValueError(
            f"特征名不匹配，缺失特征：{missing}，多余特征：{extra}"
        )


def predict(input_csv):
    artifacts = load_artifacts()

    # 读取数据
    raw_data = pd.read_csv('New_data.csv')
    validate_input(raw_data, artifacts["metadata"])

    # 数据处理
    processed_data = artifacts["pipeline"].transform(raw_data.values)

    # 预测
    predictions = artifacts["model"].predict(processed_data)

    # 结果格式化
    return pd.DataFrame({
        "prediction": predictions,
        "input_features": [str(row) for row in raw_data.values]
    })


if __name__ == "__main__":
    result = predict("New_data.csv")
    result.to_csv("predictions.csv", index=False)
