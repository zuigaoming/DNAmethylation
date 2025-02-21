# 人类精液年龄推断模型-表观遗传标记

## 📂 目录结构

```bash
.
├── data/                   # 数据目录
│   ├── Train_data.csv      # 训练数据集
│   └── new_data.csv        # 待预测数据
├── model_artifacts/        # 模型文件（自动生成）
├── model_train.py          # 模型训练脚本
├── model_predict.py        # 模型预测脚本
└── requirements.txt        # 依赖库列表
# 创建虚拟环境（推荐）
python -m venv .venv

# 激活环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

 #环境验证
python -c "import sklearn; print(f'scikit-learn版本: {sklearn.__version__}')"
# 预期输出: scikit-learn版本: 1.2.2

#📊 数据规范
#训练数据要求
#文件格式:CSV(UTF-8编码)
#数据规范：
#最后一列为目标变量
#特征列命名需唯一且不含特殊字符
#示例格式：
#chr12_80853660	cg13872326	cg09785625	chr14_30407630_1	chrY_7858986-age_semen	cg18857873+1	chr10_71661611_age_semen	chr14_30407630	cg18857873
#78	67	87	91	89	100	79	96	87
#56	56	74	56	56	99	57	80	93
#53	71	79	69	89	98	68	88	85
#66	73	90	89	79	98	72	96	89
#56	52	82	87	64	98	65	95	89
...
...

#🚀 使用流程
#模型训练
python model_train.py

#生成文件说明：
model_artifacts/
├── full_pipeline.pkl     # 完整预处理管道
├── trained_model.pkl     # 训练好的模型
└── metadata.json         # 元数据文件

#执行预测
python model_predict.py --input=data/new_data.csv

#输出文件
predictions.csv           # 预测结果文件
validation_report.txt     # 数据验证报告

#🚨 常见问题
#问题现象	解决方案
#ValueError: 特征数量不匹配	检查数据列数是否与metadata.json中的feature_names一致
#TypeError: float() argument must be...	检查数据是否包含非数值字符
#KeyError: 'feature3'	验证预测数据是否包含所有训练时的特征列




