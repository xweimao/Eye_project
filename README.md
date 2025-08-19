# Eye State Classification Project

一个基于深度学习的眼部状态识别项目，能够自动识别三种眼部状态：吸毒者(stoner)、酗酒者(alcohol)和正常人(normal)。

## 项目概述

本项目实现了一个完整的机器学习流水线，包括：
- 自动化网络数据收集（2000张图片，比例1:1:2）
- 数据验证和清洗
- 图像预处理和特征提取
- 多种深度学习模型训练（CNN、ResNet、EfficientNet）
- 模型评估和性能分析

## 项目结构

```
Eye_project/
├── config.py                    # 项目配置文件
├── main.py                      # 主执行脚本
├── requirements.txt             # 依赖包列表
├── data_collection/             # 数据收集模块
│   ├── web_scraper.py          # 网络爬虫
│   └── data_validator.py       # 数据验证
├── preprocessing/               # 数据预处理模块
│   └── image_processor.py      # 图像处理
├── models/                      # 模型定义
│   ├── cnn_model.py            # CNN模型架构
│   └── saved_models/           # 保存的模型
├── training/                    # 模型训练
│   └── trainer.py              # 训练脚本
├── evaluation/                  # 模型评估
│   └── evaluator.py            # 评估脚本
├── data/                        # 数据目录
│   ├── raw/                    # 原始数据
│   └── processed/              # 处理后数据
└── logs/                        # 日志文件
```

## 安装和设置

### 1. 克隆项目
```bash
git clone https://github.com/xweimao/Eye_project.git
cd Eye_project
```

### 2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 系统要求
- Python 3.8+
- CUDA支持的GPU（推荐，用于加速训练）
- 至少8GB RAM
- 10GB可用磁盘空间

## 使用方法

### 快速开始 - 运行完整流水线
```bash
python main.py --step all
```

### 分步执行

#### 1. 数据收集
```bash
python main.py --step collect
```
自动从Google Images和Bing Images收集眼部图像。

#### 2. 数据验证
```bash
python main.py --step validate
```
验证图像质量，检测人脸和眼部，清理无效数据。

#### 3. 数据预处理
```bash
python main.py --step preprocess
```
提取眼部区域，调整图像尺寸，应用数据增强。

#### 4. 模型训练
```bash
python main.py --step train
```
训练TensorFlow和PyTorch模型。

#### 5. 模型评估
```bash
python main.py --step evaluate
```
评估模型性能，生成详细报告和可视化结果。

## 配置选项

主要配置参数在 `config.py` 中：

```python
# 数据收集设置
TOTAL_IMAGES = 2000
CLASS_DISTRIBUTION = {
    'stoner': 500,    # 吸毒者
    'alcohol': 500,   # 酗酒者
    'normal': 1000    # 正常人
}

# 模型训练设置
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
```

## 模型架构

项目支持多种模型架构：

### 1. 自定义CNN
- 4层卷积网络
- 批量归一化
- Dropout正则化

### 2. ResNet18（迁移学习）
- 预训练ImageNet权重
- 自定义分类头

### 3. EfficientNet-B0（迁移学习）
- 高效的卷积神经网络
- 优秀的准确率/参数比

## 数据集

### 数据来源
- Google Images API
- Bing Images API
- 自动化关键词搜索

### 数据分布
- **吸毒者眼部图像**: 500张 (25%)
- **酗酒者眼部图像**: 500张 (25%)
- **正常人眼部图像**: 1000张 (50%)

### 数据预处理
- 人脸检测和眼部区域提取
- 图像尺寸标准化 (224x224)
- 数据增强（旋转、翻转、亮度调整等）
- 数据集分割（训练70%、验证20%、测试10%）

## 评估指标

项目提供全面的模型评估：

- **准确率 (Accuracy)**
- **精确率 (Precision)**
- **召回率 (Recall)**
- **F1分数 (F1-Score)**
- **混淆矩阵 (Confusion Matrix)**
- **ROC曲线和AUC**
- **类别分布分析**

## 结果可视化

自动生成的可视化包括：
- 训练历史曲线
- 混淆矩阵热图
- ROC曲线
- 类别分布对比
- 模型性能对比

## 注意事项

### 伦理和法律考虑
- 本项目仅用于学术研究和技术演示
- 请遵守当地法律法规和隐私政策
- 不应用于歧视或侵犯个人隐私
- 图像数据来源于公开网络资源

### 技术限制
- 模型准确性受训练数据质量影响
- 需要大量计算资源进行训练
- 实际应用需要更多医学专业验证

## 故障排除

### 常见问题

1. **内存不足错误**
   ```bash
   # 减少批次大小
   BATCH_SIZE = 16  # 在config.py中修改
   ```

2. **CUDA错误**
   ```bash
   # 检查CUDA安装
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **网络连接问题**
   ```bash
   # 使用代理或VPN
   export HTTP_PROXY=http://proxy:port
   export HTTPS_PROXY=https://proxy:port
   ```

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 联系方式

- 项目维护者: XMao
- Email: 37339423+xweimao@users.noreply.github.com
- GitHub: [@xweimao](https://github.com/xweimao)

## 致谢

- OpenCV团队提供的计算机视觉库
- TensorFlow和PyTorch深度学习框架
- 所有开源贡献者

---

**免责声明**: 本项目仅供学习和研究使用，不应用于任何可能造成歧视或伤害的场景。使用者需要承担相应的法律和伦理责任。
