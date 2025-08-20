"""
医学眼科疾病AI模型训练器
Medical Eye Disease AI Model Trainer
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json
from tqdm import tqdm
import config

class MedicalEyeTrainer:
    def __init__(self):
        self.medical_data_dir = os.path.join(config.RAW_DATA_DIR, "medical")
        self.models_dir = os.path.join("models", "medical_models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 医学类别
        self.medical_classes = {
            'normal': 0,
            'diabetic_retinopathy': 1,
            'hypertensive_retinopathy': 2,
            'age_macular_degeneration': 3
        }
        
        self.class_names = list(self.medical_classes.keys())
        
    def extract_medical_features(self, image_path):
        """提取医学图片特征"""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_array = np.array(img)
                height, width, channels = img_array.shape
                
                # 基础统计特征
                features = []
                
                # 1. 颜色统计特征
                for channel in range(3):  # RGB
                    channel_data = img_array[:, :, channel]
                    features.extend([
                        np.mean(channel_data),      # 均值
                        np.std(channel_data),       # 标准差
                        np.median(channel_data),    # 中位数
                        np.min(channel_data),       # 最小值
                        np.max(channel_data),       # 最大值
                    ])
                
                # 2. 颜色比率特征（医学诊断重要）
                r_channel = img_array[:, :, 0].astype(float)
                g_channel = img_array[:, :, 1].astype(float)
                b_channel = img_array[:, :, 2].astype(float)
                
                # 避免除零
                g_channel_safe = np.where(g_channel == 0, 1, g_channel)
                b_channel_safe = np.where(b_channel == 0, 1, b_channel)
                
                features.extend([
                    np.mean(r_channel / g_channel_safe),    # 红绿比
                    np.mean(r_channel / b_channel_safe),    # 红蓝比
                    np.mean(g_channel / b_channel_safe),    # 绿蓝比
                ])
                
                # 3. 亮度和对比度特征
                gray = np.mean(img_array, axis=2)
                features.extend([
                    np.mean(gray),                          # 平均亮度
                    np.std(gray),                           # 亮度标准差
                    np.max(gray) - np.min(gray),            # 对比度
                ])
                
                # 4. 纹理特征（简化版）
                # 水平梯度
                grad_x = np.abs(np.diff(gray, axis=1))
                features.append(np.mean(grad_x))
                
                # 垂直梯度
                grad_y = np.abs(np.diff(gray, axis=0))
                features.append(np.mean(grad_y))
                
                # 5. 区域特征（中心vs边缘）
                center_h, center_w = height // 2, width // 2
                center_region = img_array[center_h-50:center_h+50, center_w-50:center_w+50]
                if center_region.size > 0:
                    features.extend([
                        np.mean(center_region[:, :, 0]),    # 中心红色
                        np.mean(center_region[:, :, 1]),    # 中心绿色
                        np.mean(center_region[:, :, 2]),    # 中心蓝色
                    ])
                else:
                    features.extend([0, 0, 0])
                
                # 6. 血管相关特征（红色成分分析）
                red_threshold = np.percentile(r_channel, 75)
                red_regions = r_channel > red_threshold
                features.extend([
                    np.sum(red_regions) / (width * height), # 红色区域比例
                    np.mean(r_channel[red_regions]) if np.any(red_regions) else 0,  # 红色区域平均值
                ])
                
                # 7. 病变检测特征
                # 检测异常亮点（可能的渗出物）
                bright_threshold = np.percentile(gray, 95)
                bright_spots = gray > bright_threshold
                features.append(np.sum(bright_spots) / (width * height))
                
                # 检测暗点（可能的出血点）
                dark_threshold = np.percentile(gray, 5)
                dark_spots = gray < dark_threshold
                features.append(np.sum(dark_spots) / (width * height))
                
                return np.array(features)
                
        except Exception as e:
            print(f"特征提取错误 {image_path}: {e}")
            return np.zeros(26)  # 返回零特征向量
    
    def load_medical_dataset(self):
        """加载医学数据集"""
        print("📊 加载医学眼科疾病数据集...")
        
        X = []
        y = []
        image_paths = []
        
        for class_name, class_label in self.medical_classes.items():
            class_dir = os.path.join(self.medical_data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"⚠️  类别目录不存在: {class_dir}")
                continue
            
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"📁 {class_name}: {len(image_files)} 张图片")
            
            for filename in tqdm(image_files, desc=f"处理 {class_name}"):
                filepath = os.path.join(class_dir, filename)
                
                # 提取特征
                features = self.extract_medical_features(filepath)
                
                if features is not None and len(features) > 0:
                    X.append(features)
                    y.append(class_label)
                    image_paths.append(filepath)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"📊 数据集加载完成:")
        print(f"  样本数量: {len(X)}")
        print(f"  特征维度: {X.shape[1] if len(X) > 0 else 0}")
        print(f"  类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y, image_paths
    
    def train_medical_models(self, X, y):
        """训练医学诊断模型"""
        print("\n🤖 训练医学眼科疾病诊断模型...")
        
        if len(X) == 0:
            print("❌ 没有数据可训练")
            return {}
        
        # 数据预处理
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 标签编码
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"📊 训练集: {len(X_train)} 样本")
        print(f"📊 测试集: {len(X_test)} 样本")
        
        # 定义模型
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42,
                class_weight='balanced',
                probability=True
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\n🔬 训练 {model_name} 模型...")
            
            try:
                # 训练模型
                model.fit(X_train, y_train)
                
                # 预测
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # 评估
                accuracy = accuracy_score(y_test, y_pred)
                
                # 交叉验证
                cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5)
                
                # 分类报告
                class_report = classification_report(
                    y_test, y_pred, 
                    target_names=[self.class_names[i] for i in label_encoder.classes_],
                    output_dict=True
                )
                
                # 混淆矩阵
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'cv_mean': np.mean(cv_scores),
                    'cv_std': np.std(cv_scores),
                    'classification_report': class_report,
                    'confusion_matrix': conf_matrix.tolist(),
                    'predictions': y_pred.tolist(),
                    'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
                }
                
                print(f"  ✅ {model_name}:")
                print(f"    准确率: {accuracy:.3f}")
                print(f"    交叉验证: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
                
            except Exception as e:
                print(f"  ❌ {model_name} 训练失败: {e}")
                continue
        
        # 保存预处理器
        joblib.dump(scaler, os.path.join(self.models_dir, 'medical_scaler.joblib'))
        joblib.dump(label_encoder, os.path.join(self.models_dir, 'medical_label_encoder.joblib'))
        
        return results, X_test, y_test, scaler, label_encoder
    
    def save_medical_models(self, results):
        """保存医学模型"""
        print("\n💾 保存医学诊断模型...")
        
        best_model = None
        best_accuracy = 0
        
        for model_name, result in results.items():
            if 'model' in result:
                # 保存模型
                model_path = os.path.join(self.models_dir, f'medical_{model_name.lower()}_model.joblib')
                joblib.dump(result['model'], model_path)
                
                # 保存结果
                result_copy = result.copy()
                del result_copy['model']  # 移除模型对象
                
                result_path = os.path.join(self.models_dir, f'medical_{model_name.lower()}_results.json')
                with open(result_path, 'w') as f:
                    json.dump(result_copy, f, indent=2)
                
                print(f"  ✅ {model_name}: {model_path}")
                
                # 记录最佳模型
                if result['accuracy'] > best_accuracy:
                    best_accuracy = result['accuracy']
                    best_model = model_name
        
        if best_model:
            print(f"🏆 最佳模型: {best_model} (准确率: {best_accuracy:.3f})")
        
        return best_model, best_accuracy
    
    def generate_medical_report(self, results):
        """生成医学诊断报告"""
        print("\n📋 生成医学诊断模型报告...")
        
        report = {
            'dataset_info': {
                'classes': self.class_names,
                'class_mapping': self.medical_classes,
                'total_samples': sum(len(os.listdir(os.path.join(self.medical_data_dir, cls))) 
                                   for cls in self.class_names if os.path.exists(os.path.join(self.medical_data_dir, cls))),
                'features_count': 26
            },
            'model_performance': {},
            'feature_importance': {
                'color_statistics': '颜色统计特征 (RGB均值、标准差等)',
                'color_ratios': '颜色比率特征 (红绿比、红蓝比等)',
                'brightness_contrast': '亮度对比度特征',
                'texture_features': '纹理特征 (梯度)',
                'region_features': '区域特征 (中心vs边缘)',
                'vessel_features': '血管相关特征',
                'lesion_features': '病变检测特征'
            }
        }
        
        for model_name, result in results.items():
            if 'accuracy' in result:
                report['model_performance'][model_name] = {
                    'accuracy': result['accuracy'],
                    'cv_mean': result['cv_mean'],
                    'cv_std': result['cv_std'],
                    'classification_report': result['classification_report']
                }
        
        # 保存报告
        report_path = os.path.join(self.models_dir, 'medical_training_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 报告已保存: {report_path}")
        
        return report
    
    def train_complete_medical_system(self):
        """训练完整的医学诊断系统"""
        print("🏥 医学眼科疾病AI诊断系统训练")
        print("=" * 60)
        print("🎯 目标: 训练四类眼科疾病诊断模型")
        print("📋 类别: 正常、糖尿病视网膜病变、高血压视网膜病变、老年黄斑变性")
        
        # 加载数据
        X, y, image_paths = self.load_medical_dataset()
        
        if len(X) == 0:
            print("❌ 没有找到医学图片数据")
            return None
        
        # 训练模型
        results, X_test, y_test, scaler, label_encoder = self.train_medical_models(X, y)
        
        if not results:
            print("❌ 模型训练失败")
            return None
        
        # 保存模型
        best_model, best_accuracy = self.save_medical_models(results)
        
        # 生成报告
        report = self.generate_medical_report(results)
        
        print(f"\n🎉 医学诊断系统训练完成!")
        print("=" * 60)
        print(f"📊 训练结果:")
        for model_name, result in results.items():
            if 'accuracy' in result:
                print(f"  {model_name:20s}: {result['accuracy']:.3f} 准确率")
        
        print(f"🏆 最佳模型: {best_model} ({best_accuracy:.3f})")
        print(f"💾 模型保存位置: {self.models_dir}")
        
        return results

def main():
    """主函数"""
    print("🏥 医学眼科疾病AI模型训练器")
    print("🎯 训练眼科疾病诊断模型")
    print("=" * 60)
    
    trainer = MedicalEyeTrainer()
    results = trainer.train_complete_medical_system()
    
    if results:
        print(f"\n💡 下一步:")
        print(f"  1. 测试模型: python medical_inference.py")
        print(f"  2. 推送到GitHub: git add . && git commit && git push")
    
    return results

if __name__ == "__main__":
    results = main()
