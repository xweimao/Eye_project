"""
医学眼科疾病AI推理测试器
Medical Eye Disease AI Inference Tester
"""

import os
import numpy as np
from PIL import Image
import joblib
import json
import random
from medical_ml_trainer import MedicalEyeTrainer

class MedicalEyeInference:
    def __init__(self):
        self.models_dir = os.path.join("models", "medical_models")
        self.medical_data_dir = os.path.join("data", "raw", "medical")
        
        # 类别名称
        self.class_names = ['normal', 'diabetic_retinopathy', 'hypertensive_retinopathy', 'age_macular_degeneration']
        self.class_descriptions = {
            'normal': '正常健康眼底',
            'diabetic_retinopathy': '糖尿病视网膜病变',
            'hypertensive_retinopathy': '高血压视网膜病变',
            'age_macular_degeneration': '老年黄斑变性'
        }
        
        # 加载模型和预处理器
        self.load_models()
    
    def load_models(self):
        """加载训练好的模型"""
        try:
            # 加载最佳模型 (RandomForest)
            model_path = os.path.join(self.models_dir, 'medical_randomforest_model.joblib')
            self.model = joblib.load(model_path)
            
            # 加载预处理器
            scaler_path = os.path.join(self.models_dir, 'medical_scaler.joblib')
            self.scaler = joblib.load(scaler_path)
            
            label_encoder_path = os.path.join(self.models_dir, 'medical_label_encoder.joblib')
            self.label_encoder = joblib.load(label_encoder_path)
            
            print("✅ 医学诊断模型加载成功")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self.model = None
            self.scaler = None
            self.label_encoder = None
    
    def extract_features(self, image_path):
        """提取图片特征（与训练时相同）"""
        trainer = MedicalEyeTrainer()
        return trainer.extract_medical_features(image_path)
    
    def predict_disease(self, image_path):
        """预测眼科疾病"""
        if self.model is None:
            return None, None, "模型未加载"
        
        try:
            # 提取特征
            features = self.extract_features(image_path)
            if features is None or len(features) == 0:
                return None, None, "特征提取失败"
            
            # 预处理
            features_scaled = self.scaler.transform([features])
            
            # 预测
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # 解码标签
            class_name = self.class_names[prediction]
            confidence = probabilities[prediction]
            
            return class_name, confidence, "预测成功"
            
        except Exception as e:
            return None, None, f"预测错误: {e}"
    
    def test_sample_images(self):
        """测试样本图片"""
        print("🧪 测试医学诊断模型性能")
        print("=" * 60)
        
        if self.model is None:
            print("❌ 模型未加载，无法测试")
            return
        
        total_correct = 0
        total_tested = 0
        class_results = {}
        
        for class_name in self.class_names:
            class_dir = os.path.join(self.medical_data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"⚠️  类别目录不存在: {class_dir}")
                continue
            
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                print(f"⚠️  {class_name} 类别没有图片")
                continue
            
            # 随机选择3张图片测试
            test_images = random.sample(image_files, min(3, len(image_files)))
            
            print(f"\n📁 测试 {class_name.upper()} 类别:")
            print(f"   📋 描述: {self.class_descriptions[class_name]}")
            
            class_correct = 0
            class_total = 0
            
            for filename in test_images:
                filepath = os.path.join(class_dir, filename)
                
                # 预测
                predicted_class, confidence, message = self.predict_disease(filepath)
                
                if predicted_class is not None:
                    is_correct = predicted_class == class_name
                    status = "✅" if is_correct else "❌"
                    
                    print(f"   {status} {filename}")
                    print(f"      预测: {self.class_descriptions.get(predicted_class, predicted_class)}")
                    print(f"      置信度: {confidence:.3f}")
                    print(f"      实际: {self.class_descriptions[class_name]}")
                    
                    if is_correct:
                        class_correct += 1
                        total_correct += 1
                    
                    class_total += 1
                    total_tested += 1
                else:
                    print(f"   ❌ {filename}: {message}")
            
            if class_total > 0:
                class_accuracy = class_correct / class_total
                class_results[class_name] = {
                    'correct': class_correct,
                    'total': class_total,
                    'accuracy': class_accuracy
                }
                print(f"   📊 {class_name} 准确率: {class_correct}/{class_total} ({class_accuracy:.1%})")
        
        # 总体结果
        if total_tested > 0:
            overall_accuracy = total_correct / total_tested
            
            print(f"\n📊 总体测试结果:")
            print("=" * 60)
            print(f"✅ 正确预测: {total_correct}")
            print(f"📊 总测试数: {total_tested}")
            print(f"🎯 总体准确率: {overall_accuracy:.1%}")
            
            print(f"\n📋 各类别详细结果:")
            for class_name, result in class_results.items():
                print(f"  {class_name:25s}: {result['correct']}/{result['total']} ({result['accuracy']:.1%})")
        
        return class_results
    
    def predict_single_image(self, image_path):
        """预测单张图片"""
        print(f"🔍 分析图片: {image_path}")
        print("=" * 60)
        
        if not os.path.exists(image_path):
            print(f"❌ 图片不存在: {image_path}")
            return
        
        # 预测
        predicted_class, confidence, message = self.predict_disease(image_path)
        
        if predicted_class is not None:
            print(f"📊 诊断结果:")
            print(f"  疾病类型: {self.class_descriptions.get(predicted_class, predicted_class)}")
            print(f"  置信度: {confidence:.3f} ({confidence:.1%})")
            print(f"  状态: {message}")
            
            # 显示图片信息
            try:
                with Image.open(image_path) as img:
                    print(f"\n📸 图片信息:")
                    print(f"  尺寸: {img.size[0]}x{img.size[1]}")
                    print(f"  模式: {img.mode}")
                    print(f"  文件大小: {os.path.getsize(image_path)} 字节")
            except:
                pass
            
        else:
            print(f"❌ 预测失败: {message}")
    
    def batch_predict(self, directory_path):
        """批量预测目录中的图片"""
        print(f"📁 批量分析目录: {directory_path}")
        print("=" * 60)
        
        if not os.path.exists(directory_path):
            print(f"❌ 目录不存在: {directory_path}")
            return
        
        image_files = [f for f in os.listdir(directory_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print("❌ 目录中没有图片文件")
            return
        
        print(f"📊 找到 {len(image_files)} 张图片")
        
        results = {}
        
        for filename in image_files:
            filepath = os.path.join(directory_path, filename)
            predicted_class, confidence, message = self.predict_disease(filepath)
            
            if predicted_class is not None:
                if predicted_class not in results:
                    results[predicted_class] = []
                
                results[predicted_class].append({
                    'filename': filename,
                    'confidence': confidence
                })
                
                print(f"✅ {filename}: {self.class_descriptions.get(predicted_class, predicted_class)} ({confidence:.3f})")
            else:
                print(f"❌ {filename}: {message}")
        
        # 统计结果
        print(f"\n📊 批量预测统计:")
        for class_name, predictions in results.items():
            count = len(predictions)
            avg_confidence = np.mean([p['confidence'] for p in predictions])
            print(f"  {self.class_descriptions.get(class_name, class_name)}: {count} 张 (平均置信度: {avg_confidence:.3f})")
    
    def show_model_info(self):
        """显示模型信息"""
        print("🏥 医学眼科疾病AI诊断系统")
        print("=" * 60)
        
        if self.model is None:
            print("❌ 模型未加载")
            return
        
        print("✅ 模型状态: 已加载")
        print(f"🤖 模型类型: {type(self.model).__name__}")
        print(f"📊 支持类别: {len(self.class_names)} 种眼科疾病")
        
        print(f"\n📋 诊断类别:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {i}: {self.class_descriptions[class_name]}")
        
        # 加载训练报告
        try:
            report_path = os.path.join(self.models_dir, 'medical_training_report.json')
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            print(f"\n📊 模型性能:")
            for model_name, performance in report['model_performance'].items():
                print(f"  {model_name}: {performance['accuracy']:.3f} 准确率")
        except:
            print("\n⚠️  无法加载训练报告")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='医学眼科疾病AI推理测试器')
    parser.add_argument('--test', action='store_true', help='测试样本图片')
    parser.add_argument('--image', type=str, help='预测单张图片')
    parser.add_argument('--batch', type=str, help='批量预测目录')
    parser.add_argument('--info', action='store_true', help='显示模型信息')
    
    args = parser.parse_args()
    
    inference = MedicalEyeInference()
    
    if args.test:
        inference.test_sample_images()
    elif args.image:
        inference.predict_single_image(args.image)
    elif args.batch:
        inference.batch_predict(args.batch)
    elif args.info:
        inference.show_model_info()
    else:
        # 默认运行测试
        inference.show_model_info()
        print("\n")
        inference.test_sample_images()

if __name__ == "__main__":
    main()
