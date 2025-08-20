"""
医学眼部图片生成器 - 创建四类眼科疾病的高质量合成图片
Medical Eye Image Generator - Creates high-quality synthetic images for 4 eye diseases
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import random
import colorsys
from tqdm import tqdm
import json
import config

class MedicalEyeGenerator:
    def __init__(self):
        # 创建医学数据目录
        self.medical_data_dir = os.path.join(config.RAW_DATA_DIR, "medical")
        os.makedirs(self.medical_data_dir, exist_ok=True)
        
        # 四类眼科疾病
        self.medical_classes = {
            'normal': 25,
            'diabetic_retinopathy': 25,
            'hypertensive_retinopathy': 25,
            'age_macular_degeneration': 25
        }
        
        # 为每个类别创建目录
        for class_name in self.medical_classes.keys():
            class_dir = os.path.join(self.medical_data_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
        
        # 疾病特征配置
        self.disease_features = {
            'normal': {
                'background_color': (255, 248, 240),  # 健康的眼底颜色
                'vessel_color': (180, 50, 50),        # 正常血管颜色
                'vessel_width': 2,
                'vessel_count': 8,
                'lesions': False,
                'hemorrhages': False,
                'exudates': False,
                'drusen': False,
                'description': '正常健康眼底'
            },
            'diabetic_retinopathy': {
                'background_color': (255, 230, 220),  # 略带红色
                'vessel_color': (200, 30, 30),        # 更红的血管
                'vessel_width': 3,
                'vessel_count': 12,
                'lesions': True,                      # 病变
                'hemorrhages': True,                  # 出血点
                'exudates': True,                     # 渗出物
                'drusen': False,
                'description': '糖尿病视网膜病变'
            },
            'hypertensive_retinopathy': {
                'background_color': (255, 235, 225),  # 轻微红色
                'vessel_color': (220, 40, 40),        # 血管变粗
                'vessel_width': 4,
                'vessel_count': 10,
                'lesions': True,
                'hemorrhages': True,
                'exudates': False,
                'drusen': False,
                'arterial_narrowing': True,           # 动脉狭窄
                'description': '高血压视网膜病变'
            },
            'age_macular_degeneration': {
                'background_color': (255, 245, 235),  # 略黄色
                'vessel_color': (160, 60, 60),        # 血管颜色正常
                'vessel_width': 2,
                'vessel_count': 8,
                'lesions': False,
                'hemorrhages': False,
                'exudates': False,
                'drusen': True,                       # 玻璃膜疣
                'macular_changes': True,              # 黄斑变化
                'description': '老年黄斑变性'
            }
        }
    
    def create_fundus_background(self, width, height, disease_type):
        """创建眼底背景"""
        features = self.disease_features[disease_type]
        base_color = features['background_color']
        
        # 创建渐变背景
        img = Image.new('RGB', (width, height), base_color)
        
        # 添加径向渐变效果
        center_x, center_y = width // 2, height // 2
        max_distance = min(width, height) // 2
        
        for y in range(height):
            for x in range(width):
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                if distance < max_distance:
                    # 从中心到边缘的渐变
                    factor = distance / max_distance
                    
                    # 调整颜色
                    new_color = tuple(
                        int(c * (1 - factor * 0.3)) for c in base_color
                    )
                    
                    img.putpixel((x, y), new_color)
        
        return img
    
    def draw_blood_vessels(self, img, disease_type):
        """绘制血管系统"""
        draw = ImageDraw.Draw(img)
        width, height = img.size
        center_x, center_y = width // 2, height // 2
        
        features = self.disease_features[disease_type]
        vessel_color = features['vessel_color']
        vessel_width = features['vessel_width']
        vessel_count = features['vessel_count']
        
        # 主要血管从中心放射出去
        for i in range(vessel_count):
            angle = (360 / vessel_count) * i + random.uniform(-15, 15)
            
            # 血管路径
            start_r = random.uniform(20, 40)
            end_r = random.uniform(min(width, height) * 0.3, min(width, height) * 0.45)
            
            # 起点和终点
            start_x = center_x + start_r * np.cos(np.radians(angle))
            start_y = center_y + start_r * np.sin(np.radians(angle))
            end_x = center_x + end_r * np.cos(np.radians(angle))
            end_y = center_y + end_r * np.sin(np.radians(angle))
            
            # 绘制主血管
            draw.line([start_x, start_y, end_x, end_y], 
                     fill=vessel_color, width=vessel_width)
            
            # 绘制分支血管
            branch_count = random.randint(2, 4)
            for j in range(branch_count):
                branch_factor = 0.3 + 0.4 * j / branch_count
                branch_x = start_x + (end_x - start_x) * branch_factor
                branch_y = start_y + (end_y - start_y) * branch_factor
                
                branch_angle = angle + random.uniform(-45, 45)
                branch_length = random.uniform(20, 40)
                
                branch_end_x = branch_x + branch_length * np.cos(np.radians(branch_angle))
                branch_end_y = branch_y + branch_length * np.sin(np.radians(branch_angle))
                
                draw.line([branch_x, branch_y, branch_end_x, branch_end_y],
                         fill=vessel_color, width=max(1, vessel_width - 1))
        
        # 高血压特有的动脉狭窄效果
        if disease_type == 'hypertensive_retinopathy':
            for i in range(4):
                x = random.randint(width//4, 3*width//4)
                y = random.randint(height//4, 3*height//4)
                
                # 绘制狭窄的血管段
                draw.ellipse([x-15, y-3, x+15, y+3], 
                           fill=(255, 100, 100), outline=(200, 50, 50))
    
    def add_disease_lesions(self, img, disease_type):
        """添加疾病特异性病变"""
        draw = ImageDraw.Draw(img)
        width, height = img.size
        features = self.disease_features[disease_type]
        
        # 糖尿病视网膜病变的病变
        if disease_type == 'diabetic_retinopathy':
            # 微动脉瘤（红点）
            for _ in range(random.randint(8, 15)):
                x = random.randint(50, width-50)
                y = random.randint(50, height-50)
                size = random.randint(2, 5)
                draw.ellipse([x-size, y-size, x+size, y+size], 
                           fill=(150, 0, 0))
            
            # 硬性渗出物（黄色斑点）
            for _ in range(random.randint(5, 10)):
                x = random.randint(100, width-100)
                y = random.randint(100, height-100)
                size = random.randint(3, 8)
                draw.ellipse([x-size, y-size, x+size, y+size], 
                           fill=(255, 255, 0))
            
            # 出血点
            for _ in range(random.randint(6, 12)):
                x = random.randint(80, width-80)
                y = random.randint(80, height-80)
                size = random.randint(4, 10)
                draw.ellipse([x-size, y-size, x+size, y+size], 
                           fill=(100, 0, 0))
        
        # 高血压视网膜病变的病变
        elif disease_type == 'hypertensive_retinopathy':
            # 火焰状出血
            for _ in range(random.randint(4, 8)):
                x = random.randint(100, width-100)
                y = random.randint(100, height-100)
                
                # 绘制火焰状出血
                points = []
                for angle in range(0, 360, 30):
                    r = random.uniform(8, 15)
                    px = x + r * np.cos(np.radians(angle))
                    py = y + r * np.sin(np.radians(angle))
                    points.append((px, py))
                
                draw.polygon(points, fill=(120, 0, 0))
            
            # 棉絮状斑点
            for _ in range(random.randint(3, 6)):
                x = random.randint(120, width-120)
                y = random.randint(120, height-120)
                size = random.randint(6, 12)
                draw.ellipse([x-size, y-size, x+size, y+size], 
                           fill=(255, 255, 255), outline=(200, 200, 200))
        
        # 老年黄斑变性的病变
        elif disease_type == 'age_macular_degeneration':
            center_x, center_y = width // 2, height // 2
            
            # 玻璃膜疣（黄色小点）
            for _ in range(random.randint(15, 25)):
                # 主要分布在黄斑区域
                angle = random.uniform(0, 360)
                distance = random.uniform(20, 80)
                x = center_x + distance * np.cos(np.radians(angle))
                y = center_y + distance * np.sin(np.radians(angle))
                
                size = random.randint(2, 6)
                draw.ellipse([x-size, y-size, x+size, y+size], 
                           fill=(255, 255, 150))
            
            # 黄斑区色素紊乱
            for _ in range(random.randint(8, 12)):
                angle = random.uniform(0, 360)
                distance = random.uniform(10, 60)
                x = center_x + distance * np.cos(np.radians(angle))
                y = center_y + distance * np.sin(np.radians(angle))
                
                size = random.randint(4, 8)
                draw.ellipse([x-size, y-size, x+size, y+size], 
                           fill=(139, 69, 19))  # 棕色色素
    
    def add_optic_disc(self, img):
        """添加视盘"""
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # 视盘通常位于中心偏鼻侧
        disc_x = width // 2 + random.randint(-50, -20)
        disc_y = height // 2 + random.randint(-30, 30)
        disc_size = random.randint(25, 40)
        
        # 绘制视盘
        draw.ellipse([disc_x-disc_size, disc_y-disc_size, 
                     disc_x+disc_size, disc_y+disc_size], 
                    fill=(255, 220, 180), outline=(200, 150, 100))
        
        # 视盘中央凹陷
        inner_size = disc_size // 2
        draw.ellipse([disc_x-inner_size, disc_y-inner_size,
                     disc_x+inner_size, disc_y+inner_size],
                    fill=(255, 200, 150))
    
    def post_process_medical_image(self, img, disease_type):
        """医学图片后处理"""
        # 添加轻微模糊模拟真实拍摄
        img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
        
        # 添加噪声
        img_array = np.array(img)
        noise = np.random.normal(0, 1.5, img_array.shape).astype(np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # 根据疾病类型调整色调
        if disease_type == 'diabetic_retinopathy':
            # 增加红色调
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.1)
        elif disease_type == 'age_macular_degeneration':
            # 增加黄色调
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(0.9)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.05)
        
        return img
    
    def generate_medical_eye_image(self, width=800, height=600, disease_type='normal'):
        """生成医学眼部图片"""
        
        # 创建眼底背景
        img = self.create_fundus_background(width, height, disease_type)
        
        # 添加视盘
        self.add_optic_disc(img)
        
        # 绘制血管系统
        self.draw_blood_vessels(img, disease_type)
        
        # 添加疾病特异性病变
        if disease_type != 'normal':
            self.add_disease_lesions(img, disease_type)
        
        # 后处理
        img = self.post_process_medical_image(img, disease_type)
        
        return img
    
    def generate_medical_dataset(self):
        """生成完整的医学数据集"""
        print("🏥 医学眼部图片生成器")
        print("=" * 60)
        print("🎯 生成100张医学眼科疾病图片 (1:1:1:1)")
        print("📋 类别: 正常、糖尿病视网膜病变、高血压视网膜病变、老年黄斑变性")
        
        results = {}
        
        for disease_type, target_count in self.medical_classes.items():
            print(f"\n🏥 生成 {disease_type} 类别图片...")
            features = self.disease_features[disease_type]
            print(f"   📋 特征: {features['description']}")
            
            class_dir = os.path.join(self.medical_data_dir, disease_type)
            
            # 清理现有图片
            if os.path.exists(class_dir):
                existing_files = [f for f in os.listdir(class_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                for filename in existing_files:
                    filepath = os.path.join(class_dir, filename)
                    os.remove(filepath)
                    print(f"  🗑️  删除旧图片: {filename}")
            
            # 生成图片
            generated_count = 0
            for i in tqdm(range(target_count), desc=f"生成 {disease_type}"):
                try:
                    # 随机尺寸
                    width = random.randint(600, 1000)
                    height = random.randint(600, 1000)
                    
                    # 生成医学图片
                    img = self.generate_medical_eye_image(width, height, disease_type)
                    
                    # 保存图片
                    filename = f"{disease_type}_medical_{i+1:03d}.jpg"
                    filepath = os.path.join(class_dir, filename)
                    img.save(filepath, 'JPEG', quality=random.randint(88, 95))
                    
                    generated_count += 1
                    
                except Exception as e:
                    print(f"    ❌ 生成第 {i+1} 张图片时出错: {e}")
                    continue
            
            results[disease_type] = generated_count
            print(f"  📊 成功生成: {generated_count}/{target_count} 张医学图片")
        
        # 保存医学配置
        config_data = {
            'medical_classes': self.medical_classes,
            'disease_features': self.disease_features,
            'data_directory': self.medical_data_dir,
            'total_images': sum(results.values())
        }
        
        config_file = os.path.join(self.medical_data_dir, 'medical_dataset_info.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        total_generated = sum(results.values())
        
        print(f"\n🎉 医学数据集生成完成!")
        print("=" * 60)
        print(f"📊 结果:")
        for disease_type, count in results.items():
            target = self.medical_classes[disease_type]
            percentage = (count / target * 100) if target > 0 else 0
            print(f"  {disease_type:25s}: {count:2d}/{target} ({percentage:.1f}%)")
        
        print(f"📊 总计: {total_generated}/100 张医学眼科图片")
        print(f"💾 配置文件: {config_file}")
        
        return results

def main():
    """主函数"""
    print("🏥 医学眼部图片生成器")
    print("🎯 生成眼科疾病诊断图片")
    print("=" * 60)
    
    generator = MedicalEyeGenerator()
    results = generator.generate_medical_dataset()
    
    print(f"\n💡 下一步:")
    print(f"  1. 验证图片质量: python simple_validator.py")
    print(f"  2. 训练医学模型: python simple_ml_trainer.py")
    print(f"  3. 推送到GitHub: git add . && git commit && git push")
    
    return results

if __name__ == "__main__":
    results = main()
