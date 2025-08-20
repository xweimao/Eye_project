"""
专业人眼图片生成器 - 创建高质量的合成人眼图片
Professional Eye Image Generator - Creates high-quality synthetic human eye images
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import random
import colorsys
from tqdm import tqdm
import config

class ProfessionalEyeGenerator:
    def __init__(self):
        # 真实的虹膜颜色（基于人类眼睛的实际颜色分布）
        self.iris_colors = {
            'brown': [(101, 67, 33), (139, 90, 43), (160, 82, 45), (139, 69, 19)],
            'blue': [(70, 130, 180), (100, 149, 237), (135, 206, 235), (176, 196, 222)],
            'green': [(34, 139, 34), (85, 107, 47), (107, 142, 35), (154, 205, 50)],
            'hazel': [(139, 115, 85), (160, 134, 90), (205, 133, 63), (218, 165, 32)],
            'gray': [(105, 105, 105), (128, 128, 128), (169, 169, 169), (192, 192, 192)]
        }
        
        # 肤色范围（不同种族的真实肤色）
        self.skin_tones = [
            (255, 219, 172), (241, 194, 125), (224, 172, 105), (198, 134, 66),
            (141, 85, 36), (255, 239, 213), (255, 222, 173), (255, 205, 148),
            (241, 194, 125), (224, 172, 105), (198, 134, 66), (141, 85, 36)
        ]
    
    def create_realistic_iris(self, size, color, eye_state='normal'):
        """创建真实的虹膜纹理"""
        iris = Image.new('RGB', (size, size), color)
        draw = ImageDraw.Draw(iris)
        
        center = size // 2
        
        # 添加虹膜的放射状纹理
        for i in range(50):
            angle = random.uniform(0, 360)
            length = random.uniform(size * 0.2, size * 0.4)
            start_r = random.uniform(size * 0.1, size * 0.2)
            
            start_x = center + start_r * np.cos(np.radians(angle))
            start_y = center + start_r * np.sin(np.radians(angle))
            end_x = center + (start_r + length) * np.cos(np.radians(angle))
            end_y = center + (start_r + length) * np.sin(np.radians(angle))
            
            # 虹膜纹理颜色变化
            texture_color = tuple(max(0, min(255, c + random.randint(-30, 30))) for c in color)
            draw.line([start_x, start_y, end_x, end_y], fill=texture_color, width=1)
        
        # 添加同心圆纹理
        for r in range(size//8, size//2, size//16):
            circle_color = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in color)
            draw.ellipse([center-r, center-r, center+r, center+r], 
                        outline=circle_color, width=1)
        
        # 根据眼部状态调整颜色
        if eye_state == 'alcohol':
            # 血丝效果 - 增加红色
            enhancer = ImageEnhance.Color(iris)
            iris = enhancer.enhance(1.2)
            # 添加红色调
            iris_array = np.array(iris)
            iris_array[:, :, 0] = np.clip(iris_array[:, :, 0] + 20, 0, 255)
            iris = Image.fromarray(iris_array)
        
        elif eye_state == 'stoner':
            # 疲倦效果 - 降低亮度和饱和度
            enhancer = ImageEnhance.Brightness(iris)
            iris = enhancer.enhance(0.8)
            enhancer = ImageEnhance.Color(iris)
            iris = enhancer.enhance(0.7)
        
        return iris
    
    def create_realistic_sclera(self, width, height, eye_state='normal'):
        """创建真实的眼白（巩膜）"""
        if eye_state == 'normal':
            base_color = (255, 248, 240)
        elif eye_state == 'alcohol':
            base_color = (255, 230, 230)  # 带血丝的粉红色
        else:  # stoner
            base_color = (255, 240, 235)  # 略带黄色
        
        sclera = Image.new('RGB', (width, height), base_color)
        
        # 添加血管纹理
        if eye_state in ['alcohol', 'stoner']:
            draw = ImageDraw.Draw(sclera)
            vessel_count = 8 if eye_state == 'alcohol' else 4
            
            for _ in range(vessel_count):
                # 随机血管路径
                start_x = random.randint(0, width)
                start_y = random.randint(0, height)
                
                points = [(start_x, start_y)]
                for _ in range(random.randint(3, 6)):
                    last_x, last_y = points[-1]
                    next_x = last_x + random.randint(-30, 30)
                    next_y = last_y + random.randint(-20, 20)
                    next_x = max(0, min(width, next_x))
                    next_y = max(0, min(height, next_y))
                    points.append((next_x, next_y))
                
                # 绘制血管
                vessel_color = (255, 150, 150) if eye_state == 'alcohol' else (255, 200, 180)
                for i in range(len(points) - 1):
                    draw.line([points[i], points[i+1]], fill=vessel_color, width=1)
        
        return sclera
    
    def create_realistic_eyelid(self, width, height, eye_state='normal'):
        """创建真实的眼睑"""
        skin_color = random.choice(self.skin_tones)
        
        # 根据状态调整眼睑
        if eye_state == 'stoner':
            # 下垂的眼睑 - 更多的皮肤覆盖
            lid_coverage = 0.3
        else:
            lid_coverage = 0.15
        
        # 创建渐变的眼睑
        eyelid = Image.new('RGB', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(eyelid)
        
        # 上眼睑
        lid_height = int(height * lid_coverage)
        for y in range(lid_height):
            alpha = int(255 * (lid_height - y) / lid_height)
            lid_color = tuple(int(c * alpha / 255) for c in skin_color)
            draw.line([(0, y), (width, y)], fill=lid_color)
        
        # 下眼睑
        bottom_lid_height = int(height * 0.1)
        for y in range(height - bottom_lid_height, height):
            alpha = int(255 * (y - (height - bottom_lid_height)) / bottom_lid_height)
            lid_color = tuple(int(c * alpha / 255) for c in skin_color)
            draw.line([(0, y), (width, y)], fill=lid_color)
        
        return eyelid
    
    def generate_professional_eye(self, width=800, height=600, eye_state='normal'):
        """生成专业质量的人眼图片"""
        
        # 创建基础画布
        img = Image.new('RGB', (width, height), (240, 230, 220))
        
        # 眼部区域参数
        eye_center_x, eye_center_y = width // 2, height // 2
        eye_width = int(width * 0.6)
        eye_height = int(height * 0.4)
        
        # 创建眼白
        sclera = self.create_realistic_sclera(eye_width, eye_height, eye_state)
        
        # 选择虹膜颜色
        iris_color_family = random.choice(list(self.iris_colors.keys()))
        iris_color = random.choice(self.iris_colors[iris_color_family])
        
        # 创建虹膜
        iris_size = int(eye_height * 0.8)
        iris = self.create_realistic_iris(iris_size, iris_color, eye_state)
        
        # 创建瞳孔
        pupil_size = int(iris_size * 0.35)
        if eye_state == 'stoner':
            pupil_size = int(iris_size * 0.45)  # 放大的瞳孔
        
        pupil = Image.new('RGB', (pupil_size, pupil_size), (0, 0, 0))
        
        # 组合眼部
        eye_left = eye_center_x - eye_width // 2
        eye_top = eye_center_y - eye_height // 2
        
        # 粘贴眼白
        img.paste(sclera, (eye_left, eye_top))
        
        # 粘贴虹膜
        iris_x = eye_center_x - iris_size // 2
        iris_y = eye_center_y - iris_size // 2
        
        # 创建椭圆形遮罩
        mask = Image.new('L', (iris_size, iris_size), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse([0, 0, iris_size, iris_size], fill=255)
        
        img.paste(iris, (iris_x, iris_y), mask)
        
        # 粘贴瞳孔
        pupil_x = eye_center_x - pupil_size // 2
        pupil_y = eye_center_y - pupil_size // 2
        
        pupil_mask = Image.new('L', (pupil_size, pupil_size), 0)
        pupil_mask_draw = ImageDraw.Draw(pupil_mask)
        pupil_mask_draw.ellipse([0, 0, pupil_size, pupil_size], fill=255)
        
        img.paste(pupil, (pupil_x, pupil_y), pupil_mask)
        
        # 添加高光
        highlight_size = pupil_size // 4
        highlight_x = pupil_x + pupil_size // 3
        highlight_y = pupil_y + pupil_size // 4
        
        draw = ImageDraw.Draw(img)
        draw.ellipse([highlight_x, highlight_y, 
                     highlight_x + highlight_size, highlight_y + highlight_size],
                    fill=(255, 255, 255))
        
        # 添加眼睑
        eyelid = self.create_realistic_eyelid(width, height, eye_state)
        img = Image.alpha_composite(img.convert('RGBA'), eyelid.convert('RGBA')).convert('RGB')
        
        # 添加眼睫毛
        self.add_eyelashes(img, eye_center_x, eye_center_y, eye_width, eye_height)
        
        # 后处理
        img = self.post_process_image(img, eye_state)
        
        return img
    
    def add_eyelashes(self, img, center_x, center_y, eye_width, eye_height):
        """添加眼睫毛"""
        draw = ImageDraw.Draw(img)
        
        # 上眼睫毛
        lash_count = random.randint(15, 25)
        for i in range(lash_count):
            angle = random.uniform(-30, 30)
            start_x = center_x - eye_width//2 + (i * eye_width // lash_count)
            start_y = center_y - eye_height//2
            
            length = random.randint(10, 25)
            end_x = start_x + length * np.sin(np.radians(angle))
            end_y = start_y - length * np.cos(np.radians(angle))
            
            draw.line([start_x, start_y, end_x, end_y], 
                     fill=(50, 30, 20), width=random.randint(1, 2))
        
        # 下眼睫毛
        lash_count = random.randint(8, 15)
        for i in range(lash_count):
            angle = random.uniform(-15, 15)
            start_x = center_x - eye_width//2 + (i * eye_width // lash_count)
            start_y = center_y + eye_height//2
            
            length = random.randint(5, 12)
            end_x = start_x + length * np.sin(np.radians(angle))
            end_y = start_y + length * np.cos(np.radians(angle))
            
            draw.line([start_x, start_y, end_x, end_y], 
                     fill=(50, 30, 20), width=1)
    
    def post_process_image(self, img, eye_state):
        """图片后处理"""
        # 添加轻微的高斯模糊以增加真实感
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # 添加噪声
        img_array = np.array(img)
        noise = np.random.normal(0, 2, img_array.shape).astype(np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # 根据状态调整整体色调
        if eye_state == 'alcohol':
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.1)
        elif eye_state == 'stoner':
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(0.9)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(0.8)
        
        return img
    
    def generate_dataset(self, target_distribution=None):
        """生成完整的数据集"""
        if target_distribution is None:
            target_distribution = {'normal': 50, 'alcohol': 25, 'stoner': 25}
        
        print("🎨 专业人眼图片生成器")
        print("=" * 60)
        print("🎯 生成高质量合成人眼图片")
        print("✅ 特征: 真实虹膜纹理、血管、眼睫毛、肤色")
        print("❌ 排除: 动物、卡通、插图等不相关内容")
        
        results = {}
        
        for class_name, target_count in target_distribution.items():
            print(f"\n👁️  生成 {class_name} 类别图片...")
            print(f"目标: {target_count} 张专业质量人眼图片")
            
            class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
            
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
            for i in tqdm(range(target_count), desc=f"生成 {class_name}"):
                try:
                    # 随机尺寸
                    width = random.randint(600, 1200)
                    height = random.randint(450, 900)
                    
                    # 生成图片
                    img = self.generate_professional_eye(width, height, class_name)
                    
                    # 保存图片
                    filename = f"{class_name}_pro_{i+1:03d}.jpg"
                    filepath = os.path.join(class_dir, filename)
                    img.save(filepath, 'JPEG', quality=random.randint(85, 95))
                    
                    generated_count += 1
                    
                except Exception as e:
                    print(f"    ❌ 生成第 {i+1} 张图片时出错: {e}")
                    continue
            
            results[class_name] = generated_count
            print(f"  📊 成功生成: {generated_count}/{target_count} 张专业图片")
        
        total_generated = sum(results.values())
        total_target = sum(target_distribution.values())
        
        print(f"\n🎉 生成完成!")
        print("=" * 60)
        print(f"📊 结果:")
        for class_name, count in results.items():
            target = target_distribution[class_name]
            percentage = (count / target * 100) if target > 0 else 0
            print(f"  {class_name:8s}: {count:2d}/{target} ({percentage:.1f}%)")
        
        print(f"📊 总计: {total_generated}/{total_target} 张专业人眼图片")
        
        return results

def main():
    """主函数"""
    print("🎨 专业人眼图片生成器")
    print("🎯 生成高质量的合成人眼图片")
    print("=" * 60)
    
    generator = ProfessionalEyeGenerator()
    results = generator.generate_dataset()
    
    print(f"\n💡 下一步:")
    print(f"  1. 验证图片质量: python simple_validator.py")
    print(f"  2. 重新训练模型: python simple_ml_trainer.py")
    print(f"  3. 测试模型性能: python simple_inference.py --test")
    print(f"  4. 推送到GitHub: git add . && git commit && git push")
    
    return results

if __name__ == "__main__":
    results = main()
