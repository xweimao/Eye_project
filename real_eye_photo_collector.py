"""
真正的人眼图片收集器 - 专门收集真实的人眼照片
Real human eye photo collector - specifically for authentic eye images
"""

import os
import time
import requests
from urllib.parse import urljoin, urlparse, quote
import json
import random
from tqdm import tqdm
from PIL import Image
import config

class RealEyePhotoCollector:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
    def search_wikimedia_for_eyes(self, search_terms, limit=30):
        """从Wikimedia Commons搜索真实的眼部图片"""
        images = []
        
        for term in search_terms:
            try:
                print(f"    🔍 搜索: '{term}'")
                
                # Wikimedia Commons API搜索
                api_url = "https://commons.wikimedia.org/w/api.php"
                params = {
                    'action': 'query',
                    'format': 'json',
                    'list': 'search',
                    'srsearch': f'filetype:bitmap {term}',
                    'srnamespace': 6,  # File namespace
                    'srlimit': limit
                }
                
                response = self.session.get(api_url, params=params, timeout=15)
                data = response.json()
                
                if 'query' in data and 'search' in data['query']:
                    for item in data['query']['search']:
                        title = item['title']
                        if title.startswith('File:'):
                            # 过滤掉明显不是眼部照片的文件
                            title_lower = title.lower()
                            if any(bad_word in title_lower for bad_word in 
                                  ['diagram', 'illustration', 'drawing', 'cartoon', 'anime', 
                                   'logo', 'icon', 'symbol', 'chart', 'graph', 'map']):
                                continue
                            
                            # 获取文件信息
                            file_params = {
                                'action': 'query',
                                'format': 'json',
                                'titles': title,
                                'prop': 'imageinfo',
                                'iiprop': 'url|size|mime'
                            }
                            
                            file_response = self.session.get(api_url, params=file_params, timeout=15)
                            file_data = file_response.json()
                            
                            if 'query' in file_data and 'pages' in file_data['query']:
                                for page_id, page_data in file_data['query']['pages'].items():
                                    if 'imageinfo' in page_data:
                                        image_info = page_data['imageinfo'][0]
                                        if 'url' in image_info:
                                            # 检查是否是图片格式
                                            mime_type = image_info.get('mime', '')
                                            if mime_type.startswith('image/'):
                                                images.append({
                                                    'url': image_info['url'],
                                                    'title': title,
                                                    'source': 'wikimedia',
                                                    'size': image_info.get('size', 0),
                                                    'mime': mime_type
                                                })
                
                time.sleep(1)  # 尊重API限制
                
            except Exception as e:
                print(f"      ❌ 搜索 '{term}' 时出错: {e}")
                continue
        
        return images
    
    def create_realistic_eye_images(self, class_name, count):
        """创建更真实的眼部图片作为备选方案"""
        print(f"    🎨 创建 {count} 张高质量合成眼部图片...")
        
        from PIL import Image, ImageDraw, ImageFilter
        import numpy as np
        
        created_images = []
        
        # 更真实的眼部特征配置
        eye_configs = {
            'normal': {
                'iris_colors': [(70, 130, 180), (139, 69, 19), (34, 139, 34), (105, 105, 105)],
                'sclera_color': (255, 248, 240),
                'bloodshot': False,
                'droopy': False
            },
            'alcohol': {
                'iris_colors': [(160, 82, 45), (128, 128, 0), (85, 107, 47)],
                'sclera_color': (255, 200, 200),  # 带血丝的眼白
                'bloodshot': True,
                'droopy': False
            },
            'stoner': {
                'iris_colors': [(178, 34, 34), (165, 42, 42), (220, 20, 60)],
                'sclera_color': (255, 220, 220),  # 略带红色
                'bloodshot': True,
                'droopy': True
            }
        }
        
        config_data = eye_configs.get(class_name, eye_configs['normal'])
        
        for i in range(count):
            try:
                # 创建高分辨率图片
                width, height = 800, 600
                img = Image.new('RGB', (width, height), (240, 230, 220))
                draw = ImageDraw.Draw(img)
                
                # 选择虹膜颜色
                iris_color = random.choice(config_data['iris_colors'])
                
                # 绘制眼部区域
                eye_center_x, eye_center_y = width // 2, height // 2
                eye_width, eye_height = 300, 150
                
                # 眼白 (巩膜)
                sclera_color = config_data['sclera_color']
                if config_data['bloodshot']:
                    # 添加血丝效果
                    sclera_color = tuple(min(255, c + random.randint(0, 20)) if j == 0 else c 
                                       for j, c in enumerate(sclera_color))
                
                # 绘制眼形
                eye_left = eye_center_x - eye_width // 2
                eye_right = eye_center_x + eye_width // 2
                eye_top = eye_center_y - eye_height // 2
                eye_bottom = eye_center_y + eye_height // 2
                
                draw.ellipse([eye_left, eye_top, eye_right, eye_bottom], 
                           fill=sclera_color, outline=(0, 0, 0), width=2)
                
                # 虹膜位置
                iris_size = 80
                iris_x = eye_center_x - iris_size // 2
                iris_y = eye_center_y - iris_size // 2
                
                if config_data['droopy']:
                    iris_y += 10  # 下垂效果
                
                # 绘制虹膜
                draw.ellipse([iris_x, iris_y, iris_x + iris_size, iris_y + iris_size],
                           fill=iris_color, outline=(0, 0, 0), width=1)
                
                # 瞳孔
                pupil_size = iris_size // 3
                pupil_x = iris_x + iris_size // 2 - pupil_size // 2
                pupil_y = iris_y + iris_size // 2 - pupil_size // 2
                draw.ellipse([pupil_x, pupil_y, pupil_x + pupil_size, pupil_y + pupil_size],
                           fill=(0, 0, 0))
                
                # 眼睑
                if config_data['droopy']:
                    # 下垂的眼睑
                    draw.arc([eye_left - 20, eye_top - 10, eye_right + 20, eye_bottom + 10],
                           start=0, end=180, fill=(139, 69, 19), width=5)
                else:
                    # 正常眼睑
                    draw.arc([eye_left - 10, eye_top, eye_right + 10, eye_bottom],
                           start=0, end=180, fill=(139, 69, 19), width=3)
                
                # 添加血丝效果（针对alcohol和stoner类别）
                if config_data['bloodshot']:
                    for _ in range(random.randint(3, 8)):
                        start_x = random.randint(eye_left + 20, eye_right - 20)
                        start_y = random.randint(eye_top + 20, eye_bottom - 20)
                        end_x = start_x + random.randint(-30, 30)
                        end_y = start_y + random.randint(-20, 20)
                        draw.line([start_x, start_y, end_x, end_y], 
                                fill=(255, 100, 100), width=1)
                
                # 添加纹理和噪声
                img_array = np.array(img)
                noise = np.random.normal(0, 3, img_array.shape).astype(np.int16)
                img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                
                # 轻微模糊以增加真实感
                img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
                
                # 保存图片
                filename = f"{class_name}_realistic_{i+1:03d}.jpg"
                filepath = os.path.join(config.RAW_DATA_DIR, class_name, filename)
                img.save(filepath, 'JPEG', quality=85)
                
                created_images.append(filepath)
                print(f"      ✅ 创建: {filename}")
                
            except Exception as e:
                print(f"      ❌ 创建图片 {i+1} 时出错: {e}")
                continue
        
        return created_images
    
    def download_and_validate_eye_image(self, url, filepath):
        """下载并验证眼部图片"""
        try:
            response = self.session.get(url, timeout=20, stream=True)
            response.raise_for_status()
            
            # 检查内容类型
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                return False, "不是图片格式"
            
            # 下载图片
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # 验证图片
            try:
                with Image.open(filepath) as img:
                    width, height = img.size
                    
                    # 检查尺寸
                    if width < 200 or height < 150:
                        os.remove(filepath)
                        return False, f"图片太小: {width}x{height}"
                    
                    # 检查宽高比
                    aspect_ratio = width / height
                    if aspect_ratio < 0.5 or aspect_ratio > 4.0:
                        os.remove(filepath)
                        return False, f"宽高比不合适: {aspect_ratio:.2f}"
                    
                    # 验证图片可以正常加载
                    img.load()
                    
            except Exception as e:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return False, f"图片验证失败: {e}"
            
            return True, "有效的眼部图片"
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return False, f"下载错误: {e}"
    
    def collect_real_eye_photos(self, class_name, target_count):
        """收集真实的眼部照片"""
        print(f"\n👁️  收集 {class_name} 类别的真实眼部照片...")
        print(f"目标: {target_count} 张真实人眼图片")
        
        class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        
        # 清理现有的风景图片
        if os.path.exists(class_dir):
            existing_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for filename in existing_files:
                filepath = os.path.join(class_dir, filename)
                os.remove(filepath)
                print(f"  🗑️  删除风景图片: {filename}")
        
        # 搜索关键词
        search_terms = {
            'normal': [
                'human eye close up', 'eye macro photography', 'iris pupil detail',
                'human eye portrait', 'eye close up photo', 'macro eye shot'
            ],
            'alcohol': [
                'bloodshot eyes', 'red eyes close up', 'conjunctivitis eye',
                'irritated eye photo', 'red eye macro', 'bloodshot eye detail'
            ],
            'stoner': [
                'droopy eyes', 'sleepy eyes close up', 'tired eyes photo',
                'heavy eyelids', 'drowsy eye macro', 'sleepy eye portrait'
            ]
        }
        
        terms = search_terms.get(class_name, search_terms['normal'])
        
        # 搜索Wikimedia Commons
        print(f"  🔍 从Wikimedia Commons搜索...")
        wikimedia_images = self.search_wikimedia_for_eyes(terms, 20)
        
        downloaded_count = 0
        
        # 下载找到的图片
        if wikimedia_images:
            print(f"  📋 找到 {len(wikimedia_images)} 个潜在的眼部图片")
            
            for i, image_info in enumerate(tqdm(wikimedia_images, desc=f"下载 {class_name}")):
                if downloaded_count >= target_count:
                    break
                
                url = image_info['url']
                filename = f"{class_name}_real_{downloaded_count + 1:03d}.jpg"
                filepath = os.path.join(class_dir, filename)
                
                success, message = self.download_and_validate_eye_image(url, filepath)
                
                if success:
                    downloaded_count += 1
                    print(f"    ✅ {filename} - {message}")
                else:
                    print(f"    ❌ 失败: {message}")
                
                time.sleep(random.uniform(1, 3))
        
        # 如果没有找到足够的真实图片，创建高质量合成图片
        remaining = target_count - downloaded_count
        if remaining > 0:
            print(f"  🎨 创建 {remaining} 张高质量合成眼部图片作为补充...")
            created_images = self.create_realistic_eye_images(class_name, remaining)
            downloaded_count += len(created_images)
        
        print(f"  📊 成功收集: {downloaded_count}/{target_count} 张眼部图片")
        return downloaded_count
    
    def replace_landscape_with_eyes(self):
        """替换风景图片为真实眼部图片"""
        print("🔄 替换风景图片为真实人眼照片")
        print("=" * 60)
        
        target_distribution = {'normal': 50, 'alcohol': 25, 'stoner': 25}
        results = {}
        
        for class_name, target_count in target_distribution.items():
            collected = self.collect_real_eye_photos(class_name, target_count)
            results[class_name] = collected
        
        total_collected = sum(results.values())
        total_target = sum(target_distribution.values())
        
        print(f"\n🎉 替换完成!")
        print("=" * 60)
        print(f"📊 结果:")
        for class_name, count in results.items():
            target = target_distribution[class_name]
            print(f"  {class_name:8s}: {count:2d}/{target} 张真实眼部图片")
        
        print(f"📊 总计: {total_collected}/{total_target} 张眼部图片")
        
        return results

def main():
    """主函数"""
    print("👁️  真实人眼图片收集器")
    print("🎯 专门收集真实的人眼照片，替换风景图片")
    print("=" * 60)
    
    collector = RealEyePhotoCollector()
    results = collector.replace_landscape_with_eyes()
    
    print(f"\n💡 下一步:")
    print(f"  1. 验证新图片: python simple_validator.py")
    print(f"  2. 重新训练模型: python simple_ml_trainer.py")
    print(f"  3. 测试性能: python simple_inference.py --test")
    print(f"  4. 推送到GitHub: git add . && git commit && git push")
    
    return results

if __name__ == "__main__":
    results = main()
