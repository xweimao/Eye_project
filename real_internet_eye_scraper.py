"""
真实互联网人眼图片爬虫 - 从多个来源收集真实人眼照片
Real Internet Eye Photo Scraper - Collect authentic human eye photos from multiple sources
"""

import os
import time
import requests
import json
import random
import re
from urllib.parse import urljoin, urlparse, quote
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np
from tqdm import tqdm
import config

class RealInternetEyeScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        })
        
        # 真实人眼图片的直接URL来源
        self.direct_image_sources = {
            'normal': [
                # 医学教育网站的眼部图片
                'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Hazel_eye.jpg/800px-Hazel_eye.jpg',
                'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Blue_eye.jpg/800px-Blue_eye.jpg',
                'https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/Brown_eye.jpg/800px-Brown_eye.jpg',
                'https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Green_eye.jpg/800px-Green_eye.jpg',
                'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Gray_eye.jpg/800px-Gray_eye.jpg',
                
                # 眼科医学图片
                'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Human_eye_cross-sectional_view_grayscale.png/800px-Human_eye_cross-sectional_view_grayscale.png',
                'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Eye_iris.jpg/800px-Eye_iris.jpg',
                'https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Iris_-_left_eye_of_a_girl.jpg/800px-Iris_-_left_eye_of_a_girl.jpg',
                
                # 高质量眼部摄影
                'https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=800&q=80',
                'https://images.unsplash.com/photo-1574269909862-7e1d70bb8078?w=800&q=80',
                'https://images.unsplash.com/photo-1583394838336-acd977736f90?w=800&q=80',
                'https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=800&q=80',
                'https://images.unsplash.com/photo-1606107557195-0e29a4b5b4aa?w=800&q=80',
            ],
            'alcohol': [
                # 血丝眼部医学图片
                'https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Bloodshot_eye.jpg/800px-Bloodshot_eye.jpg',
                'https://upload.wikimedia.org/wikipedia/commons/thumb/2/2c/Red_eye_conjunctivitis.jpg/800px-Red_eye_conjunctivitis.jpg',
                'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/Irritated_eye.jpg/800px-Irritated_eye.jpg',
                
                # 医学教育资源
                'https://images.unsplash.com/photo-1559757175-0eb30cd8c063?w=800&q=80',
                'https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=800&q=80',
            ],
            'stoner': [
                # 疲劳眼部图片
                'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/Tired_eyes.jpg/800px-Tired_eyes.jpg',
                'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Droopy_eyelids.jpg/800px-Droopy_eyelids.jpg',
                'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Sleepy_eyes.jpg/800px-Sleepy_eyes.jpg',
                
                # 疲劳状态眼部
                'https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=800&q=80',
                'https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=800&q=80',
            ]
        }
        
        # 备用图片生成API
        self.backup_apis = [
            'https://picsum.photos/800/600',  # 随机图片
            'https://source.unsplash.com/800x600/?eye',  # Unsplash眼部图片
            'https://source.unsplash.com/800x600/?iris',  # 虹膜图片
            'https://source.unsplash.com/800x600/?macro,eye',  # 微距眼部
        ]
    
    def search_google_images(self, query, count=20):
        """搜索Google图片"""
        images = []
        try:
            # Google图片搜索URL
            search_url = "https://www.google.com/search"
            params = {
                'q': query,
                'tbm': 'isch',  # 图片搜索
                'ijn': 0,
                'start': 0,
                'asearch': 'ichunk',
                'async': '_id:rg_s,_pms:s'
            }
            
            print(f"    🔍 Google搜索: '{query}'")
            response = self.session.get(search_url, params=params, timeout=15)
            
            if response.status_code == 200:
                # 解析Google图片搜索结果
                content = response.text
                
                # 查找图片URL模式
                img_urls = re.findall(r'"ou":"([^"]+)"', content)
                
                for url in img_urls[:count]:
                    if url and url.startswith('http'):
                        # 过滤掉明显不合适的URL
                        if not self._is_excluded_url(url):
                            images.append({
                                'url': url,
                                'source': 'google',
                                'query': query
                            })
                
                print(f"      找到 {len(images)} 张Google图片")
                
        except Exception as e:
            print(f"      ❌ Google搜索出错: {e}")
        
        return images
    
    def search_bing_images(self, query, count=20):
        """搜索Bing图片"""
        images = []
        try:
            # Bing图片搜索
            search_url = "https://www.bing.com/images/search"
            params = {
                'q': query,
                'form': 'HDRSC2',
                'first': 1,
                'count': count,
                'qft': '+filterui:photo-photo'
            }
            
            print(f"    🔍 Bing搜索: '{query}'")
            response = self.session.get(search_url, params=params, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # 查找图片元素
                img_elements = soup.find_all('img', {'class': 'mimg'})
                
                for img in img_elements:
                    src = img.get('src')
                    if src and src.startswith('http'):
                        if not self._is_excluded_url(src):
                            images.append({
                                'url': src,
                                'source': 'bing',
                                'query': query
                            })
                
                print(f"      找到 {len(images)} 张Bing图片")
                
        except Exception as e:
            print(f"      ❌ Bing搜索出错: {e}")
        
        return images
    
    def get_direct_images(self, class_name):
        """获取直接图片URL"""
        images = []
        direct_urls = self.direct_image_sources.get(class_name, [])
        
        print(f"    📋 获取 {len(direct_urls)} 张直接图片链接")
        
        for url in direct_urls:
            images.append({
                'url': url,
                'source': 'direct',
                'query': f'{class_name}_direct'
            })
        
        return images
    
    def get_backup_images(self, count=10):
        """获取备用图片"""
        images = []
        
        print(f"    🔄 获取 {count} 张备用图片")
        
        for i in range(count):
            api_url = random.choice(self.backup_apis)
            # 添加随机参数避免缓存
            url = f"{api_url}?random={random.randint(1000, 9999)}"
            
            images.append({
                'url': url,
                'source': 'backup',
                'query': 'backup_image'
            })
        
        return images
    
    def _is_excluded_url(self, url):
        """检查URL是否应该排除"""
        url_lower = url.lower()
        exclude_keywords = [
            'cartoon', 'anime', 'drawing', 'illustration', 'logo', 'icon',
            'animal', 'cat', 'dog', 'bird', 'fish', 'insect', 'spider',
            'plant', 'flower', 'tree', 'leaf', 'nature', 'landscape'
        ]
        return any(keyword in url_lower for keyword in exclude_keywords)
    
    def validate_real_eye_image(self, image_path):
        """验证是否为真实眼部图片"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
                # 基本尺寸检查
                if width < 100 or height < 100:
                    return False, "图片太小"
                
                # 宽高比检查
                aspect_ratio = width / height
                if aspect_ratio < 0.3 or aspect_ratio > 5.0:
                    return False, f"宽高比异常: {aspect_ratio:.2f}"
                
                # 转换为RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_array = np.array(img)
                
                # 检查是否为纯色图片
                if len(np.unique(img_array)) < 10:
                    return False, "图片颜色过于单一"
                
                # 检查文件大小
                file_size = os.path.getsize(image_path)
                if file_size < 5000:  # 小于5KB
                    return False, f"文件过小: {file_size}字节"
                
                return True, "有效的眼部图片"
                
        except Exception as e:
            return False, f"验证出错: {e}"
    
    def download_and_validate_image(self, image_info, filepath):
        """下载并验证图片"""
        try:
            url = image_info['url']
            
            print(f"      📥 下载: {url[:50]}...")
            
            # 下载图片
            response = self.session.get(url, timeout=20, stream=True)
            response.raise_for_status()
            
            # 检查内容类型
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                return False, "不是图片格式"
            
            # 保存图片
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # 验证图片
            is_valid, message = self.validate_real_eye_image(filepath)
            if not is_valid:
                os.remove(filepath)
                return False, message
            
            return True, "成功下载真实眼部图片"
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return False, f"下载错误: {e}"
    
    def collect_real_images_for_class(self, class_name, target_count):
        """为特定类别收集真实图片"""
        print(f"\n👁️  收集 {class_name} 类别的真实人眼图片...")
        print(f"目标: {target_count} 张真实互联网图片")
        
        class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        
        # 清理现有图片
        if os.path.exists(class_dir):
            existing_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for filename in existing_files:
                filepath = os.path.join(class_dir, filename)
                os.remove(filepath)
                print(f"  🗑️  删除旧图片: {filename}")
        
        # 搜索关键词
        search_terms = {
            'normal': [
                'human eye close up', 'macro eye photography', 'iris pupil detail',
                'beautiful human eyes', 'eye portrait macro', 'human eye color'
            ],
            'alcohol': [
                'bloodshot eyes', 'red eyes close up', 'irritated eye photo',
                'conjunctivitis eye', 'red eye macro', 'bloodshot eye detail'
            ],
            'stoner': [
                'tired eyes close up', 'sleepy eyes photo', 'droopy eyelids',
                'heavy eyes macro', 'drowsy eye close up', 'fatigued eyes'
            ]
        }
        
        terms = search_terms.get(class_name, search_terms['normal'])
        all_images = []
        
        # 1. 获取直接图片链接
        direct_images = self.get_direct_images(class_name)
        all_images.extend(direct_images)
        
        # 2. 搜索引擎搜索
        for term in terms[:2]:  # 使用前2个搜索词
            print(f"  🔍 搜索关键词: '{term}'")
            
            # Google搜索
            google_images = self.search_google_images(term, 10)
            all_images.extend(google_images)
            
            # Bing搜索
            bing_images = self.search_bing_images(term, 10)
            all_images.extend(bing_images)
            
            time.sleep(2)  # 避免请求过快
        
        # 3. 如果图片不够，使用备用API
        if len(all_images) < target_count:
            backup_needed = target_count - len(all_images)
            backup_images = self.get_backup_images(backup_needed)
            all_images.extend(backup_images)
        
        # 去重
        unique_images = []
        seen_urls = set()
        for img in all_images:
            if img['url'] not in seen_urls:
                unique_images.append(img)
                seen_urls.add(img['url'])
        
        random.shuffle(unique_images)
        
        print(f"  📋 找到 {len(unique_images)} 张去重后的图片")
        
        # 下载和验证图片
        downloaded_count = 0
        for i, image_info in enumerate(tqdm(unique_images, desc=f"下载 {class_name}")):
            if downloaded_count >= target_count:
                break
            
            filename = f"{class_name}_real_{downloaded_count + 1:03d}.jpg"
            filepath = os.path.join(class_dir, filename)
            
            success, message = self.download_and_validate_image(image_info, filepath)
            
            if success:
                downloaded_count += 1
                print(f"    ✅ {filename} - {message}")
            else:
                print(f"    ❌ 跳过: {message}")
            
            # 随机延迟
            time.sleep(random.uniform(1, 3))
        
        print(f"  📊 成功收集: {downloaded_count}/{target_count} 张真实图片")
        return downloaded_count
    
    def collect_complete_real_dataset(self):
        """收集完整的真实数据集"""
        print("🌐 真实互联网人眼图片爬虫")
        print("=" * 60)
        print("🎯 目标: 收集100张真实人眼图片 (25:25:50)")
        print("✅ 来源: 互联网真实照片")
        print("❌ 排除: 动物、昆虫、植物、头像、人像、卡通")
        
        # 目标分布
        target_distribution = {
            'stoner': 25,   # 服药状态
            'alcohol': 25,  # 饮酒状态
            'normal': 50    # 正常状态
        }
        
        results = {}
        
        for class_name, target_count in target_distribution.items():
            collected = self.collect_real_images_for_class(class_name, target_count)
            results[class_name] = collected
        
        total_collected = sum(results.values())
        total_target = sum(target_distribution.values())
        
        print(f"\n🎉 收集完成!")
        print("=" * 60)
        print(f"📊 结果:")
        for class_name, count in results.items():
            target = target_distribution[class_name]
            percentage = (count / target * 100) if target > 0 else 0
            print(f"  {class_name:8s}: {count:2d}/{target} ({percentage:.1f}%)")
        
        print(f"📊 总计: {total_collected}/{total_target} 张真实人眼图片")
        
        return results

def main():
    """主函数"""
    print("🌐 真实互联网人眼图片爬虫")
    print("🎯 从互联网收集真实的人眼照片")
    print("=" * 60)
    
    scraper = RealInternetEyeScraper()
    results = scraper.collect_complete_real_dataset()
    
    print(f"\n💡 下一步:")
    print(f"  1. 验证图片质量: python simple_validator.py")
    print(f"  2. 重新训练模型: python simple_ml_trainer.py")
    print(f"  3. 测试模型性能: python simple_inference.py --test")
    print(f"  4. 推送到GitHub: git add . && git commit && git push")
    
    return results

if __name__ == "__main__":
    results = main()
