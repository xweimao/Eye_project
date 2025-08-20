"""
高级人眼图片爬虫 - 专门收集真实人眼照片
Advanced Human Eye Scraper - Specialized for real human eye photos
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

class AdvancedEyeScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,zh-CN,zh;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        })
        
        # 专业搜索关键词
        self.search_keywords = {
            'normal': [
                'human eye close up macro photography',
                'beautiful human eyes macro shot',
                'human iris pupil close up photo',
                'eye macro photography portrait',
                'human eye detail close up',
                'macro human eye photography',
                'close up human eye shot',
                'human eye iris macro',
                'detailed human eye photo',
                'human eye close up portrait'
            ],
            'alcohol': [
                'bloodshot eyes close up photo',
                'red eyes alcohol effect',
                'bloodshot human eyes macro',
                'red irritated eyes close up',
                'alcohol bloodshot eyes photo',
                'red eyes hangover close up',
                'bloodshot eye macro photography',
                'red tired eyes close up',
                'alcohol red eyes photo',
                'bloodshot eye detail shot'
            ],
            'stoner': [
                'droopy sleepy eyes close up',
                'tired heavy eyelids photo',
                'sleepy droopy eyes macro',
                'cannabis droopy eyes photo',
                'heavy eyelids close up shot',
                'sleepy tired eyes macro',
                'droopy eyelids photography',
                'tired droopy eyes close up',
                'sleepy eyes macro photo',
                'heavy droopy eyelids shot'
            ]
        }
        
        # 排除关键词
        self.exclude_keywords = [
            'animal', 'cat', 'dog', 'bird', 'fish', 'insect', 'spider', 'snake',
            'cartoon', 'anime', 'drawing', 'illustration', 'art', 'painting',
            'diagram', 'medical diagram', 'anatomy chart', 'infographic',
            'logo', 'icon', 'symbol', 'graphic', 'design', 'vector',
            'plant', 'flower', 'leaf', 'tree', 'nature', 'landscape',
            'full face', 'portrait', 'headshot', 'person', 'people', 'man', 'woman'
        ]
    
    def search_bing_images(self, query, count=50):
        """使用Bing图片搜索"""
        images = []
        try:
            # Bing图片搜索URL
            search_url = "https://www.bing.com/images/search"
            params = {
                'q': query,
                'form': 'HDRSC2',
                'first': 1,
                'count': count,
                'qft': '+filterui:photo-photo+filterui:aspect-square+filterui:imagesize-medium'
            }
            
            print(f"    🔍 Bing搜索: '{query}'")
            response = self.session.get(search_url, params=params, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # 查找图片链接
                img_tags = soup.find_all('img', {'class': 'mimg'})
                
                for img_tag in img_tags:
                    src = img_tag.get('src')
                    if src and src.startswith('http'):
                        # 过滤掉明显不合适的图片
                        if not self._is_excluded_url(src):
                            images.append({
                                'url': src,
                                'source': 'bing',
                                'query': query,
                                'alt': img_tag.get('alt', '')
                            })
                
                print(f"      找到 {len(images)} 张潜在图片")
                
        except Exception as e:
            print(f"      ❌ Bing搜索出错: {e}")
        
        return images
    
    def search_duckduckgo_images(self, query, count=50):
        """使用DuckDuckGo图片搜索"""
        images = []
        try:
            # DuckDuckGo图片搜索API
            search_url = "https://duckduckgo.com/"
            
            # 首先获取搜索token
            response = self.session.get(search_url, timeout=15)
            
            # 然后搜索图片
            api_url = "https://duckduckgo.com/i.js"
            params = {
                'l': 'us-en',
                'o': 'json',
                'q': query,
                'vqd': '',  # 需要从首页获取
                'f': ',,,',
                'p': '1'
            }
            
            print(f"    🔍 DuckDuckGo搜索: '{query}'")
            # 这里简化处理，实际需要解析vqd token
            
        except Exception as e:
            print(f"      ❌ DuckDuckGo搜索出错: {e}")
        
        return images
    
    def search_unsplash_api(self, query, count=30):
        """使用Unsplash API搜索（需要API密钥）"""
        images = []
        try:
            # 注意：实际使用需要Unsplash API密钥
            # 这里提供框架，用户需要自己申请API密钥
            api_key = "YOUR_UNSPLASH_API_KEY"  # 用户需要替换
            
            if api_key == "YOUR_UNSPLASH_API_KEY":
                print(f"      ⚠️  需要Unsplash API密钥")
                return images
            
            api_url = "https://api.unsplash.com/search/photos"
            headers = {'Authorization': f'Client-ID {api_key}'}
            params = {
                'query': query,
                'per_page': count,
                'orientation': 'landscape'
            }
            
            print(f"    🔍 Unsplash搜索: '{query}'")
            response = self.session.get(api_url, headers=headers, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                for result in data.get('results', []):
                    images.append({
                        'url': result['urls']['regular'],
                        'source': 'unsplash',
                        'query': query,
                        'description': result.get('description', '')
                    })
                
                print(f"      找到 {len(images)} 张高质量图片")
                
        except Exception as e:
            print(f"      ❌ Unsplash搜索出错: {e}")
        
        return images
    
    def search_wikimedia_commons(self, query, count=30):
        """搜索Wikimedia Commons"""
        images = []
        try:
            api_url = "https://commons.wikimedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': f'filetype:bitmap {query}',
                'srnamespace': 6,
                'srlimit': count
            }
            
            print(f"    🔍 Wikimedia搜索: '{query}'")
            response = self.session.get(api_url, params=params, timeout=15)
            data = response.json()
            
            if 'query' in data and 'search' in data['query']:
                for item in data['query']['search']:
                    title = item['title']
                    if title.startswith('File:'):
                        # 过滤不合适的文件名
                        if not self._is_excluded_filename(title):
                            # 获取文件URL
                            file_params = {
                                'action': 'query',
                                'format': 'json',
                                'titles': title,
                                'prop': 'imageinfo',
                                'iiprop': 'url|size|mime'
                            }
                            
                            file_response = self.session.get(api_url, params=file_params, timeout=10)
                            file_data = file_response.json()
                            
                            if 'query' in file_data and 'pages' in file_data['query']:
                                for page_id, page_data in file_data['query']['pages'].items():
                                    if 'imageinfo' in page_data:
                                        image_info = page_data['imageinfo'][0]
                                        if 'url' in image_info:
                                            mime_type = image_info.get('mime', '')
                                            if mime_type.startswith('image/'):
                                                images.append({
                                                    'url': image_info['url'],
                                                    'source': 'wikimedia',
                                                    'query': query,
                                                    'title': title,
                                                    'size': image_info.get('size', 0)
                                                })
                
                print(f"      找到 {len(images)} 张Wikimedia图片")
                
        except Exception as e:
            print(f"      ❌ Wikimedia搜索出错: {e}")
        
        return images
    
    def _is_excluded_url(self, url):
        """检查URL是否包含排除关键词"""
        url_lower = url.lower()
        return any(keyword in url_lower for keyword in self.exclude_keywords)
    
    def _is_excluded_filename(self, filename):
        """检查文件名是否包含排除关键词"""
        filename_lower = filename.lower()
        return any(keyword in filename_lower for keyword in self.exclude_keywords)
    
    def validate_eye_image(self, image_path):
        """验证是否为合适的眼部图片"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
                # 基本尺寸检查
                if width < 200 or height < 150:
                    return False, "图片太小"
                
                # 宽高比检查（眼部图片通常不会太极端）
                aspect_ratio = width / height
                if aspect_ratio < 0.5 or aspect_ratio > 3.0:
                    return False, f"宽高比不合适: {aspect_ratio:.2f}"
                
                # 转换为RGB进行分析
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_array = np.array(img)
                
                # 颜色分析 - 检查是否有肤色
                r_channel = img_array[:, :, 0]
                g_channel = img_array[:, :, 1]
                b_channel = img_array[:, :, 2]
                
                # 肤色检测（简单的RGB范围）
                skin_mask = (
                    (r_channel > 95) & (g_channel > 40) & (b_channel > 20) &
                    (r_channel > g_channel) & (r_channel > b_channel) &
                    (abs(r_channel.astype(int) - g_channel.astype(int)) > 15)
                )
                
                skin_ratio = np.sum(skin_mask) / (width * height)
                if skin_ratio < 0.1:  # 至少10%的肤色像素
                    return False, "缺少肤色特征"
                
                # 检查颜色多样性（避免单色图片）
                unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
                color_diversity = unique_colors / (width * height)
                if color_diversity < 0.05:
                    return False, "颜色过于单一"
                
                # 检查是否过于饱和（卡通特征）
                hsv_img = img.convert('HSV')
                hsv_array = np.array(hsv_img)
                saturation = hsv_array[:, :, 1]
                high_saturation_ratio = np.sum(saturation > 200) / (width * height)
                if high_saturation_ratio > 0.4:
                    return False, "饱和度过高（可能是卡通）"
                
                return True, "合适的眼部图片"
                
        except Exception as e:
            return False, f"验证出错: {e}"
    
    def download_and_validate_image(self, image_info, filepath):
        """下载并验证图片"""
        try:
            url = image_info['url']
            
            # 下载图片
            response = self.session.get(url, timeout=20, stream=True)
            response.raise_for_status()
            
            # 检查内容类型
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                return False, "不是图片格式"
            
            # 检查文件大小
            content_length = response.headers.get('content-length')
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > 10:  # 超过10MB
                    return False, f"文件过大: {size_mb:.1f}MB"
                if int(content_length) < 10240:  # 小于10KB
                    return False, f"文件过小: {int(content_length)/1024:.1f}KB"
            
            # 保存图片
            with open(filepath, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded > 10 * 1024 * 1024:  # 超过10MB停止
                            os.remove(filepath)
                            return False, "下载过程中文件过大"
            
            # 验证图片
            is_valid, message = self.validate_eye_image(filepath)
            if not is_valid:
                os.remove(filepath)
                return False, message
            
            return True, "成功下载并验证"
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return False, f"下载错误: {e}"
    
    def collect_images_for_class(self, class_name, target_count):
        """为特定类别收集图片"""
        print(f"\n👁️  收集 {class_name} 类别图片...")
        print(f"目标: {target_count} 张真实人眼图片")
        
        class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        
        # 清理现有图片
        if os.path.exists(class_dir):
            existing_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for filename in existing_files:
                filepath = os.path.join(class_dir, filename)
                os.remove(filepath)
                print(f"  🗑️  删除旧图片: {filename}")
        
        keywords = self.search_keywords[class_name]
        all_images = []
        
        # 从多个搜索引擎收集图片
        for keyword in keywords[:3]:  # 使用前3个关键词
            print(f"  🔍 搜索关键词: '{keyword}'")
            
            # Bing搜索
            bing_images = self.search_bing_images(keyword, 20)
            all_images.extend(bing_images)
            
            # Wikimedia搜索
            wiki_images = self.search_wikimedia_commons(keyword, 15)
            all_images.extend(wiki_images)
            
            # Unsplash搜索（如果有API密钥）
            unsplash_images = self.search_unsplash_api(keyword, 10)
            all_images.extend(unsplash_images)
            
            time.sleep(2)  # 避免请求过快
        
        # 去重和随机化
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
            
            filename = f"{class_name}_web_{downloaded_count + 1:03d}.jpg"
            filepath = os.path.join(class_dir, filename)
            
            success, message = self.download_and_validate_image(image_info, filepath)
            
            if success:
                downloaded_count += 1
                print(f"    ✅ {filename} - {message}")
            else:
                print(f"    ❌ 跳过: {message}")
            
            # 随机延迟，避免被封IP
            time.sleep(random.uniform(1, 4))
        
        print(f"  📊 成功收集: {downloaded_count}/{target_count} 张图片")
        return downloaded_count
    
    def collect_complete_dataset(self):
        """收集完整的数据集"""
        print("🌐 高级人眼图片爬虫")
        print("=" * 60)
        print("🎯 目标: 收集100张真实人眼图片 (25:25:50)")
        print("✅ 包含: 服药眼部、饮酒眼部、正常眼部")
        print("❌ 排除: 动物、昆虫、植物、头像、人像、卡通")
        
        # 目标分布
        target_distribution = {
            'stoner': 25,   # 服药（大麻等）
            'alcohol': 25,  # 饮酒
            'normal': 50    # 正常
        }
        
        results = {}
        
        for class_name, target_count in target_distribution.items():
            collected = self.collect_images_for_class(class_name, target_count)
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
    print("👁️  高级人眼图片爬虫")
    print("🎯 专门收集真实的人眼照片")
    print("=" * 60)
    
    scraper = AdvancedEyeScraper()
    results = scraper.collect_complete_dataset()
    
    print(f"\n💡 下一步:")
    print(f"  1. 验证图片质量: python simple_validator.py")
    print(f"  2. 重新训练模型: python simple_ml_trainer.py")
    print(f"  3. 测试模型性能: python simple_inference.py --test")
    print(f"  4. 推送到GitHub: git add . && git commit && git push")
    
    return results

if __name__ == "__main__":
    results = main()
