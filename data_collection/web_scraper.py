"""
Web scraper for collecting eye images from various sources
"""

import os
import time
import requests
import json
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import logging
from tqdm import tqdm
import config

class EyeImageScraper:
    def __init__(self):
        self.setup_logging()
        self.setup_driver()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(config.LOGS_DIR, 'scraper.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_driver(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
    def search_google_images(self, query, max_images=100):
        """Search Google Images for a specific query"""
        images = []
        try:
            search_url = f"https://www.google.com/search?q={query}&tbm=isch"
            self.driver.get(search_url)
            
            # Scroll to load more images
            for _ in range(3):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
            # Find image elements
            img_elements = self.driver.find_elements(By.CSS_SELECTOR, "img[data-src]")
            
            for img in img_elements[:max_images]:
                try:
                    src = img.get_attribute('data-src') or img.get_attribute('src')
                    if src and src.startswith('http'):
                        images.append(src)
                except Exception as e:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error searching Google Images: {e}")
            
        return images
    
    def search_bing_images(self, query, max_images=100):
        """Search Bing Images for a specific query"""
        images = []
        try:
            search_url = f"https://www.bing.com/images/search?q={query}"
            self.driver.get(search_url)
            
            # Scroll to load more images
            for _ in range(3):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
            # Find image elements
            img_elements = self.driver.find_elements(By.CSS_SELECTOR, ".mimg")
            
            for img in img_elements[:max_images]:
                try:
                    src = img.get_attribute('src')
                    if src and src.startswith('http'):
                        images.append(src)
                except Exception as e:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error searching Bing Images: {e}")
            
        return images
    
    def download_image(self, url, filepath):
        """Download an image from URL"""
        try:
            response = self.session.get(url, timeout=config.REQUEST_TIMEOUT, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            return True
        except Exception as e:
            self.logger.error(f"Error downloading {url}: {e}")
            return False
    
    def collect_images_for_class(self, class_name, target_count):
        """Collect images for a specific class"""
        self.logger.info(f"Starting collection for {class_name} class, target: {target_count}")
        
        class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        downloaded_count = 0
        keywords = config.SEARCH_KEYWORDS[class_name]
        
        for keyword in keywords:
            if downloaded_count >= target_count:
                break
                
            self.logger.info(f"Searching for: {keyword}")
            
            # Search multiple sources
            google_images = self.search_google_images(keyword, config.MAX_IMAGES_PER_SEARCH)
            bing_images = self.search_bing_images(keyword, config.MAX_IMAGES_PER_SEARCH)
            
            all_images = list(set(google_images + bing_images))  # Remove duplicates
            
            for i, img_url in enumerate(tqdm(all_images, desc=f"Downloading {keyword}")):
                if downloaded_count >= target_count:
                    break
                    
                filename = f"{class_name}_{downloaded_count + 1:04d}.jpg"
                filepath = os.path.join(class_dir, filename)
                
                if self.download_image(img_url, filepath):
                    downloaded_count += 1
                    self.logger.info(f"Downloaded: {filename}")
                    
                time.sleep(config.DOWNLOAD_DELAY)
                
        self.logger.info(f"Completed {class_name}: {downloaded_count}/{target_count} images")
        return downloaded_count
    
    def collect_all_images(self):
        """Collect images for all classes"""
        self.logger.info("Starting image collection for all classes")
        
        results = {}
        for class_name, target_count in config.CLASS_DISTRIBUTION.items():
            results[class_name] = self.collect_images_for_class(class_name, target_count)
            
        self.logger.info("Collection completed!")
        self.logger.info(f"Results: {results}")
        
        return results
    
    def __del__(self):
        if hasattr(self, 'driver'):
            self.driver.quit()

if __name__ == "__main__":
    scraper = EyeImageScraper()
    scraper.collect_all_images()
