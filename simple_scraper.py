"""
Simple image scraper for testing - collect 20 images
"""

import os
import time
import requests
from urllib.parse import urlparse
import config
from tqdm import tqdm

def download_image_from_url(url, filepath, timeout=10):
    """Download image from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Check if it's an image
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            return False
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Check file size
        if os.path.getsize(filepath) < 1024:  # Less than 1KB
            os.remove(filepath)
            return False
            
        return True
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def get_sample_image_urls():
    """Get sample image URLs for testing"""
    # Using some public domain/creative commons image URLs for testing
    sample_urls = [
        # Normal eyes (using placeholder/demo images)
        "https://via.placeholder.com/300x200/87CEEB/000000?text=Normal+Eye+1",
        "https://via.placeholder.com/300x200/98FB98/000000?text=Normal+Eye+2", 
        "https://via.placeholder.com/300x200/F0E68C/000000?text=Normal+Eye+3",
        "https://via.placeholder.com/300x200/DDA0DD/000000?text=Normal+Eye+4",
        "https://via.placeholder.com/300x200/F5DEB3/000000?text=Normal+Eye+5",
        "https://via.placeholder.com/300x200/FFB6C1/000000?text=Normal+Eye+6",
        "https://via.placeholder.com/300x200/20B2AA/000000?text=Normal+Eye+7",
        
        # Alcohol-related (red/bloodshot simulation)
        "https://via.placeholder.com/300x200/FF6347/000000?text=Alcohol+Eye+1",
        "https://via.placeholder.com/300x200/DC143C/000000?text=Alcohol+Eye+2",
        "https://via.placeholder.com/300x200/B22222/000000?text=Alcohol+Eye+3",
        "https://via.placeholder.com/300x200/CD5C5C/000000?text=Alcohol+Eye+4",
        "https://via.placeholder.com/300x200/FA8072/000000?text=Alcohol+Eye+5",
        "https://via.placeholder.com/300x200/E9967A/000000?text=Alcohol+Eye+6",
        "https://via.placeholder.com/300x200/FFA07A/000000?text=Alcohol+Eye+7",
        
        # Stoner-related (red/droopy simulation)
        "https://via.placeholder.com/300x200/FF4500/000000?text=Stoner+Eye+1",
        "https://via.placeholder.com/300x200/FF8C00/000000?text=Stoner+Eye+2",
        "https://via.placeholder.com/300x200/FF7F50/000000?text=Stoner+Eye+3",
        "https://via.placeholder.com/300x200/FF6347/000000?text=Stoner+Eye+4",
        "https://via.placeholder.com/300x200/FF69B4/000000?text=Stoner+Eye+5",
        "https://via.placeholder.com/300x200/FF1493/000000?text=Stoner+Eye+6",
    ]
    
    return sample_urls

def collect_sample_images():
    """Collect 20 sample images for testing"""
    print("🌐 Starting sample image collection...")
    print("📝 Note: Using placeholder images for demonstration")
    
    urls = get_sample_image_urls()
    
    # Distribute images across classes
    class_assignments = {
        'normal': urls[:7],
        'alcohol': urls[7:14], 
        'stoner': urls[14:]
    }
    
    total_downloaded = 0
    
    for class_name, class_urls in class_assignments.items():
        print(f"\n📥 Collecting {class_name} images...")
        class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        
        downloaded_count = 0
        
        for i, url in enumerate(tqdm(class_urls, desc=f"Downloading {class_name}")):
            filename = f"{class_name}_{i+1:03d}.jpg"
            filepath = os.path.join(class_dir, filename)
            
            if download_image_from_url(url, filepath):
                downloaded_count += 1
                total_downloaded += 1
                print(f"  ✅ Downloaded: {filename}")
            else:
                print(f"  ❌ Failed: {filename}")
            
            time.sleep(0.5)  # Be respectful
        
        print(f"📊 {class_name}: {downloaded_count}/{len(class_urls)} images downloaded")
    
    print(f"\n🎉 Collection completed!")
    print(f"📊 Total downloaded: {total_downloaded}/20 images")
    
    return total_downloaded

def collect_real_images_simple():
    """Try to collect some real images using simple requests"""
    print("🌐 Attempting to collect real images...")
    print("⚠️  Note: This is a simplified approach for demonstration")
    
    # Some public image APIs/sources that might work
    test_sources = [
        # Unsplash API (requires API key, so we'll skip)
        # Pixabay API (requires API key, so we'll skip)
        # Using some direct image URLs that are likely to work
    ]
    
    # For now, let's use the placeholder approach since we don't have API keys
    print("🔄 Using placeholder images for safe demonstration...")
    return collect_sample_images()

def validate_downloaded_images():
    """Validate the downloaded images"""
    print("\n🔍 Validating downloaded images...")
    
    total_valid = 0
    
    for class_name in config.CLASS_DISTRIBUTION.keys():
        class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        
        if not os.path.exists(class_dir):
            continue
            
        files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        valid_files = []
        
        for filename in files:
            filepath = os.path.join(class_dir, filename)
            
            # Basic validation
            if os.path.exists(filepath) and os.path.getsize(filepath) > 1024:
                valid_files.append(filename)
                total_valid += 1
        
        print(f"  {class_name}: {len(valid_files)} valid images")
    
    print(f"📊 Total valid images: {total_valid}")
    return total_valid

def show_collection_summary():
    """Show summary of collected images"""
    print("\n📋 COLLECTION SUMMARY")
    print("=" * 50)
    
    for class_name, target_count in config.CLASS_DISTRIBUTION.items():
        class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) 
                    if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            current_count = len(files)
            percentage = (current_count / target_count * 100) if target_count > 0 else 0
            
            print(f"📁 {class_name:8s}: {current_count:3d}/{target_count:3d} ({percentage:5.1f}%)")
            
            # Show first few files
            if files:
                print(f"   Sample files: {', '.join(files[:3])}")
                if len(files) > 3:
                    print(f"   ... and {len(files) - 3} more")
        else:
            print(f"📁 {class_name:8s}: 0/{target_count} (0.0%) - Directory not found")
    
    print("=" * 50)

if __name__ == "__main__":
    print("👁️  SIMPLE IMAGE SCRAPER - 20 Images Test")
    print("🎯 Goal: Collect 20 sample images for testing")
    
    start_time = time.time()
    
    try:
        # Collect images
        downloaded_count = collect_real_images_simple()
        
        # Validate images
        valid_count = validate_downloaded_images()
        
        # Show summary
        show_collection_summary()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n⏱️  Collection completed in {duration:.1f} seconds")
        
        if downloaded_count >= 15:
            print("✅ Collection successful!")
        elif downloaded_count >= 10:
            print("⚠️  Partial success - some images collected")
        else:
            print("❌ Collection mostly failed")
            
        print(f"\n💡 Next steps:")
        print(f"   1. Run validation: python main.py --step validate")
        print(f"   2. Check images in data/raw/ directories")
        print(f"   3. For real scraping, configure API keys in web_scraper.py")
        
    except Exception as e:
        print(f"❌ Error during collection: {e}")
        import traceback
        traceback.print_exc()
