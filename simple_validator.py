"""
Simple image validator without OpenCV dependency
"""

import os
from PIL import Image
import config

def validate_image_simple(filepath):
    """Simple image validation using PIL only"""
    try:
        # Check if file exists and has content
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return False, "File doesn't exist or is empty"
        
        # Try to open with PIL
        with Image.open(filepath) as img:
            # Check image dimensions
            width, height = img.size
            if width < 50 or height < 50:
                return False, f"Image too small: {width}x{height}"
            
            if width > 5000 or height > 5000:
                return False, f"Image too large: {width}x{height}"
            
            # Check if image has valid mode
            if img.mode not in ['RGB', 'RGBA', 'L']:
                return False, f"Invalid image mode: {img.mode}"
            
            # Try to load the image data to verify it's not corrupted
            img.load()
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def validate_class_directory_simple(class_name):
    """Validate all images in a class directory (simplified)"""
    class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
    
    if not os.path.exists(class_dir):
        print(f"❌ Directory doesn't exist: {class_dir}")
        return [], []
    
    valid_images = []
    invalid_images = []
    
    image_files = [f for f in os.listdir(class_dir) 
                  if os.path.splitext(f.lower())[1] in config.SUPPORTED_FORMATS]
    
    print(f"🔍 Validating {len(image_files)} images in {class_name} class")
    
    for filename in image_files:
        filepath = os.path.join(class_dir, filename)
        
        # Basic validation
        is_valid, reason = validate_image_simple(filepath)
        
        if is_valid:
            # Get image info
            with Image.open(filepath) as img:
                width, height = img.size
                mode = img.mode
                file_size = os.path.getsize(filepath)
            
            image_info = {
                'filepath': filepath,
                'filename': filename,
                'class': class_name,
                'width': width,
                'height': height,
                'mode': mode,
                'file_size': file_size,
                'valid': True
            }
            
            valid_images.append(image_info)
            print(f"  ✅ {filename} ({width}x{height}, {file_size} bytes)")
        else:
            invalid_images.append({
                'filepath': filepath,
                'filename': filename,
                'class': class_name,
                'valid': False,
                'reason': reason
            })
            print(f"  ❌ {filename} - {reason}")
    
    print(f"📊 {class_name}: {len(valid_images)} valid, {len(invalid_images)} invalid images")
    
    return valid_images, invalid_images

def validate_all_classes_simple():
    """Validate images for all classes (simplified)"""
    print("🔍 SIMPLE IMAGE VALIDATION")
    print("=" * 50)
    
    all_valid = []
    all_invalid = []
    
    for class_name in config.CLASS_DISTRIBUTION.keys():
        print(f"\n📁 Validating {class_name} class...")
        valid_images, invalid_images = validate_class_directory_simple(class_name)
        all_valid.extend(valid_images)
        all_invalid.extend(invalid_images)
    
    print(f"\n📊 VALIDATION SUMMARY")
    print("=" * 50)
    print(f"✅ Valid images: {len(all_valid)}")
    print(f"❌ Invalid images: {len(all_invalid)}")
    
    # Show class distribution
    class_counts = {}
    for img in all_valid:
        class_name = img['class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"\n📊 Valid images by class:")
    for class_name, count in class_counts.items():
        target = config.CLASS_DISTRIBUTION[class_name]
        percentage = (count / target * 100) if target > 0 else 0
        print(f"  {class_name:8s}: {count:3d}/{target:3d} ({percentage:5.1f}%)")
    
    # Show image statistics
    if all_valid:
        widths = [img['width'] for img in all_valid]
        heights = [img['height'] for img in all_valid]
        sizes = [img['file_size'] for img in all_valid]
        
        print(f"\n📏 Image statistics:")
        print(f"  Width:  {min(widths):4d} - {max(widths):4d} (avg: {sum(widths)/len(widths):6.1f})")
        print(f"  Height: {min(heights):4d} - {max(heights):4d} (avg: {sum(heights)/len(heights):6.1f})")
        print(f"  Size:   {min(sizes):4d} - {max(sizes):4d} bytes (avg: {sum(sizes)/len(sizes):6.1f})")
    
    return all_valid, all_invalid

def show_sample_images():
    """Show information about sample images"""
    print(f"\n🖼️  SAMPLE IMAGES")
    print("=" * 50)
    
    for class_name in config.CLASS_DISTRIBUTION.keys():
        class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) 
                    if os.path.splitext(f.lower())[1] in config.SUPPORTED_FORMATS]
            
            if files:
                print(f"\n📁 {class_name.upper()} samples:")
                for i, filename in enumerate(files[:3]):  # Show first 3
                    filepath = os.path.join(class_dir, filename)
                    try:
                        with Image.open(filepath) as img:
                            width, height = img.size
                            mode = img.mode
                            size = os.path.getsize(filepath)
                        print(f"  {i+1}. {filename}")
                        print(f"     Size: {width}x{height}, Mode: {mode}, {size} bytes")
                    except Exception as e:
                        print(f"  {i+1}. {filename} - Error: {e}")
                
                if len(files) > 3:
                    print(f"     ... and {len(files) - 3} more")
            else:
                print(f"\n📁 {class_name.upper()}: No images found")

if __name__ == "__main__":
    print("🔍 SIMPLE IMAGE VALIDATION TOOL")
    print("🎯 Validating collected images without OpenCV dependency")
    
    try:
        # Validate all images
        valid_images, invalid_images = validate_all_classes_simple()
        
        # Show sample information
        show_sample_images()
        
        # Summary
        total_images = len(valid_images) + len(invalid_images)
        success_rate = (len(valid_images) / total_images * 100) if total_images > 0 else 0
        
        print(f"\n🎉 VALIDATION COMPLETED")
        print("=" * 50)
        print(f"📊 Results:")
        print(f"  Total processed: {total_images}")
        print(f"  Valid images: {len(valid_images)}")
        print(f"  Invalid images: {len(invalid_images)}")
        print(f"  Success rate: {success_rate:.1f}%")
        
        if len(valid_images) >= 15:
            print(f"\n✅ Validation successful! Ready for next steps.")
            print(f"💡 Next: Run preprocessing with 'python main.py --step preprocess'")
        elif len(valid_images) >= 10:
            print(f"\n⚠️  Partial success. Consider collecting more images.")
        else:
            print(f"\n❌ Low success rate. Check image collection process.")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
