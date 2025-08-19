"""
Create test images for demonstration
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import config

def create_test_image(width=300, height=200, color=(128, 128, 128), text="Test Image"):
    """Create a test image with specified color and text"""
    # Create image
    img = Image.new('RGB', (width, height), color)
    
    # Add text
    draw = ImageDraw.Draw(img)
    
    # Try to use a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Calculate text position (center)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw text
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    return img

def create_eye_like_image(width=300, height=200, eye_color=(100, 150, 200), 
                         bloodshot=False, droopy=False, class_name="normal"):
    """Create a more eye-like test image"""
    img = Image.new('RGB', (width, height), (240, 220, 200))  # Skin tone background
    draw = ImageDraw.Draw(img)
    
    # Draw eye shape (oval)
    eye_width = width // 3
    eye_height = height // 4
    eye_x = width // 2 - eye_width // 2
    eye_y = height // 2 - eye_height // 2
    
    # Eye white
    white_color = (255, 255, 255)
    if bloodshot:
        white_color = (255, 200, 200)  # Reddish white
    
    draw.ellipse([eye_x, eye_y, eye_x + eye_width, eye_y + eye_height], 
                fill=white_color, outline=(0, 0, 0), width=2)
    
    # Iris
    iris_size = eye_height // 2
    iris_x = eye_x + eye_width // 2 - iris_size // 2
    iris_y = eye_y + eye_height // 2 - iris_size // 2
    
    if droopy:
        iris_y += 5  # Make iris appear lower (droopy)
    
    draw.ellipse([iris_x, iris_y, iris_x + iris_size, iris_y + iris_size],
                fill=eye_color, outline=(0, 0, 0), width=1)
    
    # Pupil
    pupil_size = iris_size // 3
    pupil_x = iris_x + iris_size // 2 - pupil_size // 2
    pupil_y = iris_y + iris_size // 2 - pupil_size // 2
    
    draw.ellipse([pupil_x, pupil_y, pupil_x + pupil_size, pupil_y + pupil_size],
                fill=(0, 0, 0))
    
    # Add eyelids
    if droopy:
        # Droopy eyelids
        draw.arc([eye_x - 10, eye_y - 5, eye_x + eye_width + 10, eye_y + eye_height + 5],
                start=0, end=180, fill=(0, 0, 0), width=3)
    else:
        # Normal eyelids
        draw.arc([eye_x - 5, eye_y, eye_x + eye_width + 5, eye_y + eye_height],
                start=0, end=180, fill=(0, 0, 0), width=2)
    
    # Add text label
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    label = f"{class_name.upper()} EYE"
    draw.text((10, 10), label, fill=(0, 0, 0), font=font)
    
    return img

def create_test_dataset():
    """Create a test dataset with 20 images"""
    print("🎨 Creating test images for demonstration...")
    
    # Define image characteristics for each class
    image_configs = {
        'normal': {
            'count': 7,
            'colors': [(100, 150, 200), (80, 120, 180), (120, 160, 220), 
                      (90, 140, 190), (110, 170, 210), (85, 135, 185), (105, 155, 205)],
            'bloodshot': False,
            'droopy': False
        },
        'alcohol': {
            'count': 7, 
            'colors': [(150, 100, 100), (180, 120, 120), (160, 90, 90),
                      (170, 110, 110), (140, 80, 80), (190, 130, 130), (155, 95, 95)],
            'bloodshot': True,
            'droopy': False
        },
        'stoner': {
            'count': 6,
            'colors': [(120, 100, 80), (140, 120, 100), (130, 110, 90),
                      (150, 130, 110), (125, 105, 85), (135, 115, 95)],
            'bloodshot': True,
            'droopy': True
        }
    }
    
    total_created = 0
    
    for class_name, config_data in image_configs.items():
        print(f"\n🖼️  Creating {class_name} images...")
        class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        
        created_count = 0
        
        for i in range(config_data['count']):
            filename = f"{class_name}_{i+1:03d}.jpg"
            filepath = os.path.join(class_dir, filename)
            
            # Get color for this image
            color = config_data['colors'][i % len(config_data['colors'])]
            
            # Create eye-like image
            img = create_eye_like_image(
                width=300, 
                height=200,
                eye_color=color,
                bloodshot=config_data['bloodshot'],
                droopy=config_data['droopy'],
                class_name=class_name
            )
            
            # Save image
            img.save(filepath, 'JPEG', quality=85)
            created_count += 1
            total_created += 1
            
            print(f"  ✅ Created: {filename}")
        
        print(f"📊 {class_name}: {created_count} images created")
    
    print(f"\n🎉 Test dataset creation completed!")
    print(f"📊 Total created: {total_created} images")
    
    return total_created

def validate_created_images():
    """Validate the created test images"""
    print("\n🔍 Validating created images...")
    
    total_valid = 0
    
    for class_name in config.CLASS_DISTRIBUTION.keys():
        class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        
        if not os.path.exists(class_dir):
            print(f"  ❌ {class_name}: Directory not found")
            continue
            
        files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        valid_files = []
        
        for filename in files:
            filepath = os.path.join(class_dir, filename)
            
            # Basic validation
            if os.path.exists(filepath) and os.path.getsize(filepath) > 1024:
                try:
                    # Try to open with PIL to verify it's a valid image
                    with Image.open(filepath) as img:
                        img.verify()
                    valid_files.append(filename)
                    total_valid += 1
                except Exception as e:
                    print(f"    ⚠️  Invalid image: {filename} - {e}")
        
        print(f"  ✅ {class_name}: {len(valid_files)} valid images")
    
    print(f"📊 Total valid images: {total_valid}")
    return total_valid

def show_image_info():
    """Show information about created images"""
    print("\n📋 IMAGE DATASET INFORMATION")
    print("=" * 50)
    
    total_size = 0
    
    for class_name in config.CLASS_DISTRIBUTION.keys():
        class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if files:
                # Calculate total size
                class_size = sum(os.path.getsize(os.path.join(class_dir, f)) for f in files)
                total_size += class_size
                
                # Get sample image info
                sample_file = os.path.join(class_dir, files[0])
                with Image.open(sample_file) as img:
                    width, height = img.size
                    mode = img.mode
                
                print(f"📁 {class_name:8s}: {len(files)} images")
                print(f"   Size: {width}x{height}, Mode: {mode}")
                print(f"   Total size: {class_size/1024:.1f} KB")
                print(f"   Files: {', '.join(files[:3])}")
                if len(files) > 3:
                    print(f"          ... and {len(files) - 3} more")
            else:
                print(f"📁 {class_name:8s}: No images found")
        else:
            print(f"📁 {class_name:8s}: Directory not found")
    
    print(f"\n📊 Total dataset size: {total_size/1024:.1f} KB")
    print("=" * 50)

if __name__ == "__main__":
    print("🎨 TEST IMAGE CREATOR")
    print("🎯 Goal: Create 20 test images for demonstration")
    
    try:
        # Create test images
        created_count = create_test_dataset()
        
        # Validate images
        valid_count = validate_created_images()
        
        # Show information
        show_image_info()
        
        if created_count >= 15:
            print("\n✅ Test dataset creation successful!")
        elif created_count >= 10:
            print("\n⚠️  Partial success - some images created")
        else:
            print("\n❌ Test dataset creation failed")
            
        print(f"\n💡 Next steps:")
        print(f"   1. Run validation: python main.py --step validate")
        print(f"   2. View images in data/raw/ directories")
        print(f"   3. Run preprocessing: python main.py --step preprocess")
        
    except Exception as e:
        print(f"❌ Error during creation: {e}")
        import traceback
        traceback.print_exc()
