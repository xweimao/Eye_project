"""
Utility functions for data handling and analysis
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import cv2
from PIL import Image
import config

def analyze_dataset_statistics():
    """Analyze and display dataset statistics"""
    print("Dataset Statistics Analysis")
    print("=" * 50)
    
    # Check raw data
    raw_stats = {}
    for class_name in config.CLASS_DISTRIBUTION.keys():
        class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) 
                    if os.path.splitext(f.lower())[1] in config.SUPPORTED_FORMATS]
            raw_stats[class_name] = len(files)
        else:
            raw_stats[class_name] = 0
    
    print("Raw Data:")
    total_raw = sum(raw_stats.values())
    for class_name, count in raw_stats.items():
        percentage = (count / total_raw * 100) if total_raw > 0 else 0
        print(f"  {class_name}: {count} images ({percentage:.1f}%)")
    print(f"  Total: {total_raw} images")
    
    # Check processed data
    metadata_path = os.path.join(config.PROCESSED_DATA_DIR, 'dataset_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            processed_data = json.load(f)
        
        print("\nProcessed Data:")
        for split_name, images in processed_data.items():
            split_stats = Counter(img['class'] for img in images)
            total_split = len(images)
            print(f"  {split_name.upper()} ({total_split} images):")
            for class_name, count in split_stats.items():
                percentage = (count / total_split * 100) if total_split > 0 else 0
                print(f"    {class_name}: {count} ({percentage:.1f}%)")
    
    return raw_stats

def visualize_data_distribution():
    """Create visualizations of data distribution"""
    # Analyze dataset
    raw_stats = analyze_dataset_statistics()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Raw data distribution (pie chart)
    if sum(raw_stats.values()) > 0:
        axes[0, 0].pie(raw_stats.values(), labels=raw_stats.keys(), autopct='%1.1f%%')
        axes[0, 0].set_title('Raw Data Distribution')
    
    # Raw data distribution (bar chart)
    axes[0, 1].bar(raw_stats.keys(), raw_stats.values(), color=['red', 'orange', 'green'])
    axes[0, 1].set_title('Raw Data Count by Class')
    axes[0, 1].set_ylabel('Number of Images')
    
    # Processed data distribution
    metadata_path = os.path.join(config.PROCESSED_DATA_DIR, 'dataset_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            processed_data = json.load(f)
        
        # Split distribution
        split_counts = {split: len(images) for split, images in processed_data.items()}
        axes[1, 0].bar(split_counts.keys(), split_counts.values(), color=['blue', 'cyan', 'purple'])
        axes[1, 0].set_title('Data Split Distribution')
        axes[1, 0].set_ylabel('Number of Images')
        
        # Class distribution in training set
        if 'train' in processed_data:
            train_stats = Counter(img['class'] for img in processed_data['train'])
            axes[1, 1].bar(train_stats.keys(), train_stats.values(), color=['red', 'orange', 'green'])
            axes[1, 1].set_title('Training Set Class Distribution')
            axes[1, 1].set_ylabel('Number of Images')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.LOGS_DIR, 'data_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()

def analyze_image_properties():
    """Analyze properties of collected images"""
    print("\nImage Properties Analysis")
    print("=" * 50)
    
    properties = {
        'widths': [],
        'heights': [],
        'aspects': [],
        'file_sizes': [],
        'classes': []
    }
    
    for class_name in config.CLASS_DISTRIBUTION.keys():
        class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            continue
            
        files = [f for f in os.listdir(class_dir) 
                if os.path.splitext(f.lower())[1] in config.SUPPORTED_FORMATS]
        
        for filename in files:
            filepath = os.path.join(class_dir, filename)
            try:
                # Get file size
                file_size = os.path.getsize(filepath) / 1024  # KB
                
                # Get image dimensions
                with Image.open(filepath) as img:
                    width, height = img.size
                    aspect_ratio = width / height
                
                properties['widths'].append(width)
                properties['heights'].append(height)
                properties['aspects'].append(aspect_ratio)
                properties['file_sizes'].append(file_size)
                properties['classes'].append(class_name)
                
            except Exception as e:
                print(f"Error analyzing {filepath}: {e}")
                continue
    
    if not properties['widths']:
        print("No valid images found for analysis")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(properties)
    
    print(f"Total images analyzed: {len(df)}")
    print(f"\nDimensions:")
    print(f"  Width: {df['widths'].min()}-{df['widths'].max()} (avg: {df['widths'].mean():.1f})")
    print(f"  Height: {df['heights'].min()}-{df['heights'].max()} (avg: {df['heights'].mean():.1f})")
    print(f"  Aspect ratio: {df['aspects'].min():.2f}-{df['aspects'].max():.2f} (avg: {df['aspects'].mean():.2f})")
    print(f"  File size: {df['file_sizes'].min():.1f}-{df['file_sizes'].max():.1f} KB (avg: {df['file_sizes'].mean():.1f})")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Width distribution
    axes[0, 0].hist(df['widths'], bins=30, alpha=0.7, color='blue')
    axes[0, 0].set_title('Image Width Distribution')
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    
    # Height distribution
    axes[0, 1].hist(df['heights'], bins=30, alpha=0.7, color='green')
    axes[0, 1].set_title('Image Height Distribution')
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Frequency')
    
    # Aspect ratio distribution
    axes[1, 0].hist(df['aspects'], bins=30, alpha=0.7, color='red')
    axes[1, 0].set_title('Aspect Ratio Distribution')
    axes[1, 0].set_xlabel('Aspect Ratio (W/H)')
    axes[1, 0].set_ylabel('Frequency')
    
    # File size distribution
    axes[1, 1].hist(df['file_sizes'], bins=30, alpha=0.7, color='purple')
    axes[1, 1].set_title('File Size Distribution')
    axes[1, 1].set_xlabel('File Size (KB)')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.LOGS_DIR, 'image_properties.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def create_sample_grid(num_samples=9):
    """Create a grid showing sample images from each class"""
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    sample_idx = 0
    
    for class_name in config.CLASS_DISTRIBUTION.keys():
        class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            continue
            
        files = [f for f in os.listdir(class_dir) 
                if os.path.splitext(f.lower())[1] in config.SUPPORTED_FORMATS]
        
        # Show 3 samples per class
        for i in range(min(3, len(files))):
            if sample_idx >= 9:
                break
                
            filepath = os.path.join(class_dir, files[i])
            try:
                img = cv2.imread(filepath)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[sample_idx].imshow(img_rgb)
                    axes[sample_idx].set_title(f'{class_name} - {files[i]}')
                    axes[sample_idx].axis('off')
                    sample_idx += 1
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
    
    # Hide unused subplots
    for i in range(sample_idx, 9):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.LOGS_DIR, 'sample_images.png'), dpi=300, bbox_inches='tight')
    plt.show()

def cleanup_empty_directories():
    """Remove empty directories in the data folder"""
    removed_dirs = []
    
    for root, dirs, files in os.walk(config.DATA_DIR, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):  # Directory is empty
                    os.rmdir(dir_path)
                    removed_dirs.append(dir_path)
            except OSError:
                pass  # Directory not empty or permission error
    
    if removed_dirs:
        print(f"Removed {len(removed_dirs)} empty directories:")
        for dir_path in removed_dirs:
            print(f"  {dir_path}")
    else:
        print("No empty directories found")
    
    return removed_dirs

def export_dataset_summary():
    """Export a comprehensive dataset summary to JSON"""
    summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'config': {
            'total_target': config.TOTAL_IMAGES,
            'class_distribution': config.CLASS_DISTRIBUTION,
            'image_size': config.IMAGE_SIZE
        }
    }
    
    # Raw data statistics
    raw_stats = {}
    for class_name in config.CLASS_DISTRIBUTION.keys():
        class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) 
                    if os.path.splitext(f.lower())[1] in config.SUPPORTED_FORMATS]
            raw_stats[class_name] = len(files)
        else:
            raw_stats[class_name] = 0
    
    summary['raw_data'] = raw_stats
    
    # Processed data statistics
    metadata_path = os.path.join(config.PROCESSED_DATA_DIR, 'dataset_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            processed_data = json.load(f)
        
        processed_stats = {}
        for split_name, images in processed_data.items():
            split_stats = Counter(img['class'] for img in images)
            processed_stats[split_name] = dict(split_stats)
        
        summary['processed_data'] = processed_stats
    
    # Save summary
    summary_path = os.path.join(config.LOGS_DIR, 'dataset_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Dataset summary exported to: {summary_path}")
    return summary

if __name__ == "__main__":
    print("Running dataset analysis...")
    
    # Run all analyses
    analyze_dataset_statistics()
    visualize_data_distribution()
    analyze_image_properties()
    create_sample_grid()
    cleanup_empty_directories()
    export_dataset_summary()
    
    print("\nAnalysis completed!")
