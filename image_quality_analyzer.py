"""
Advanced image quality analyzer to identify problematic images
"""

import os
import numpy as np
from PIL import Image, ImageStat
import json
from tqdm import tqdm
import config

class ImageQualityAnalyzer:
    def __init__(self):
        self.problematic_indicators = {
            'cartoon_keywords': ['cartoon', 'illustration', 'drawing', 'diagram', 'sketch', 'animated'],
            'medical_keywords': ['anatomy', 'medical', 'diagram', 'cross-section', 'illustration'],
            'synthetic_keywords': ['synthetic', 'artificial', 'generated', 'rendered', '3d']
        }
        
    def analyze_image_characteristics(self, image_path):
        """Analyze image characteristics to detect problematic content"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_array = np.array(img)
                width, height = img.size
                
                # Basic image statistics
                stats = ImageStat.Stat(img)
                
                analysis = {
                    'filepath': image_path,
                    'filename': os.path.basename(image_path),
                    'width': width,
                    'height': height,
                    'aspect_ratio': width / height,
                    'file_size': os.path.getsize(image_path),
                    'mode': img.mode,
                    'format': img.format,
                }
                
                # Color analysis
                r_channel = img_array[:, :, 0]
                g_channel = img_array[:, :, 1]
                b_channel = img_array[:, :, 2]
                
                analysis.update({
                    'mean_brightness': np.mean(img_array),
                    'std_brightness': np.std(img_array),
                    'red_mean': np.mean(r_channel),
                    'green_mean': np.mean(g_channel),
                    'blue_mean': np.mean(b_channel),
                    'red_std': np.std(r_channel),
                    'green_std': np.std(g_channel),
                    'blue_std': np.std(b_channel),
                })
                
                # Detect potential issues
                issues = []
                
                # 1. Check for very uniform colors (diagrams/illustrations)
                color_uniformity = (analysis['red_std'] + analysis['green_std'] + analysis['blue_std']) / 3
                if color_uniformity < 20:
                    issues.append('very_uniform_colors')
                
                # 2. Check for extreme aspect ratios
                if analysis['aspect_ratio'] < 0.5 or analysis['aspect_ratio'] > 3.0:
                    issues.append('extreme_aspect_ratio')
                
                # 3. Check for very small images
                if width < 200 or height < 200:
                    issues.append('too_small')
                
                # 4. Check for very large images (might be full body/face shots)
                if width > 2000 or height > 2000:
                    issues.append('potentially_too_large')
                
                # 5. Check color distribution for artificial patterns
                gray = np.mean(img_array, axis=2)
                edge_variance = np.var(np.diff(gray, axis=0)) + np.var(np.diff(gray, axis=1))
                if edge_variance < 10:
                    issues.append('low_texture_variation')
                
                # 6. Check for limited color palette (diagrams)
                unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
                total_pixels = width * height
                color_diversity = unique_colors / total_pixels
                if color_diversity < 0.1:
                    issues.append('limited_color_palette')
                
                # 7. Check for overly saturated colors (cartoons)
                hsv_img = img.convert('HSV')
                hsv_array = np.array(hsv_img)
                saturation = hsv_array[:, :, 1]
                high_saturation_ratio = np.sum(saturation > 200) / total_pixels
                if high_saturation_ratio > 0.3:
                    issues.append('overly_saturated')
                
                analysis['issues'] = issues
                analysis['issue_count'] = len(issues)
                analysis['quality_score'] = max(0, 10 - len(issues) * 2)  # 0-10 scale
                
                return analysis
                
        except Exception as e:
            return {
                'filepath': image_path,
                'filename': os.path.basename(image_path),
                'error': str(e),
                'issues': ['analysis_failed'],
                'issue_count': 10,
                'quality_score': 0
            }
    
    def analyze_dataset(self):
        """Analyze all images in the dataset"""
        print("🔍 ANALYZING DATASET QUALITY")
        print("=" * 60)
        
        all_analyses = []
        
        for class_name in config.CLASS_DISTRIBUTION.keys():
            class_dir = os.path.join(config.RAW_DATA_DIR, class_name)
            
            if not os.path.exists(class_dir):
                print(f"⚠️  Directory not found: {class_dir}")
                continue
            
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"\n📁 Analyzing {class_name} class ({len(image_files)} images)...")
            
            class_analyses = []
            for filename in tqdm(image_files, desc=f"Analyzing {class_name}"):
                filepath = os.path.join(class_dir, filename)
                analysis = self.analyze_image_characteristics(filepath)
                analysis['class'] = class_name
                class_analyses.append(analysis)
                all_analyses.append(analysis)
            
            # Class summary
            problematic_count = sum(1 for a in class_analyses if a['issue_count'] > 2)
            avg_quality = np.mean([a['quality_score'] for a in class_analyses])
            
            print(f"📊 {class_name} summary:")
            print(f"  Total images: {len(class_analyses)}")
            print(f"  Problematic images: {problematic_count}")
            print(f"  Average quality score: {avg_quality:.1f}/10")
        
        return all_analyses
    
    def identify_problematic_images(self, analyses, quality_threshold=6):
        """Identify images that need replacement"""
        problematic = []
        good_quality = []
        
        for analysis in analyses:
            if analysis['quality_score'] < quality_threshold or analysis['issue_count'] > 2:
                problematic.append(analysis)
            else:
                good_quality.append(analysis)
        
        return problematic, good_quality
    
    def generate_replacement_report(self, analyses):
        """Generate detailed report of images needing replacement"""
        print("\n📋 REPLACEMENT REPORT")
        print("=" * 60)
        
        problematic, good_quality = self.identify_problematic_images(analyses)
        
        print(f"✅ Good quality images: {len(good_quality)}")
        print(f"❌ Images needing replacement: {len(problematic)}")
        
        # Group by class
        class_problems = {}
        for analysis in problematic:
            class_name = analysis['class']
            if class_name not in class_problems:
                class_problems[class_name] = []
            class_problems[class_name].append(analysis)
        
        replacement_plan = {}
        
        for class_name, problems in class_problems.items():
            print(f"\n🔍 {class_name.upper()} class issues:")
            
            issue_summary = {}
            for problem in problems:
                for issue in problem['issues']:
                    issue_summary[issue] = issue_summary.get(issue, 0) + 1
            
            print(f"  Images to replace: {len(problems)}")
            print(f"  Common issues:")
            for issue, count in sorted(issue_summary.items(), key=lambda x: x[1], reverse=True):
                print(f"    - {issue}: {count} images")
            
            replacement_plan[class_name] = {
                'count_to_replace': len(problems),
                'problematic_files': [p['filename'] for p in problems],
                'common_issues': issue_summary
            }
        
        # Save detailed report
        report_path = os.path.join(config.LOGS_DIR, 'image_quality_report.json')
        report_data = {
            'analysis_date': str(np.datetime64('now')),
            'total_images': len(analyses),
            'good_quality_count': len(good_quality),
            'problematic_count': len(problematic),
            'replacement_plan': replacement_plan,
            'detailed_analyses': analyses
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved: {report_path}")
        
        return replacement_plan, problematic, good_quality

def main():
    """Main analysis function"""
    print("🔍 IMAGE QUALITY ANALYZER")
    print("🎯 Identifying problematic images for replacement")
    
    analyzer = ImageQualityAnalyzer()
    
    # Analyze all images
    analyses = analyzer.analyze_dataset()
    
    if not analyses:
        print("❌ No images found to analyze!")
        return
    
    # Generate replacement report
    replacement_plan, problematic, good_quality = analyzer.generate_replacement_report(analyses)
    
    # Summary
    print(f"\n🎯 SUMMARY")
    print("=" * 60)
    print(f"📊 Dataset Analysis Complete:")
    print(f"  Total images analyzed: {len(analyses)}")
    print(f"  Good quality images: {len(good_quality)}")
    print(f"  Images needing replacement: {len(problematic)}")
    
    total_to_replace = sum(plan['count_to_replace'] for plan in replacement_plan.values())
    print(f"\n📋 Replacement needed by class:")
    for class_name, plan in replacement_plan.items():
        print(f"  {class_name}: {plan['count_to_replace']} images")
    
    print(f"\n💡 Next steps:")
    print(f"  1. Review flagged images manually")
    print(f"  2. Run high-quality image collector")
    print(f"  3. Replace problematic images")
    print(f"  4. Re-validate dataset quality")
    
    return replacement_plan

if __name__ == "__main__":
    replacement_plan = main()
