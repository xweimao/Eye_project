"""
Data Collection module for Eye State Classification Project
"""

from .web_scraper import EyeImageScraper
from .data_validator import ImageValidator

__all__ = ['EyeImageScraper', 'ImageValidator']
