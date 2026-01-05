"""
Satellite Image Fetcher for Property Valuation
Downloads satellite images using Google Maps Static API

Based on: https://github.com/D3vutkarsh/satellite-property-valuation
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm


class SatelliteImageFetcher:
    """Fetches satellite images from Google Maps Static API"""
    
    def __init__(self, api_key: str, zoom: int = 18, size: int = 400):
        """
        Initialize the fetcher.
        
        Args:
            api_key: Google Maps API key
            zoom: Zoom level (18 = building level)
            size: Image size in pixels
        """
        self.api_key = api_key
        self.zoom = zoom
        self.size = size
        self.base_url = "https://maps.googleapis.com/maps/api/staticmap"
        self.delay = 0.05  # Rate limiting
    
    def fetch_image(self, lat: float, lon: float) -> Image.Image:
        """Fetch a single satellite image."""
        params = {
            "center": f"{lat},{lon}",
            "zoom": self.zoom,
            "size": f"{self.size}x{self.size}",
            "maptype": "satellite",
            "key": self.api_key,
            "format": "png"
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"Error fetching ({lat}, {lon}): {e}")
            return None
    
    def download_batch(self, df: pd.DataFrame, output_dir: str, 
                       skip_existing: bool = True) -> dict:
        """
        Download images for all properties in dataframe.
        
        Args:
            df: DataFrame with 'id', 'lat', 'long' columns
            output_dir: Directory to save images
            skip_existing: Skip already downloaded images
            
        Returns:
            Stats dictionary with success/failed/skipped counts
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {'total': len(df), 'success': 0, 'failed': 0, 'skipped': 0}
        
        print(f"Downloading {len(df)} images to {output_dir}")
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
            img_path = output_dir / f"{row['id']}.png"
            
            if skip_existing and img_path.exists():
                stats['skipped'] += 1
                continue
            
            image = self.fetch_image(row['lat'], row['long'])
            
            if image:
                image.save(img_path)
                stats['success'] += 1
            else:
                stats['failed'] += 1
            
            time.sleep(self.delay)
        
        print(f"✓ Done: {stats['success']} success, {stats['failed']} failed, {stats['skipped']} skipped")
        return stats


def main():
    """Main function to download all images."""
    # Get API key from environment
    api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
    
    if not api_key:
        print("Error: GOOGLE_MAPS_API_KEY not set!")
        print("Set it with: os.environ['GOOGLE_MAPS_API_KEY'] = 'your_key'")
        return
    
    # Initialize fetcher
    fetcher = SatelliteImageFetcher(api_key=api_key)
    
    # Load datasets
    train_path = Path('data/raw/train.xlsx')
    test_path = Path('data/raw/test.xlsx')
    
    if not train_path.exists():
        print(f"Error: {train_path} not found!")
        print("Please download the dataset first.")
        return
    
    # Download training images
    print("\n=== Downloading Training Images ===")
    train_df = pd.read_excel(train_path)
    fetcher.download_batch(train_df, 'data/images/train')
    
    # Download test images
    if test_path.exists():
        print("\n=== Downloading Test Images ===")
        test_df = pd.read_excel(test_path)
        fetcher.download_batch(test_df, 'data/images/test')
    
    print("\n✓ All downloads complete!")


if __name__ == "__main__":
    main()

