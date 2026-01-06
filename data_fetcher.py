"""
Satellite Image Fetcher for Property Valuation
Downloads satellite images using Google Maps Static API with URL Signing
"""

import os
import time
import requests
import pandas as pd
import base64
import hashlib
import hmac
import urllib.parse
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm


def sign_url(url: str, secret: str) -> str:
    """
    Args:
        url: The full URL to sign (without signature)
        secret: URL signing secret from Google Cloud Console

    Returns:
        Signed URL with signature parameter
    """
    # Decode secret from URL-safe base64
    decoded_key = base64.urlsafe_b64decode(secret)

    # Parse URL
    url_parts = urllib.parse.urlparse(url)
    url_to_sign = url_parts.path + "?" + url_parts.query

    # Create HMAC-SHA1 signature
    signature = hmac.new(
        decoded_key,
        url_to_sign.encode("utf-8"),
        hashlib.sha1
    )

    # Encode signature
    encoded_signature = base64.urlsafe_b64encode(signature.digest()).decode()

    return f"{url}&signature={encoded_signature}"


class SatelliteImageFetcher:
    """Fetches satellite images from Google Maps Static API with URL signing"""

    def __init__(self, api_key: str, signing_secret: str = None,
                 zoom: int = 18, size: int = 400):
        """
        Initialize the fetcher.

        Args:
            api_key: Google Maps API key
            signing_secret: URL signing secret (optional, but recommended to bypass limits)
            zoom: Zoom level (18 = building level)
            size: Image size in pixels
        """
        self.api_key = api_key
        self.signing_secret = signing_secret
        self.zoom = zoom
        self.size = size
        self.base_url = "https://maps.googleapis.com/maps/api/staticmap"
        self.delay = 0.05  # Rate limiting

        if signing_secret:
            print("URL signing enabled")
        else:
            print("URL signing disabled")

    def fetch_image(self, lat: float, lon: float) -> Image.Image:
        """Function to fetch a image."""
        # Build base URL
        url = (
            f"{self.base_url}"
            f"?center={lat},{lon}"
            f"&zoom={self.zoom}"
            f"&size={self.size}x{self.size}"
            f"&maptype=satellite"
            f"&format=png"
            f"&key={self.api_key}"
        )

        if self.signing_secret:
            url = sign_url(url, self.signing_secret)

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # Check if we got an actual image (not an error page)
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type:
                print(
                    f"Warning: Got {content_type} instead of image for ({lat}, {lon})")
                return None

            return Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"Error fetching ({lat}, {lon}): {e}")
            return None
    
    # How many images already downloaded?
    def count_existing_images(self, df: pd.DataFrame, output_dir: Path) -> int:
        count = 0
        for _, row in df.iterrows():
            if (output_dir / f"{row['id']}.png").exists():
                count += 1
        return count

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

        # Count existing images first
        existing_count = self.count_existing_images(df, output_dir)
        remaining = len(df) - existing_count

        print(f"\nStatus: Total images: {len(df)}, Already Downloaded: {existing_count}, Remaining: {remaining}")

        if remaining == 0:
            print("All images already downloaded!")
            return {'total': len(df), 'success': 0, 'failed': 0, 'skipped': existing_count}

        stats = {'total': len(df), 'success': 0, 'failed': 0, 'skipped': 0}

        print(f"\nDownloading {remaining} images to {output_dir}")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
            img_path = output_dir / f"{row['id']}.png"

            # Skip if already exists
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

        print(f"\nDownloaded: {stats['success']}, Failed: {stats['failed']}, Skipped (existing): {stats['skipped']}")

        return stats


def main():
    """Main function to download all images."""
    # Get credentials from environment
    api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
    signing_secret = os.environ.get('GOOGLE_MAPS_SIGNING_SECRET')

    if not api_key:
        print("Set Api key and signing secret first.")
        return

    fetcher = SatelliteImageFetcher(
        api_key=api_key,
        signing_secret=signing_secret
    )

    # Load datasets
    train_path = Path('data/raw/train.xlsx')
    test_path = Path('data/raw/test.xlsx')

    if not train_path.exists():
        print("Please download the dataset first.")
        return

    # Download training images
    print(f"\nTRAINING IMAGES")
    train_df = pd.read_excel(train_path)
    train_stats = fetcher.download_batch(train_df, 'data/images/train')

    # Download test images
    if test_path.exists():
        print(f"\nTEST IMAGES")
        test_df = pd.read_excel(test_path)
        test_stats = fetcher.download_batch(test_df, 'data/images/test')

    print(f"\nDOWNLOAD COMPLETE!")
    train_total = train_stats['success'] + train_stats['skipped']
    test_total = test_stats['success'] + \
        test_stats['skipped'] if test_path.exists() else 0
    print(f"Training images: {train_total}")
    print(f"Test images: {test_total}")
    print(f"Total: {train_total + test_total}")


if __name__ == "__main__":
    main()