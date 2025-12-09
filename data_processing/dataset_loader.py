"""
Dataset Loader - Phần của THÙY
Nhiệm vụ: (1) Đọc dataset 4 mùa và tổ chức thành dictionary
"""

import os
from pathlib import Path
from typing import Dict, List
import numpy as np
from PIL import Image
import json


class SARDatasetLoader:
    """
    Load SAR dataset với cấu trúc:
    data/
    ├── spring/
    │   ├── s_01/
    │   │   ├── p_001.tif
    │   │   └── p_002.tif
    │   └── s_02/
    ├── summer/
    ├── fall/
    └── winter/
    """
    
    def __init__(self, data_root: str):
        """
        Args:
            data_root: Đường dẫn đến thư mục gốc chứa 4 mùa
        """
        self.data_root = Path(data_root)
        self.seasons = ['spring', 'summer', 'fall', 'winter']
        self.dataset = {}
        
    def load_dataset(self) -> Dict:
        """
        Đọc toàn bộ dataset và tổ chức theo cấu trúc:
        dataset[season][s_id][p_id] = path_to_image
        
        Returns:
            Dictionary chứa đường dẫn đến tất cả ảnh
        """
        print("🔍 Bắt đầu load dataset...")
        
        for season in self.seasons:
            season_path = self.data_root / season
            
            if not season_path.exists():
                print(f"⚠️ Không tìm thấy thư mục: {season_path}")
                continue
                
            self.dataset[season] = {}
            
            # Duyệt qua các thư mục s_id (vùng khảo sát)
            for s_dir in sorted(season_path.iterdir()):
                if not s_dir.is_dir():
                    continue
                    
                s_id = s_dir.name  # e.g., 's_01'
                self.dataset[season][s_id] = {}
                
                # Duyệt qua các patch
                for img_file in sorted(s_dir.glob('*.tif')):
                    p_id = img_file.stem  # e.g., 'p_001'
                    self.dataset[season][s_id][p_id] = str(img_file)
                    
                print(f"✓ {season}/{s_id}: {len(self.dataset[season][s_id])} patches")
        
        self._print_statistics()
        return self.dataset
    
    def _print_statistics(self):
        """In thống kê dataset"""
        print("\n📊 THỐNG KÊ DATASET:")
        print("-" * 50)
        
        total_images = 0
        for season in self.seasons:
            if season not in self.dataset:
                continue
                
            season_count = sum(
                len(patches) 
                for patches in self.dataset[season].values()
            )
            total_images += season_count
            print(f"{season:10s}: {season_count:4d} images")
        
        print("-" * 50)
        print(f"{'TOTAL':10s}: {total_images:4d} images\n")
    
    def get_image_path(self, season: str, s_id: str, p_id: str) -> str:
        """Lấy đường dẫn ảnh cụ thể"""
        return self.dataset.get(season, {}).get(s_id, {}).get(p_id)
    
    def get_all_s_ids(self, season: str) -> List[str]:
        """Lấy tất cả các s_id trong một mùa"""
        return list(self.dataset.get(season, {}).keys())
    
    def get_all_p_ids(self, season: str, s_id: str) -> List[str]:
        """Lấy tất cả các p_id trong một s_id"""
        return list(self.dataset.get(season, {}).get(s_id, {}).keys())
    
    def save_metadata(self, output_path: str):
        """Lưu metadata của dataset"""
        metadata = {
            'seasons': self.seasons,
            'structure': {}
        }
        
        for season in self.seasons:
            if season not in self.dataset:
                continue
            metadata['structure'][season] = {
                s_id: list(patches.keys())
                for s_id, patches in self.dataset[season].items()
            }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Đã lưu metadata tại: {output_path}")


# Test function
if __name__ == "__main__":
    # Ví dụ sử dụng
    loader = SARDatasetLoader(data_root="./data")
    dataset = loader.load_dataset()
    
    # Lưu metadata
    loader.save_metadata("dataset_metadata.json")
    
    # Ví dụ truy cập
    spring_s01_p001 = loader.get_image_path('spring', 's_01', 'p_001')
    print(f"\n📍 Ví dụ đường dẫn: {spring_s01_p001}")