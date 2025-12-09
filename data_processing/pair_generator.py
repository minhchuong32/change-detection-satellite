"""
Pair Generator - Phần của MINH
Nhiệm vụ: (2) Ghép cặp Before-After (Spring → Winter)
"""

from typing import List, Tuple, Dict
import random
from pathlib import Path


class BeforeAfterPairGenerator:
    """Tạo cặp ảnh Before-After cho Change Detection"""
    
    def __init__(self, dataset: Dict):
        """
        Args:
            dataset: Dictionary từ DatasetLoader
                     dataset[season][s_id][p_id] = path
        """
        self.dataset = dataset
        self.pairs = []
    
    def generate_pairs(self, 
                      before_season: str = 'spring',
                      after_season: str = 'winter',
                      min_time_gap: bool = True) -> List[Tuple[str, str, str, str]]:
        """
        Tạo các cặp ảnh Before-After
        
        Args:
            before_season: Mùa "trước" (mặc định: spring)
            after_season: Mùa "sau" (mặc định: winter)
            min_time_gap: Nếu True, chọn cặp có khoảng thời gian dài nhất
        
        Returns:
            List of tuples: (before_path, after_path, s_id, p_id)
        """
        print(f"\n🔗 Tạo cặp {before_season.upper()} → {after_season.upper()}...")
        
        if before_season not in self.dataset:
            raise ValueError(f"Mùa {before_season} không tồn tại trong dataset!")
        if after_season not in self.dataset:
            raise ValueError(f"Mùa {after_season} không tồn tại trong dataset!")
        
        self.pairs = []
        
        # Lấy danh sách s_id có trong cả 2 mùa
        s_ids_before = set(self.dataset[before_season].keys())
        s_ids_after = set(self.dataset[after_season].keys())
        common_s_ids = s_ids_before & s_ids_after
        
        print(f"📍 Tìm thấy {len(common_s_ids)} vùng chung: {sorted(common_s_ids)}")
        
        for s_id in sorted(common_s_ids):
            # Lấy danh sách p_id có trong cả 2 mùa
            p_ids_before = set(self.dataset[before_season][s_id].keys())
            p_ids_after = set(self.dataset[after_season][s_id].keys())
            common_p_ids = p_ids_before & p_ids_after
            
            for p_id in sorted(common_p_ids):
                before_path = self.dataset[before_season][s_id][p_id]
                after_path = self.dataset[after_season][s_id][p_id]
                
                self.pairs.append((before_path, after_path, s_id, p_id))
        
        print(f"✅ Tạo được {len(self.pairs)} cặp ảnh")
        return self.pairs
    
    def generate_multi_temporal_pairs(self) -> Dict[str, List[Tuple]]:
        """
        Tạo nhiều cặp với khoảng thời gian khác nhau:
        - Spring → Summer (3 tháng)
        - Spring → Fall (6 tháng)
        - Spring → Winter (9 tháng)
        """
        print("\n🕐 Tạo cặp đa thời điểm...")
        
        multi_pairs = {
            'short_term': [],    # Spring → Summer
            'medium_term': [],   # Spring → Fall
            'long_term': []      # Spring → Winter
        }
        
        # Short-term: Spring → Summer
        if 'spring' in self.dataset and 'summer' in self.dataset:
            multi_pairs['short_term'] = self._create_pairs('spring', 'summer')
            print(f"  ✓ Short-term (3 months): {len(multi_pairs['short_term'])} pairs")
        
        # Medium-term: Spring → Fall
        if 'spring' in self.dataset and 'fall' in self.dataset:
            multi_pairs['medium_term'] = self._create_pairs('spring', 'fall')
            print(f"  ✓ Medium-term (6 months): {len(multi_pairs['medium_term'])} pairs")
        
        # Long-term: Spring → Winter
        if 'spring' in self.dataset and 'winter' in self.dataset:
            multi_pairs['long_term'] = self._create_pairs('spring', 'winter')
            print(f"  ✓ Long-term (9 months): {len(multi_pairs['long_term'])} pairs")
        
        return multi_pairs
    
    def _create_pairs(self, season1: str, season2: str) -> List[Tuple]:
        """Helper function để tạo cặp giữa 2 mùa"""
        pairs = []
        s_ids_common = set(self.dataset[season1].keys()) & set(self.dataset[season2].keys())
        
        for s_id in s_ids_common:
            p_ids_common = (set(self.dataset[season1][s_id].keys()) & 
                           set(self.dataset[season2][s_id].keys()))
            
            for p_id in p_ids_common:
                before_path = self.dataset[season1][s_id][p_id]
                after_path = self.dataset[season2][s_id][p_id]
                pairs.append((before_path, after_path, s_id, p_id))
        
        return pairs
    
    def split_pairs(self, 
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15,
                   random_seed: int = 42) -> Dict[str, List[Tuple]]:
        """
        Chia dataset thành train/val/test theo patch ID
        để tránh data leakage
        
        Args:
            train_ratio: Tỷ lệ training set
            val_ratio: Tỷ lệ validation set
            test_ratio: Tỷ lệ test set
            random_seed: Seed cho reproducibility
        
        Returns:
            Dictionary: {'train': [...], 'val': [...], 'test': [...]}
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Tổng các ratio phải bằng 1!"
        
        # Group theo s_id để split đúng cách
        s_id_groups = {}
        for pair in self.pairs:
            s_id = pair[2]
            if s_id not in s_id_groups:
                s_id_groups[s_id] = []
            s_id_groups[s_id].append(pair)
        
        # Shuffle các s_id
        random.seed(random_seed)
        s_ids = list(s_id_groups.keys())
        random.shuffle(s_ids)
        
        # Tính số lượng s_id cho mỗi split
        n_s_ids = len(s_ids)
        n_train = int(n_s_ids * train_ratio)
        n_val = int(n_s_ids * val_ratio)
        
        # Chia s_id
        train_s_ids = s_ids[:n_train]
        val_s_ids = s_ids[n_train:n_train + n_val]
        test_s_ids = s_ids[n_train + n_val:]
        
        # Tạo splits
        splits = {
            'train': [pair for s_id in train_s_ids for pair in s_id_groups[s_id]],
            'val': [pair for s_id in val_s_ids for pair in s_id_groups[s_id]],
            'test': [pair for s_id in test_s_ids for pair in s_id_groups[s_id]]
        }
        
        print(f"\n📊 DATASET SPLIT:")
        print(f"  Train: {len(splits['train'])} pairs ({len(train_s_ids)} regions)")
        print(f"  Val:   {len(splits['val'])} pairs ({len(val_s_ids)} regions)")
        print(f"  Test:  {len(splits['test'])} pairs ({len(test_s_ids)} regions)")
        
        return splits
    
    def save_pairs(self, output_path: str):
        """Lưu danh sách các cặp vào file"""
        with open(output_path, 'w') as f:
            for before, after, s_id, p_id in self.pairs:
                f.write(f"{before},{after},{s_id},{p_id}\n")
        print(f"💾 Đã lưu {len(self.pairs)} cặp vào {output_path}")


# Test script
if __name__ == "__main__":
    # Giả sử đã có dataset từ DatasetLoader
    from dataset_loader import SARDatasetLoader
    
    loader = SARDatasetLoader(data_root="./data")
    dataset = loader.load_dataset()
    
    # Tạo pair generator
    pair_gen = BeforeAfterPairGenerator(dataset)
    
    # Tạo cặp Spring → Winter
    pairs = pair_gen.generate_pairs(before_season='spring', after_season='winter')
    
    # Split dataset
    splits = pair_gen.split_pairs(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # Lưu file
    pair_gen.save_pairs("pairs_spring_winter.txt")
    
    print("\n✅ Pair generation completed!")