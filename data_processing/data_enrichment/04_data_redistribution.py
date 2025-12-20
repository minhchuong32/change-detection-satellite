"""
Phân phối lại toàn bộ dữ liệu đã gộp vào các tập Train, Val, Test.
Sử dụng Stratified Split để đảm bảo tỷ lệ các nhóm ảnh (Low/Med/High change) 
đồng đều giữa các tập dữ liệu.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_metadata_df(root_dir):
    # Quét toàn bộ dataset và tạo dataframe lưu thông tin: filename, ratio, category
    pass

def perform_stratified_split(df, split_ratio=(0.7, 0.2, 0.1)):
    # Chia dữ liệu dựa trên nhãn phân loại tỷ lệ thay đổi
    pass