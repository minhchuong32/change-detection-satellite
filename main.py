"""
Main Pipeline - Chạy toàn bộ quy trình
"""

import yaml
import argparse
from pathlib import Path


def load_config(config_path='configs/config.yaml'):
    """Load configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_preprocessing(config):
    """
    Bước 1-4: Tiền xử lý dữ liệu
    """
    from data_processing.dataset_loader import SARDatasetLoader
    from data_processing.pair_generator import BeforeAfterPairGenerator
    
    print("=" * 60)
    print("BƯỚC 1-4: TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 60)
    
    # 1. Load dataset
    print("\n(1) Loading dataset...")
    loader = SARDatasetLoader(data_root=config['data']['root_dir'])
    dataset = loader.load_dataset()
    loader.save_metadata("dataset_metadata.json")
    
    # 2. Generate pairs
    print("\n(2) Generating Before-After pairs...")
    pair_gen = BeforeAfterPairGenerator(dataset)
    pairs = pair_gen.generate_pairs(
        before_season=config['data']['before_season'],
        after_season=config['data']['after_season']
    )
    pair_gen.save_pairs("pairs.txt")
    
    # Split dataset
    splits = pair_gen.split_pairs(
        train_ratio=config['split']['train_ratio'],
        val_ratio=config['split']['val_ratio'],
        test_ratio=config['split']['test_ratio'],
        random_seed=config['split']['random_seed']
    )
    
    print("\n✅ Preprocessing completed!")
    return splits


def run_training(config, splits):
    """
    Bước 5-6: Train model
    """
    import torch
    from torch.utils.data import DataLoader
    from models.unet import UNet
    from training.train import ChangeDetectionDataset, train_model
    from training.train import get_train_transform, get_val_transform
    
    print("\n" + "=" * 60)
    print("BƯỚC 5-6: TRAINING MODEL")
    print("=" * 60)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = ChangeDetectionDataset(
        splits['train'],
        transform=get_train_transform()
    )
    val_dataset = ChangeDetectionDataset(
        splits['val'],
        transform=get_val_transform()
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    print("\nInitializing U-Net model...")
    model = UNet(
        n_channels=config['model']['n_channels'],
        n_classes=config['model']['n_classes'],
        bilinear=config['model']['bilinear']
    )
    
    # Train
    print("\nStarting training...")
    history = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=config['training']['num_epochs'],
        learning_rate=config['training']['learning_rate'],
        device=config['training']['device']
    )
    
    print("\n✅ Training completed!")
    return model, history


def run_inference(config, model, splits):
    """
    Bước 7: Inference
    """
    from training.inference import BatchInference
    
    print("\n" + "=" * 60)
    print("BƯỚC 7: INFERENCE")
    print("=" * 60)
    
    # Create inference engine
    batch_inference = BatchInference(
        model,
        device=config['training']['device']
    )
    
    # Predict on test set
    test_pairs = [(p[0], p[1]) for p in splits['test']]
    results = batch_inference.predict_batch(
        test_pairs[:10],  # Test với 10 ảnh đầu
        output_dir=config['paths']['predictions_dir']
    )
    
    print(f"\n✅ Inference completed! Results saved to {config['paths']['predictions_dir']}")
    return results


def run_web_app(config):
    """
    Bước 9: Launch web app
    """
    from app.gradio_app import create_demo
    
    print("\n" + "=" * 60)
    print("BƯỚC 9: LAUNCHING WEB APP")
    print("=" * 60)
    
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


def main():
    parser = argparse.ArgumentParser(description='SAR Change Detection Pipeline')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['preprocess', 'train', 'inference', 'app', 'all'],
                       help='Pipeline mode')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create directories
    Path(config['paths']['checkpoint_dir']).mkdir(exist_ok=True)
    Path(config['paths']['predictions_dir']).mkdir(exist_ok=True)
    Path(config['paths']['logs_dir']).mkdir(exist_ok=True)
    
    # Run pipeline
    if args.mode in ['preprocess', 'all']:
        splits = run_preprocessing(config)
    
    if args.mode in ['train', 'all']:
        if 'splits' not in locals():
            print("⚠️ Need to run preprocessing first!")
            return
        model, history = run_training(config, splits)
    
    if args.mode in ['inference', 'all']:
        if 'model' not in locals():
            print("⚠️ Need to run training first!")
            return
        results = run_inference(config, model, splits)
    
    if args.mode in ['app', 'all']:
        run_web_app(config)


if __name__ == "__main__":
    main()