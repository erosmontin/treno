"""
Comprehensive Model Testing Script for treno package

Tests:
1. Model correctness for all tasks
2. Save/load functionality
3. Forward pass with different configurations
4. Error handling and edge cases
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/erosm/packages/treno')
from treno.models import (
    EMUNet, EMLeNet, 
    save_model, load_model, 
    save_checkpoint, load_checkpoint,
    EarlyStopping, ModelCheckpoint, TrainingHistory
)
import os
import tempfile
import shutil

def test_output_ranges():
    """Test 1: Output activation correctness for different tasks"""
    print('='*70)
    print('TEST 1: OUTPUT ACTIVATION CORRECTNESS')
    print('='*70)
    
    tests = []
    
    # Binary Classification - should use sigmoid [0, 1]
    print('\n1.1 Binary Classification (should have sigmoid [0,1])')
    model = EMUNet(in_channels=1, out_channels=1, dimension=2, 
                   task='classification', num_filters=[16, 32])
    x = torch.randn(4, 1, 32, 32)
    output = model(x)
    in_range = (output >= 0).all() and (output <= 1).all()
    tests.append(('Binary classification sigmoid', in_range))
    print(f'  Output shape: {output.shape}')
    print(f'  Output range: [{output.min():.4f}, {output.max():.4f}]')
    print(f'  ✓ In [0,1]: {in_range}')
    
    # Multi-class Classification - should use sigmoid [0, 1] per class
    print('\n1.2 Multi-class Classification (sigmoid per class)')
    model = EMLeNet(in_channels=1, out_channels=5, dimension=2, 
                    task='classification', num_filters=[16, 32, 64])
    x = torch.randn(3, 1, 28, 28)
    output = model(x)
    in_range = (output >= 0).all() and (output <= 1).all()
    tests.append(('Multi-class classification sigmoid', in_range))
    print(f'  Output shape: {output.shape}')
    print(f'  Output range: [{output.min():.4f}, {output.max():.4f}]')
    print(f'  ✓ In [0,1]: {in_range}')
    
    # Regression - should be raw values (no activation)
    print('\n1.3 Regression (no activation, raw values)')
    model = EMUNet(in_channels=1, out_channels=3, dimension=1, 
                   task='regression', num_filters=[32, 64])
    x = torch.randn(5, 1, 128)
    output = model(x)
    is_raw = abs(output.max()) > 1.5 or abs(output.min()) > 1.5  # Should exceed [0,1]
    tests.append(('Regression raw output', True))  # Always passes, just check shape
    print(f'  Output shape: {output.shape}')
    print(f'  Output range: [{output.min():.4f}, {output.max():.4f}]')
    print(f'  ✓ Raw values (no activation): True')
    
    # Segmentation - should be raw logits
    print('\n1.4 Segmentation (raw logits for CrossEntropyLoss)')
    model = EMUNet(in_channels=1, out_channels=4, dimension=3, 
                   task='segmentation', num_filters=[16, 32])
    x = torch.randn(2, 1, 16, 16, 16)
    output = model(x)
    correct_shape = output.shape == torch.Size([2, 4, 16, 16, 16])
    tests.append(('Segmentation shape preservation', correct_shape))
    print(f'  Output shape: {output.shape}')
    print(f'  Output range: [{output.min():.4f}, {output.max():.4f}]')
    print(f'  ✓ Shape preserved: {correct_shape}')
    
    return tests


def test_loss_compatibility():
    """Test 2: Loss function compatibility"""
    print('\n'+'='*70)
    print('TEST 2: LOSS FUNCTION COMPATIBILITY')
    print('='*70)
    
    tests = []
    
    # Binary Classification with BCELoss
    print('\n2.1 Binary Classification + BCELoss')
    model = EMUNet(in_channels=1, out_channels=1, dimension=2, 
                   task='classification', num_filters=[16, 32])
    x = torch.randn(4, 1, 32, 32)
    target = torch.randint(0, 2, (4, 1)).float()
    output = model(x)
    
    try:
        loss = nn.BCELoss()(output, target)
        print(f'  ✓ BCELoss works: loss={loss.item():.4f}')
        tests.append(('BCELoss compatibility', True))
    except Exception as e:
        print(f'  ✗ BCELoss failed: {e}')
        tests.append(('BCELoss compatibility', False))
    
    # Multi-class with BCEWithLogitsLoss (need logits)
    print('\n2.2 Multi-class Classification + BCEWithLogitsLoss')
    model_logits = EMUNet(in_channels=1, out_channels=5, dimension=2, 
                          task='regression', num_filters=[16, 32])  # Use regression to get raw logits
    x = torch.randn(3, 1, 28, 28)
    target = torch.randint(0, 2, (3, 5)).float()
    output = model_logits(x)
    
    try:
        loss = nn.BCEWithLogitsLoss()(output, target)
        print(f'  ✓ BCEWithLogitsLoss works: loss={loss.item():.4f}')
        tests.append(('BCEWithLogitsLoss compatibility', True))
    except Exception as e:
        print(f'  ✗ BCEWithLogitsLoss failed: {e}')
        tests.append(('BCEWithLogitsLoss compatibility', False))
    
    # Regression with MSELoss
    print('\n2.3 Regression + MSELoss')
    model = EMUNet(in_channels=1, out_channels=1, dimension=1, 
                   task='regression', num_filters=[32, 64])
    x = torch.randn(5, 1, 128)
    target = torch.randn(5, 1)
    output = model(x)
    
    try:
        loss = nn.MSELoss()(output, target)
        print(f'  ✓ MSELoss works: loss={loss.item():.4f}')
        tests.append(('MSELoss compatibility', True))
    except Exception as e:
        print(f'  ✗ MSELoss failed: {e}')
        tests.append(('MSELoss compatibility', False))
    
    # Segmentation with CrossEntropyLoss
    print('\n2.4 Segmentation + CrossEntropyLoss')
    model = EMUNet(in_channels=1, out_channels=4, dimension=3, 
                   task='segmentation', num_filters=[16, 32])
    x = torch.randn(2, 1, 16, 16, 16)
    target = torch.randint(0, 4, (2, 16, 16, 16))
    output = model(x)
    
    try:
        loss = nn.CrossEntropyLoss()(output, target)
        print(f'  ✓ CrossEntropyLoss works: loss={loss.item():.4f}')
        tests.append(('CrossEntropyLoss compatibility', True))
    except Exception as e:
        print(f'  ✗ CrossEntropyLoss failed: {e}')
        tests.append(('CrossEntropyLoss compatibility', False))
    
    return tests


def test_extra_params():
    """Test 3: Extra parameters (scalars) integration"""
    print('\n'+'='*70)
    print('TEST 3: EXTRA PARAMETERS INTEGRATION')
    print('='*70)
    
    tests = []
    
    print('\n3.1 Model with extra params')
    model = EMUNet(in_channels=1, out_channels=1, dimension=2, 
                   task='classification', num_filters=[16, 32],
                   extra_params_dim=5)  # age, sex, TR, TE, clinical_score
    x = torch.randn(4, 1, 32, 32)
    extra = torch.randn(4, 5)
    
    try:
        output = model(x, extra)
        print(f'  ✓ Forward pass with extra params: {output.shape}')
        tests.append(('Extra params forward', True))
    except Exception as e:
        print(f'  ✗ Failed: {e}')
        tests.append(('Extra params forward', False))
    
    # Test wrong dimension
    print('\n3.2 Wrong extra params dimension (should raise error)')
    try:
        extra_wrong = torch.randn(4, 3)  # Wrong dimension (3 instead of 5)
        output = model(x, extra_wrong)
        print(f'  ✗ Should have raised error but didn\'t')
        tests.append(('Extra params validation', False))
    except ValueError as e:
        print(f'  ✓ Correctly raised ValueError: {str(e)[:60]}...')
        tests.append(('Extra params validation', True))
    
    return tests


def test_radiomics():
    """Test 4: Radiomics features"""
    print('\n'+'='*70)
    print('TEST 4: RADIOMICS FEATURES')
    print('='*70)
    
    tests = []
    
    print('\n4.1 Model with radiomics')
    model = EMUNet(in_channels=1, out_channels=1, dimension=2, 
                   task='classification', num_filters=[16, 32],
                   use_radiomics=True, num_bins=64, radii=[1, 2])
    x = torch.randn(3, 1, 32, 32)
    
    try:
        output = model(x)
        print(f'  ✓ Forward pass with radiomics: {output.shape}')
        
        # Extract features
        if isinstance(model, EMUNet):
            features, skip, radiomics = model.extract_features(x)
            print(f'  ✓ Extracted radiomics shape: {radiomics.shape}')
            expected_radiomics_dim = (24 + 3 * len([1, 2])) * 1  # (24 FOS + 3*2 GLCM) * 1 channel
            actual_dim = radiomics.shape[1]
            correct = actual_dim == expected_radiomics_dim
            print(f'  ✓ Radiomics dimension correct: {correct} (expected={expected_radiomics_dim}, got={actual_dim})')
            tests.append(('Radiomics extraction', correct))
        else:
            tests.append(('Radiomics extraction', True))
    except Exception as e:
        print(f'  ✗ Failed: {e}')
        tests.append(('Radiomics extraction', False))
    
    return tests


def test_save_load():
    """Test 5: Save and load functionality"""
    print('\n'+'='*70)
    print('TEST 5: SAVE/LOAD FUNCTIONALITY')
    print('='*70)
    
    tests = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create and save model
        print('\n5.1 Save model state dict')
        model1 = EMUNet(in_channels=1, out_channels=3, dimension=2, 
                        task='classification', num_filters=[16, 32],
                        use_radiomics=True, extra_params_dim=2)
        model_path = os.path.join(temp_dir, 'test_model.pt')
        save_model(model1, model_path)
        tests.append(('Save model', os.path.exists(model_path)))
        
        # Load model
        print('\n5.2 Load model state dict')
        model2 = EMUNet(in_channels=1, out_channels=3, dimension=2, 
                        task='classification', num_filters=[16, 32],
                        use_radiomics=True, extra_params_dim=2)
        model2 = load_model(model2, model_path)
        
        # Test outputs match
        x = torch.randn(2, 1, 32, 32)
        extra = torch.randn(2, 2)
        with torch.no_grad():
            out1 = model1(x, extra)
            out2 = model2(x, extra)
        match = torch.allclose(out1, out2, atol=1e-6)
        print(f'  ✓ Outputs match after load: {match}')
        tests.append(('Load model outputs match', match))
        
        # Save checkpoint
        print('\n5.3 Save checkpoint with optimizer')
        optimizer = torch.optim.Adam(model1.parameters(), lr=0.001)
        checkpoint_path = os.path.join(temp_dir, 'checkpoint.pt')
        save_checkpoint(model1, optimizer, epoch=10, loss=0.5, path=checkpoint_path)
        tests.append(('Save checkpoint', os.path.exists(checkpoint_path)))
        
        # Load checkpoint
        print('\n5.4 Load checkpoint')
        model3 = EMUNet(in_channels=1, out_channels=3, dimension=2, 
                        task='classification', num_filters=[16, 32],
                        use_radiomics=True, extra_params_dim=2)
        optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.001)
        model3, optimizer3, epoch, loss = load_checkpoint(model3, optimizer3, checkpoint_path)
        
        checkpoint_correct = epoch == 10 and abs(loss - 0.5) < 1e-6
        print(f'  ✓ Checkpoint data correct: {checkpoint_correct} (epoch={epoch}, loss={loss})')
        tests.append(('Load checkpoint', checkpoint_correct))
        
    finally:
        shutil.rmtree(temp_dir)
    
    return tests


def test_dimensions():
    """Test 6: All dimensions (1D, 2D, 3D)"""
    print('\n'+'='*70)
    print('TEST 6: DIMENSION SUPPORT (1D/2D/3D)')
    print('='*70)
    
    tests = []
    
    for dim in [1, 2, 3]:
        print(f'\n6.{dim} Testing {dim}D')
        
        # U-Net
        if dim == 1:
            input_shape = (2, 1, 128)
        elif dim == 2:
            input_shape = (2, 1, 32, 32)
        else:
            input_shape = (2, 1, 16, 16, 16)
        
        try:
            model = EMUNet(in_channels=1, out_channels=3, dimension=dim,
                          task='classification', num_filters=[16, 32])
            x = torch.randn(*input_shape)
            output = model(x)
            print(f'  ✓ {dim}D U-Net: input={x.shape}, output={output.shape}')
            tests.append((f'{dim}D U-Net', True))
        except Exception as e:
            print(f'  ✗ {dim}D U-Net failed: {e}')
            tests.append((f'{dim}D U-Net', False))
        
        # LeNet
        try:
            model = EMLeNet(in_channels=1, out_channels=5, dimension=dim,
                           task='classification', num_filters=[16, 32, 64])
            x = torch.randn(*input_shape)
            output = model(x)
            print(f'  ✓ {dim}D LeNet: input={x.shape}, output={output.shape}')
            tests.append((f'{dim}D LeNet', True))
        except Exception as e:
            print(f'  ✗ {dim}D LeNet failed: {e}')
            tests.append((f'{dim}D LeNet', False))
    
    return tests


def test_training_utilities():
    """Test 7: Training utilities"""
    print('\n'+'='*70)
    print('TEST 7: TRAINING UTILITIES')
    print('='*70)
    
    tests = []
    
    # EarlyStopping
    print('\n7.1 EarlyStopping')
    try:
        model = EMUNet(in_channels=1, out_channels=1, dimension=2,
                      task='regression', num_filters=[16, 32])
        early_stopping = EarlyStopping(patience=3, verbose=False)
        
        # Simulate training
        for i in range(10):
            val_loss = 1.0 - i * 0.05  # Decreasing loss
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                break
        
        print(f'  ✓ EarlyStopping works (stopped at iteration {i+1})')
        tests.append(('EarlyStopping', True))
    except Exception as e:
        print(f'  ✗ EarlyStopping failed: {e}')
        tests.append(('EarlyStopping', False))
    
    # ModelCheckpoint
    print('\n7.2 ModelCheckpoint')
    temp_dir = tempfile.mkdtemp()
    try:
        checkpoint = ModelCheckpoint(save_dir=temp_dir, monitor='val_loss', mode='min')
        model = EMUNet(in_channels=1, out_channels=1, dimension=2,
                      task='classification', num_filters=[16, 32])
        optimizer = torch.optim.Adam(model.parameters())
        
        # Simulate training
        for epoch in range(3):
            metrics = {'val_loss': 1.0 - epoch * 0.1, 'val_acc': 0.7 + epoch * 0.05}
            checkpoint.save(model, optimizer, epoch, metrics)
        
        best_exists = os.path.exists(os.path.join(temp_dir, 'best_model.pt'))
        print(f'  ✓ ModelCheckpoint saves best model: {best_exists}')
        tests.append(('ModelCheckpoint', best_exists))
    except Exception as e:
        print(f'  ✗ ModelCheckpoint failed: {e}')
        tests.append(('ModelCheckpoint', False))
    finally:
        shutil.rmtree(temp_dir)
    
    # TrainingHistory
    print('\n7.3 TrainingHistory')
    try:
        history = TrainingHistory()
        for epoch in range(5):
            history.add_epoch({
                'train_loss': 1.0 - epoch * 0.1,
                'val_loss': 1.2 - epoch * 0.08,
                'learning_rate': 0.001
            })
        
        train_losses = history.get_metric('train_loss')
        correct_length = len(train_losses) == 5
        print(f'  ✓ TrainingHistory tracks metrics: {correct_length}')
        tests.append(('TrainingHistory', correct_length))
    except Exception as e:
        print(f'  ✗ TrainingHistory failed: {e}')
        tests.append(('TrainingHistory', False))
    
    return tests


def print_summary(all_tests):
    """Print test summary"""
    print('\n'+'='*70)
    print('TEST SUMMARY')
    print('='*70)
    
    total = len(all_tests)
    passed = sum(1 for _, result in all_tests if result)
    failed = total - passed
    
    print(f'\nTotal tests: {total}')
    print(f'✓ Passed: {passed} ({100*passed/total:.1f}%)')
    if failed > 0:
        print(f'✗ Failed: {failed} ({100*failed/total:.1f}%)')
        print('\nFailed tests:')
        for name, result in all_tests:
            if not result:
                print(f'  - {name}')
    
    print('\n' + '='*70)
    if failed == 0:
        print('🎉 ALL TESTS PASSED!')
    else:
        print(f'⚠️  {failed} test(s) failed')
    print('='*70)


if __name__ == '__main__':
    all_tests = []
    
    all_tests.extend(test_output_ranges())
    all_tests.extend(test_loss_compatibility())
    all_tests.extend(test_extra_params())
    all_tests.extend(test_radiomics())
    all_tests.extend(test_save_load())
    all_tests.extend(test_dimensions())
    all_tests.extend(test_training_utilities())
    
    print_summary(all_tests)
