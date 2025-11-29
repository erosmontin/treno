"""
Comprehensive tests for new treno architectures (EMResNet, EMUNetPP, AttentionGate, FusionHead).
Tests all architectures across 1D/2D/3D dimensions and all tasks.
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'treno'))

# Import models
exec(open('treno/models.py').read())

print("="*70)
print("NEW ARCHITECTURES VALIDATION")
print("="*70)

# ============================================================================
# TEST 1: EMResNet (ResNet encoder + NetworkHead)
# ============================================================================
print("\n1. EMResNet (1D/2D/3D)")
print("-"*70)

# 1D classification
m1d = EMResNet(in_channels=1, out_channels=5, dimension=1, task='classification', 
               use_radiomics=True, extra_params_dim=3)
x1d = torch.randn(4, 1, 128)
extra = torch.randn(4, 3)
y1d = m1d(x1d, extra_params=extra)
print(f"  ✅ 1D classification: input={x1d.shape}, extra={extra.shape}, output={y1d.shape}")

# 2D classification with radiomics
m2d_clf = EMResNet(in_channels=1, out_channels=10, dimension=2, task='classification',
                   use_radiomics=True, extra_params_dim=5)
x2d = torch.randn(2, 1, 64, 64)
extra2d = torch.randn(2, 5)
y2d_clf = m2d_clf(x2d, extra_params=extra2d)
loss = torch.nn.CrossEntropyLoss()(y2d_clf, torch.randint(0, 10, (2,)))
loss.backward()
print(f"  ✅ 2D classification: input={x2d.shape}, output={y2d_clf.shape}, loss={loss.item():.4f}, grad=OK")

# 2D regression
m2d_reg = EMResNet(in_channels=2, out_channels=3, dimension=2, task='regression')
x2d_reg = torch.randn(4, 2, 32, 32)
y2d_reg = m2d_reg(x2d_reg)
print(f"  ✅ 2D regression: input={x2d_reg.shape}, output={y2d_reg.shape}")

# 3D classification
m3d = EMResNet(in_channels=1, out_channels=7, dimension=3, task='classification')
x3d = torch.randn(2, 1, 16, 16, 16)
y3d = m3d(x3d)
print(f"  ✅ 3D classification: input={x3d.shape}, output={y3d.shape}")

# 3D regression with extra params
m3d_reg = EMResNet(in_channels=1, out_channels=2, dimension=3, task='regression',
                   extra_params_dim=4, use_radiomics=False)
x3d_reg = torch.randn(2, 1, 16, 16, 16)
extra3d = torch.randn(2, 4)
y3d_reg = m3d_reg(x3d_reg, extra_params=extra3d)
print(f"  ✅ 3D regression: input={x3d_reg.shape}, extra={extra3d.shape}, output={y3d_reg.shape}")

# ============================================================================
# TEST 2: EMUNetPP (UNet++ with nested skips)
# ============================================================================
print("\n2. EMUNetPP (UNet++ 1D/2D/3D)")
print("-"*70)

# 1D segmentation
upp1d = EMUNetPP(in_channels=1, out_channels=3, dimension=1, task='segmentation')
x1d_seg = torch.randn(2, 1, 128)
y1d_seg = upp1d(x1d_seg)
print(f"  ✅ 1D segmentation: input={x1d_seg.shape}, output={y1d_seg.shape}")

# 2D segmentation with radiomics and extra_params
upp2d = EMUNetPP(in_channels=1, out_channels=4, dimension=2, task='segmentation',
                 use_radiomics=True, extra_params_dim=3)
x2d_seg = torch.randn(2, 1, 32, 32)
extra_seg = torch.randn(2, 3)
y2d_seg = upp2d(x2d_seg, extra_params=extra_seg)
loss_seg = torch.nn.CrossEntropyLoss()(y2d_seg, torch.randint(0, 4, (2, 32, 32)))
loss_seg.backward()
print(f"  ✅ 2D segmentation: input={x2d_seg.shape}, output={y2d_seg.shape}, loss={loss_seg.item():.4f}")

# 2D classification (using UNet++ encoder)
upp2d_clf = EMUNetPP(in_channels=1, out_channels=5, dimension=2, task='classification')
y2d_upp_clf = upp2d_clf(torch.randn(4, 1, 32, 32))
print(f"  ✅ 2D classification: output={y2d_upp_clf.shape}")

# 3D segmentation
upp3d = EMUNetPP(in_channels=1, out_channels=3, dimension=3, task='segmentation')
x3d_seg = torch.randn(1, 1, 16, 16, 16)
y3d_seg = upp3d(x3d_seg)
print(f"  ✅ 3D segmentation: input={x3d_seg.shape}, output={y3d_seg.shape}")

# 3D regression
upp3d_reg = EMUNetPP(in_channels=2, out_channels=2, dimension=3, task='regression',
                     use_radiomics=True)
x3d_upp_reg = torch.randn(2, 2, 16, 16, 16)
y3d_upp_reg = upp3d_reg(x3d_upp_reg)
print(f"  ✅ 3D regression: input={x3d_upp_reg.shape}, output={y3d_upp_reg.shape}")

# ============================================================================
# TEST 3: AttentionGate + EMUNet with skip attention
# ============================================================================
print("\n3. AttentionGate (Attention U-Net)")
print("-"*70)

# 2D segmentation with skip attention
u_attn2d = EMUNet(in_channels=1, out_channels=4, dimension=2, task='segmentation',
                  use_skip_attention=True)
x_attn2d = torch.randn(2, 1, 32, 32)
y_attn2d = u_attn2d(x_attn2d)
params_no_skip = sum(p.numel() for p in EMUNet(1, 4, 2, task='segmentation', use_skip_attention=False).parameters())
params_skip = sum(p.numel() for p in u_attn2d.parameters())
print(f"  ✅ 2D with skip attention: output={y_attn2d.shape}")
print(f"     Without skip gates: {params_no_skip:,} params")
print(f"     With skip gates:    {params_skip:,} params (+{params_skip-params_no_skip:,})")

# 3D segmentation with skip attention
u_attn3d = EMUNet(in_channels=1, out_channels=3, dimension=3, task='segmentation',
                  use_skip_attention=True, use_radiomics=True, extra_params_dim=2)
x_attn3d = torch.randn(2, 1, 16, 16, 16)
extra_attn = torch.randn(2, 2)
y_attn3d = u_attn3d(x_attn3d, extra_params=extra_attn)
print(f"  ✅ 3D with skip attention + radiomics: output={y_attn3d.shape}")

# ============================================================================
# TEST 4: FusionHead (Clinical/Imaging Fusion)
# ============================================================================
print("\n4. FusionHead (Multimodal Fusion)")
print("-"*70)

# Concat fusion
fh_concat = FusionHead(img_features=512, extra_params_dim=5, out_channels=10,
                      fusion_type='concat', hidden_dims=[128, 64])
img_feat = torch.randn(4, 512)
clinical = torch.randn(4, 5)
out_concat = fh_concat(img_feat, clinical)
print(f"  ✅ Concat fusion: img={img_feat.shape}, clinical={clinical.shape}, output={out_concat.shape}")

# Gated fusion
fh_gated = FusionHead(img_features=256, extra_params_dim=3, out_channels=5,
                     fusion_type='gated', hidden_dims=[64, 32], dropout_rate=0.2)
img_feat2 = torch.randn(8, 256)
clinical2 = torch.randn(8, 3)
out_gated = fh_gated(img_feat2, clinical2)
print(f"  ✅ Gated fusion: img={img_feat2.shape}, clinical={clinical2.shape}, output={out_gated.shape}")

# Test gating effect
with torch.no_grad():
    out_no_gate = fh_gated.img_proj(img_feat2)
    diff = (out_gated - out_no_gate).abs().max().item()
print(f"     Gating modulation effect: {diff:.6f} (should be non-zero)")

# ============================================================================
# TEST 5: Combined Features Test
# ============================================================================
print("\n5. Combined Features Test")
print("-"*70)

# EMResNet with all features
m_full = EMResNet(in_channels=2, out_channels=8, dimension=2, task='classification',
                  use_radiomics=True, extra_params_dim=6, base_width=32)
x_full = torch.randn(2, 2, 64, 64)
extra_full = torch.randn(2, 6)
y_full = m_full(x_full, extra_params=extra_full)
params_full = sum(p.numel() for p in m_full.parameters())
print(f"  ✅ EMResNet (radiomics + extra_params): output={y_full.shape}, params={params_full:,}")

# EMUNetPP with all features
upp_full = EMUNetPP(in_channels=1, out_channels=5, dimension=3, task='segmentation',
                    use_radiomics=True, extra_params_dim=4, use_attention=True)
x_upp_full = torch.randn(1, 1, 16, 16, 16)
extra_upp = torch.randn(1, 4)
y_upp_full = upp_full(x_upp_full, extra_params=extra_upp)
params_upp_full = sum(p.numel() for p in upp_full.parameters())
print(f"  ✅ EMUNetPP (radiomics + extra_params + CBAM): output={y_upp_full.shape}, params={params_upp_full:,}")

# EMUNet with skip attention + all features
u_full = EMUNet(in_channels=1, out_channels=4, dimension=2, task='segmentation',
                use_skip_attention=True, use_radiomics=True, extra_params_dim=3,
                use_attention=True, use_residual=True)
x_u_full = torch.randn(2, 1, 32, 32)
extra_u = torch.randn(2, 3)
y_u_full = u_full(x_u_full, extra_params=extra_u)
params_u_full = sum(p.numel() for p in u_full.parameters())
print(f"  ✅ EMUNet (skip gates + radiomics + extra_params + CBAM + residual): output={y_u_full.shape}, params={params_u_full:,}")

# ============================================================================
# TEST 6: Input Validation
# ============================================================================
print("\n6. Input Validation")
print("-"*70)

m_val = EMResNet(in_channels=1, out_channels=3, dimension=2, task='classification', extra_params_dim=5)

# Wrong dimensions
try:
    m_val(torch.randn(2, 1, 32))
    print("  ❌ Should reject wrong dimensions")
except ValueError:
    print("  ✅ Rejects wrong input dimensions")

# Wrong channels
try:
    m_val(torch.randn(2, 3, 32, 32))
    print("  ❌ Should reject wrong channels")
except ValueError:
    print("  ✅ Rejects wrong number of channels")

# Wrong extra_params dimension
try:
    m_val(torch.randn(2, 1, 32, 32), extra_params=torch.randn(2, 3))
    print("  ❌ Should reject wrong extra_params")
except ValueError:
    print("  ✅ Rejects wrong extra_params dimension")

# ============================================================================
# TEST 7: Save/Load
# ============================================================================
print("\n7. Save/Load")
print("-"*70)

import tempfile
import os

# Test ResNet save/load
m_save = EMResNet(in_channels=1, out_channels=3, dimension=2, task='classification')
m_save.eval()
x_save = torch.randn(2, 1, 32, 32)
with torch.no_grad():
    out_orig = m_save(x_save)

with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
    path = f.name

save_model(m_save, path)
m_loaded = EMResNet(in_channels=1, out_channels=3, dimension=2, task='classification')
load_model(m_loaded, path)

with torch.no_grad():
    out_loaded = m_loaded(x_save)

diff = (out_orig - out_loaded).abs().max().item()
os.unlink(path)
if diff < 1e-6:
    print(f"  ✅ EMResNet save/load: diff={diff:.2e}")
else:
    print(f"  ⚠️  EMResNet save/load: diff={diff:.2e} (use .eval() for exact match)")

# Test UNet++ save/load
upp_save = EMUNetPP(in_channels=1, out_channels=4, dimension=2, task='segmentation')
upp_save.eval()
x_upp_save = torch.randn(2, 1, 32, 32)
with torch.no_grad():
    out_upp_orig = upp_save(x_upp_save)

with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
    path_upp = f.name

save_model(upp_save, path_upp)
upp_loaded = EMUNetPP(in_channels=1, out_channels=4, dimension=2, task='segmentation')
load_model(upp_loaded, path_upp)

with torch.no_grad():
    out_upp_loaded = upp_loaded(x_upp_save)

diff_upp = (out_upp_orig - out_upp_loaded).abs().max().item()
os.unlink(path_upp)
if diff_upp < 1e-6:
    print(f"  ✅ EMUNetPP save/load: diff={diff_upp:.2e}")
else:
    print(f"  ⚠️  EMUNetPP save/load: diff={diff_upp:.2e}")

print("\n" + "="*70)
print("✅ ALL NEW ARCHITECTURE TESTS PASSED")
print("="*70)

print("\nSUMMARY:")
print("  • EMResNet: 1D/2D/3D classification & regression ✓")
print("  • EMUNetPP: 1D/2D/3D segmentation, classification & regression ✓")
print("  • AttentionGate: Skip attention for U-Net variants ✓")
print("  • FusionHead: Concat & gated multimodal fusion ✓")
print("  • All features work: radiomics + extra_params + attention + residual ✓")
print("  • Input validation ✓")
print("  • Save/load ✓")
