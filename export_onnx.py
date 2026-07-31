import os, sys
import torch
import torch.nn as nn

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout = open(os.path.join(OUTPUT_DIR, "export_onnx_output.txt"), "w", buffering=1)
sys.stderr = sys.stdout

ORIGINAL_CODE_PATH = "/public/home/dongshuai_ht/AnomalyNCD-main-0.749"
sys.path.insert(0, ORIGINAL_CODE_PATH)

from models.modules.load_backbone import load_backbone
from models.modules._classifier import MultiHead

CHECKPOINT_PATH = "/public/home/dongshuai_ht/AnomalyNCD-main-0.749/outputs/train_bottle_ours_seed70/log/AnomalyNCD_MTD_bottle_(2026.06.10_00:26)/checkpoints/model.pt"

print("=" * 60)
print("  AnomalyNCD ONNX export")
print("=" * 60)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[1] device: {device}")

pretrained_backbone = "dino_vitb8"
mask_layers = 3
feat_dim = 768
num_mlp_layers = 3
mlp_out_dim = 8
n_head = 4

print(f"[2] build model: backbone={pretrained_backbone}, mask_layers={mask_layers}")
MGViT = load_backbone(pretrained_backbone, mask_layers=mask_layers)
projector = MultiHead(in_dim=feat_dim, out_dim=mlp_out_dim, nlayers=num_mlp_layers, n_head=n_head)
model = nn.Sequential(MGViT, projector).to(device)

print(f"[3] load checkpoint: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
print(f"    missing_keys: {missing}")
print(f"    unexpected_keys: {unexpected}")
model.eval()

print(f"[4] test forward ...")
dummy_img = torch.zeros(1, 3, 224, 224).to(device)
dummy_mask = torch.ones(1, 1, 224, 224).to(device)
with torch.no_grad():
    cls_token = MGViT(dummy_img, dummy_mask)
    proj, logits = projector(cls_token)
print(f"    CLS token shape: {cls_token.shape}")
print(f"    proj shape: {proj.shape}")
print(f"    logits[0] shape: {logits[0].shape}")

print(f"\n[5] export MGViT to ONNX ...")
onnx_path = os.path.join(OUTPUT_DIR, "mgvit_bottle.onnx")

class MGViTWrapper(nn.Module):
    def __init__(self, mgvit):
        super().__init__()
        self.mgvit = mgvit
    def forward(self, x, mask):
        return self.mgvit(x, mask)

wrapped = MGViTWrapper(MGViT).to(device).eval()

try:
    torch.onnx.export(
        wrapped,
        (dummy_img, dummy_mask),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["image", "mask"],
        output_names=["cls_token"],
        dynamic_axes={"image": {0: "batch"}, "mask": {0: "batch"}, "cls_token": {0: "batch"}},
    )
    print(f"    SUCCESS: {onnx_path}")
    sz = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"    size: {sz:.1f} MB")
except Exception as e:
    print(f"    FAILED: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'=' * 60}")
print("  done")
print("=" * 60)
