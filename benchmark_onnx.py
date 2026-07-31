import os, sys, time
import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
from torchvision import transforms

TEST_IMAGE = '/public/home/dongshuai_ht/datasets/mvtec_ad/mvt_ad/bottle/test/broken_large/000.png'
ONNX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mgvit_bottle.onnx')

print("=" * 60)
print("  ONNX vs PyTorch 推理速度对比")
print("=" * 60)

print(f"\n[1] ONNX model: {ONNX_PATH}")
print(f"    size: {os.path.getsize(ONNX_PATH) / (1024*1024):.1f} MB")

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(ONNX_PATH, providers=providers)
print(f"    providers: {session.get_providers()}")

for inp in session.get_inputs():
    print(f"    input: {inp.name} shape={inp.shape}")

print(f"\n[2] 准备输入 (与 PyTorch benchmark 相同预处理) ...")
img = Image.open(TEST_IMAGE).convert('RGB')

transform = transforms.Compose([
    transforms.Resize(int(224 / 0.875)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img_tensor = transform(img).unsqueeze(0).numpy()
mask_tensor = np.ones((1, 1, 224, 224), dtype=np.float32)
print(f"    image: shape={img_tensor.shape} dtype={img_tensor.dtype}")
print(f"    mask: shape={mask_tensor.shape} dtype={mask_tensor.dtype}")

print(f"\n[3] 预热 10 次 ...")
for _ in range(10):
    _ = session.run(None, {"image": img_tensor, "mask": mask_tensor})
print("    预热完成")

print(f"\n[4] 正式测试 100 次 ...")
times = []
for i in range(100):
    t0 = time.time()
    outputs = session.run(None, {"image": img_tensor, "mask": mask_tensor})
    t1 = time.time()
    times.append((t1 - t0) * 1000)

avg_ms = sum(times) / len(times)
fps = 1000.0 / avg_ms
speedup = 100.0 / avg_ms

print(f"\n{'=' * 60}")
print(f"  结果对比")
print(f"{'=' * 60}")
print(f"  {'指标':<16} {'PyTorch':>12} {'ONNX':>12} {'提升':>12}")
print(f"  {'-'*52}")
print(f"  {'平均耗时(ms)':<16} {'100.0':>12} {avg_ms:>12.1f} {f'{speedup:.2f}x':>12}")
print(f"  {'等效 FPS':<16} {'10.0':>12} {fps:>12.1f} {'':>12}")
print(f"  {'最快(ms)':<16} {'79.8':>12} {min(times):>12.1f} {'':>12}")
print(f"  {'最慢(ms)':<16} {'150.2':>12} {max(times):>12.1f} {'':>12}")

cls_token = outputs[0]
print(f"\n  CLS token shape: {cls_token.shape}")
print(f"  CLS token norm: {np.linalg.norm(cls_token):.4f}")

print(f"\n{'=' * 60}")
print("  benchmark done")
print("=" * 60)
