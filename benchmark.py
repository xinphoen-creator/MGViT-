import time
import sys
sys.path.insert(0, '/public/home/dongshuai_ht/AnomalyNCD-Project')

from shared_utils.inference_utils import load_model, predict_single
from PIL import Image

TEST_IMAGE = '/public/home/dongshuai_ht/datasets/mvtec_ad/mvt_ad/bottle/test/broken_large/000.png'

print("=" * 50)
print("  AnomalyNCD 推理性能基准测试")
print("=" * 50)

print("\n[1] 加载模型 ...")
state = load_model()
print(f"  模型加载完成, device={state.device}")
print(f"  类别数: {len(state.class_names)}")

print(f"\n[2] 加载测试图片: {TEST_IMAGE}")
img = Image.open(TEST_IMAGE).convert("RGB")
print(f"  图片尺寸: {img.size}")

print("\n[3] 预热 10 次 ...")
for i in range(10):
    _ = predict_single(state, img)
print("  预热完成")

print("\n[4] 正式测试 100 次 ...")
times = []
for i in range(100):
    t0 = time.time()
    result = predict_single(state, img)
    t1 = time.time()
    times.append((t1 - t0) * 1000)

avg_ms = sum(times) / len(times)
fps = 1000.0 / avg_ms

print(f"\n{'=' * 50}")
print(f"  测试结果")
print(f"{'=' * 50}")
print(f"  总测试次数: {len(times)}")
print(f"  平均耗时:   {avg_ms:.1f} ms")
print(f"  等效 FPS:   {fps:.1f}")
print(f"  最快:       {min(times):.1f} ms")
print(f"  最慢:       {max(times):.1f} ms")

print(f"\n  最后一次推理结果:")
print(f"    预测类别:   {result['predicted_class']}")
print(f"    是否异常:   {result['is_anomaly']}")
print(f"    异常分数:   {result['image_score']}")
print(f"    推理耗时:   {result['elapsed_ms']} ms")
print(f"    最高相似度: {max(result['cosine_similarities'].values())}")

print(f"\n{'=' * 50}")
print("  基准测试完成")
print(f"{'=' * 50}")