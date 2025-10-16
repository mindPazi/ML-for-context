import pickle
import numpy as np
from sklearn.decomposition import PCA

print("=" * 60)
print("ANALYZING EMBEDDINGS STRUCTURE")
print("=" * 60)


print("\n[1/3] Loading embeddings from cache...")
with open('cache/embeddings_normalized.pkl', 'rb') as f:
    embeddings = pickle.load(f)

print(f"      Shape: {embeddings.shape}")
print(f"      Total values: {embeddings.size:,}")


print("\n[2/3] Sparsity Analysis:")
print("-" * 60)

exact_zeros = np.sum(embeddings == 0)
near_zeros_001 = np.sum(np.abs(embeddings) < 0.001)
near_zeros_01 = np.sum(np.abs(embeddings) < 0.01)
near_zeros_05 = np.sum(np.abs(embeddings) < 0.05)
total = embeddings.size

print(f"  Exact zeros (= 0):        {exact_zeros:,} ({exact_zeros/total*100:.4f}%)")
print(f"  Near-zero (< 0.001):      {near_zeros_001:,} ({near_zeros_001/total*100:.4f}%)")
print(f"  Near-zero (< 0.01):       {near_zeros_01:,} ({near_zeros_01/total*100:.4f}%)")
print(f"  Near-zero (< 0.05):       {near_zeros_05:,} ({near_zeros_05/total*100:.4f}%)")

print(f"\n  Value Statistics:")
print(f"    Min:    {embeddings.min():.6f}")
print(f"    Max:    {embeddings.max():.6f}")
print(f"    Mean:   {embeddings.mean():.6f}")
print(f"    Std:    {embeddings.std():.6f}")
print(f"    Median: {np.median(embeddings):.6f}")


zeros_per_doc = np.sum(embeddings == 0, axis=1)
print(f"\n  Per-document zeros:")
print(f"    Mean:   {zeros_per_doc.mean():.2f} / 768")
print(f"    Min:    {zeros_per_doc.min()} / 768")
print(f"    Max:    {zeros_per_doc.max()} / 768")


print("\n[3/3] Intrinsic Dimensionality:")
print("-" * 60)

print("  Computing PCA ...")
pca = PCA()
pca.fit(embeddings)

cumvar = np.cumsum(pca.explained_variance_ratio_)
dims_50 = np.argmax(cumvar >= 0.50) + 1
dims_90 = np.argmax(cumvar >= 0.90) + 1
dims_95 = np.argmax(cumvar >= 0.95) + 1
dims_99 = np.argmax(cumvar >= 0.99) + 1

print(f"\n  Dimensions needed to explain variance:")
print(f"    50% variance: {dims_50:3d} / 768 ({dims_50/768*100:.1f}%)")
print(f"    90% variance: {dims_90:3d} / 768 ({dims_90/768*100:.1f}%)")
print(f"    95% variance: {dims_95:3d} / 768 ({dims_95/768*100:.1f}%)")
print(f"    99% variance: {dims_99:3d} / 768 ({dims_99/768*100:.1f}%)")

print(f"\n  Top 10 components explain: {cumvar[9]*100:.2f}% of variance")
print(f"  Top 50 components explain: {cumvar[49]*100:.2f}% of variance")


print("\nValue Distribution:")
print("-" * 60)

bins = [-1, -0.1, -0.05, -0.01, 0, 0.01, 0.05, 0.1, 1]
hist, _ = np.histogram(embeddings.flatten(), bins=bins)
bin_labels = ["< -0.1", "[-0.1, -0.05)", "[-0.05, -0.01)", "[-0.01, 0)", 
              "[0, 0.01)", "[0.01, 0.05)", "[0.05, 0.1)", ">= 0.1"]

for label, count in zip(bin_labels, hist):
    pct = count / total * 100
    print(f"  {label:15s}: {count:10,} ({pct:5.2f}%)")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
