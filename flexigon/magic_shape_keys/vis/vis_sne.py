import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Prepare image features
clip_features_over_time = np.stack(img_opt_features_clip_arr)
# X = clip_features_over_time[:, -1].reshape((-1, 512))
X = clip_features_over_time.mean(axis=1)

# Prepare tiled text features
text_feat = np.array(text_ref_features_clip.cpu().detach().numpy()).reshape(1, 512)
text_feat = np.tile(text_feat, (X.shape[0], 1))  # Shape: (B, 512)

# Combine
X_combined = np.vstack([X, text_feat])  # Shape: (2B, 512)

# Run t-SNE
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_combined)

# Split back
B = X.shape[0]
X_tsne_imgs = X_tsne[:B]
X_tsne_text = X_tsne[B:]

# Color gradient for images
colors = plt.cm.jet(np.linspace(0, 1, B))

# Plot
plt.figure(figsize=(8, 6))
plt.title("t-SNE: Image Features vs. Tiled Text")

plt.scatter(X_tsne_imgs[:, 0], X_tsne_imgs[:, 1], c=colors, s=10, label='Images')
plt.scatter(X_tsne_text[:, 0], X_tsne_text[:, 1], c='black', s=10, marker='x', label='Text (Tiled)')

plt.legend()
plt.tight_layout()
plt.show()

# Plot as bars (bins)
std_per_feature = np.std(X, axis=0)
plt.figure(figsize=(12, 4))
# plt.yscale('log')
plt.bar(range(std_per_feature.shape[0]), -np.log10(std_per_feature))
plt.xlabel('Feature index')
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation per Feature')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
