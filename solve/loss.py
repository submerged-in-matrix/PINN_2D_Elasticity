import matplotlib.pyplot as plt
from src.train import hist

plt.figure(figsize=(6,4))
plt.plot(hist, lw=2)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Relative Loss')
plt.title('Loss Convergence Curve')
plt.grid(True, which='both', ls='--', lw=0.5)
plt.tight_layout()
plt.show()