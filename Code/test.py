import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 4 * np.pi, 500)
y = np.sin(x) + 0.5 * np.sin(3 * x) + 0.2 * np.random.normal(size=x.size)

plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Noisy double sine', color='blue')

# 添加渐变色散点，颜色根据y值映射
scatter = plt.scatter(x, y, c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Amplitude')

plt.title('Noisy Double Sine Wave with Colored Scatter')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.legend()
plt.grid(True)
plt.show()
