import matplotlib.pyplot as plt
import numpy as np

# 数据点 (F1 score, 时间)
tecdi = [(100, 0.25), (200, 0.20), (300, 0.18)]
dynotears = [(50, 0.6), (100, 0.3), (200, 0.2)]
nts_noteas = [(5, 0.95), (10, 0.92), (20, 0.85)]
local_ours = [(5, 0.92), (10, 0.90), (20, 0.87)]

# 颜色和标记样式
colors = {'Tecdi': 'm', 'Dynotears': 'y', 'Nts-noteas': 'g', 'LOCAL': 'r'}
markers = {'Tecdi': 'o', 'Dynotears': 'd', 'Nts-noteas': 's', 'LOCAL': '^'}

plt.figure(figsize=(8, 6))

# 绘制 Tecdi
x, y = zip(*tecdi)
plt.plot(x, y, linestyle='dashed', color=colors['Tecdi'], marker=markers['Tecdi'], label='Tecdi')
for i, (a, b) in enumerate(tecdi):
    plt.text(a, b, str((i+1)*5), fontsize=12)

# 绘制 Dynotears
x, y = zip(*dynotears)
plt.plot(x, y, linestyle='dashed', color=colors['Dynotears'], marker=markers['Dynotears'], label='Dynotears')
for i, (a, b) in enumerate(dynotears):
    plt.text(a, b, str((i+1)*5), fontsize=12)

# 绘制 Nts-noteas
x, y = zip(*nts_noteas)
plt.plot(x, y, linestyle='dotted', color=colors['Nts-noteas'], marker=markers['Nts-noteas'], label='Nts-noteas')
for i, (a, b) in enumerate(nts_noteas):
    plt.text(a, b, str((i+1)*5), fontsize=12)

# 绘制 LOCAL
x, y = zip(*local_ours)
plt.plot(x, y, linestyle='dotted', color=colors['LOCAL'], marker=markers['LOCAL'], label='LOCAL')
for i, (a, b) in enumerate(local_ours):
    plt.text(a, b, str((i+1)*5), fontsize=12)

# 设置图例
plt.legend()
plt.xlabel("Time (seconds)")
plt.ylabel("Performance (F1 score)")
plt.title("Performance vs. Time")
plt.xlim(0, 320)
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.5)

# 插入小图
ax_inset = plt.axes([0.55, 0.55, 0.3, 0.3])
ax_inset.plot(*zip(*nts_noteas), linestyle='dotted', color=colors['Nts-noteas'], marker=markers['Nts-noteas'])
ax_inset.plot(*zip(*local_ours), linestyle='dotted', color=colors['LOCAL'], marker=markers['LOCAL'])
ax_inset.set_xlim(0, 15)
ax_inset.set_ylim(0.75, 1.0)
ax_inset.grid(True, linestyle='--', alpha=0.5)

plt.show()