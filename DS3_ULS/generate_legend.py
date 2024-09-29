import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 4.5))
ax = fig.add_subplot(111)

color_choices = plt.get_cmap('Set2').colors
color_choices = color_choices[0::int(len(color_choices)/5)]

hatch_choices = [None, '////', '\\\\\\\\', '....', 'x', '.', '+', '*']
ax.barh(1, 2, edgecolor='black', color=color_choices[0], alpha=0.95, hatch=hatch_choices[0])
ax.barh(2, 2, edgecolor='black', color=color_choices[1], alpha=0.95, hatch=hatch_choices[1])
ax.barh(3, 2, edgecolor='black', color=color_choices[2], alpha=0.95, hatch=hatch_choices[2])
ax.barh(4, 2, edgecolor='black', color=color_choices[3], alpha=0.95, hatch=hatch_choices[3])

ax.legend(['Instance 1', 'Instance 2', 'Instance 3', 'Instance 4'], 
        ncol=4, 
        loc='upper center',
        framealpha=0)

ax.set_ylim(top=6)

plt.savefig('dummy_legend_figure.pdf', bbox_inches='tight')
