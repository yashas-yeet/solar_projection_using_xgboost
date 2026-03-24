# ==============================================================================
#  Hybrid Solar Grid Output Forecaster
#  Copyright (c) 2025 Yashas Vishwakarma
#
#  DUAL LICENSING NOTICE:
#  This source code is protected by copyright law and is available under two
#  distinct licensing models. You may choose to use it under:
#
#  1. OPEN SOURCE (GPLv3):
#     Free for academic, personal, and open-source projects.
#     Condition: If you distribute software using this code, your ENTIRE
#     project must also be open-source under GPLv3.
#
#  2. COMMERCIAL LICENSE:
#     Required for proprietary (closed-source) commercial products.
#     Allows you to keep your source code private and provides legal support.
#
#  For commercial licensing inquiries, contact: [yashasakvish@gmail.com]
#  Full terms available in the LICENSE file.
# ==============================================================================


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as patches

# ==========================================
# XGBoost Single Decision Tree Simulator
# ==========================================

# 1. Setup the main window and axes
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.35) # Leave room for sliders at the bottom
fig.patch.set_facecolor('#242424')
ax.set_facecolor('#242424')
ax.axis('off')
ax.set_title("Inside the AI: Single Decision Tree Logic", color='white', fontsize=16, pad=20)

# 2. Define the exact coordinates for the Tree Nodes (X, Y)
nodes = {
    'root':  {'pos': (0.5, 0.9),  'label': "Node 1:\nIs GHI < 0.2 kW?"},
    'leaf1': {'pos': (0.2, 0.65), 'label': "LEAF A\nNight / Heavy Cloud\nPrediction: 0 kW"},
    'node2': {'pos': (0.8, 0.65), 'label': "Node 2:\nIs Hour between 9 & 15?"},
    'leaf2': {'pos': (0.55, 0.4), 'label': "LEAF B\nOff-Peak Sun\nPrediction: 15 kW"},
    'node3': {'pos': (1.05, 0.4), 'label': "Node 3:\nIs Temp > 35°C?"},
    'leaf3': {'pos': (0.8, 0.15), 'label': "LEAF C\nOptimal Generation\nPrediction: 50 kW"},
    'leaf4': {'pos': (1.3, 0.15), 'label': "LEAF D\nThermal Degradation\nPrediction: 38 kW"}
}

# Define the lines connecting the nodes (Start, End, True/False condition)
edges = [
    ('root', 'leaf1', True, 'YES'),
    ('root', 'node2', False, 'NO'),
    ('node2', 'leaf2', False, 'NO'),
    ('node2', 'node3', True, 'YES'),
    ('node3', 'leaf4', True, 'YES'),
    ('node3', 'leaf3', False, 'NO')
]

ax.set_xlim(0, 1.5)
ax.set_ylim(0, 1.0)

# Dictionaries to store drawn objects so we can update them later
drawn_nodes = {}
drawn_edges = {}
drawn_texts = {}

# 3. Draw the initial static tree
def draw_tree():
    # Draw edges (lines)
    for start, end, condition, text in edges:
        x1, y1 = nodes[start]['pos']
        x2, y2 = nodes[end]['pos']
        line, = ax.plot([x1, x2], [y1, y2], color='#555555', lw=3, zorder=1)
        drawn_edges[(start, end)] = line
        
        # Add Yes/No text to the lines
        mid_x, mid_y = (x1 + x2)/2, (y1 + y2)/2
        ax.text(mid_x, mid_y + 0.02, text, color='white', fontsize=10, ha='center',
                bbox=dict(facecolor='#242424', edgecolor='none', pad=1))

    # Draw nodes (boxes)
    for name, data in nodes.items():
        x, y = data['pos']
        is_leaf = 'LEAF' in data['label']
        box_color = '#2c3e50' if not is_leaf else '#8e44ad'
        
        bbox = dict(boxstyle="round,pad=0.6", facecolor=box_color, edgecolor='#7f8c8d', lw=2)
        txt = ax.text(x, y, data['label'], ha="center", va="center", color="white", 
                      fontsize=11, fontweight='bold', bbox=bbox, zorder=2)
        drawn_nodes[name] = txt

draw_tree()

# 4. Display the Final Output text at the top
output_display = ax.text(0.5, 0.05, "Final Output: -- kW", ha="center", va="center", 
                         color="#2ecc71", fontsize=20, fontweight='bold', 
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a1a", edgecolor="#2ecc71", lw=2))

# ==========================================
# Create Interactive Sliders
# ==========================================
axcolor = '#333333'
ax_ghi  = plt.axes([0.2, 0.25, 0.65, 0.03], facecolor=axcolor)
ax_temp = plt.axes([0.2, 0.18, 0.65, 0.03], facecolor=axcolor)
ax_hour = plt.axes([0.2, 0.11, 0.65, 0.03], facecolor=axcolor)

s_ghi  = Slider(ax_ghi, 'GHI (kW/m²)', 0.0, 1.2, valinit=0.8, valstep=0.05, color='#f1c40f')
s_temp = Slider(ax_temp, 'Temp (°C)', 10.0, 50.0, valinit=25.0, valstep=1.0, color='#e74c3c')
s_hour = Slider(ax_hour, 'Hour (0-23)', 0, 23, valinit=12, valstep=1, color='#3498db')

# Format slider text color
for s in [s_ghi, s_temp, s_hour]:
    s.label.set_color('white')
    s.valtext.set_color('white')

# ==========================================
# Core Logic: The Update Function
# ==========================================
def update(val):
    # 1. Read current slider values
    ghi = s_ghi.val
    temp = s_temp.val
    hour = s_hour.val
    
    # 2. Reset all visuals to dark/inactive
    for line in drawn_edges.values():
        line.set_color('#555555')
        line.set_linewidth(3)
    for name, txt in drawn_nodes.items():
        is_leaf = 'LEAF' in nodes[name]['label']
        txt.get_bbox_patch().set_edgecolor('#7f8c8d')
        txt.get_bbox_patch().set_facecolor('#2c3e50' if not is_leaf else '#8e44ad')

    # 3. Simulate the Data routing through the Tree
    active_path = []
    active_nodes = ['root']
    final_output = ""
    
    # Node 1 Check
    if ghi < 0.2:
        active_path.append(('root', 'leaf1'))
        active_nodes.append('leaf1')
        final_output = "0 kW (Night/Cloud)"
    else:
        active_path.append(('root', 'node2'))
        active_nodes.append('node2')
        
        # Node 2 Check
        if not (9 <= hour <= 15):
            active_path.append(('node2', 'leaf2'))
            active_nodes.append('leaf2')
            final_output = "15 kW (Off-Peak)"
        else:
            active_path.append(('node2', 'node3'))
            active_nodes.append('node3')
            
            # Node 3 Check
            if temp > 35:
                active_path.append(('node3', 'leaf4'))
                active_nodes.append('leaf4')
                final_output = "38 kW (Thermal Drop)"
            else:
                active_path.append(('node3', 'leaf3'))
                active_nodes.append('leaf3')
                final_output = "50 kW (Optimal!)"

    # 4. Highlight the Active Path
    for start, end in active_path:
        drawn_edges[(start, end)].set_color('#2ecc71') # Bright Green
        drawn_edges[(start, end)].set_linewidth(5)
        
    for node in active_nodes:
        drawn_nodes[node].get_bbox_patch().set_edgecolor('#2ecc71')
        if 'LEAF' in nodes[node]['label']:
            drawn_nodes[node].get_bbox_patch().set_facecolor('#27ae60') # Bright Green Leaf

    # 5. Update Text
    output_display.set_text(f"Final Prediction: {final_output}")
    fig.canvas.draw_idle()

# Connect the sliders to the update function
s_ghi.on_changed(update)
s_temp.on_changed(update)
s_hour.on_changed(update)

# Run the update function once to highlight the default starting state
update(0)

plt.show()