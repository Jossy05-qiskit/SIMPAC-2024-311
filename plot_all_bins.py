import matplotlib
matplotlib.use('Agg')  # backend non-interactif pour sauvegarder les fichiers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os

def plot_all_bins(packers, output_path='./outputs/bins.png'):
    """
    Visualize all bins with their packed items + BOUNDING BOX RED.
    packers: list of Packer objects with .items attribute
    """
    # Créer le dossier outputs si nécessaire
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not packers or len(packers) == 0:
        print(f"plot_all_bins: No packers to visualize (packers={packers})")
        return
    
    # Compter les bins non vides
    non_empty = []
    for p in packers:
        if hasattr(p, 'items') and p.items and len(p.items) > 0:
            non_empty.append(p)
    
    n_bins = len(non_empty)
    
    if n_bins == 0:
        print(f"plot_all_bins: All bins are empty (no items placed)")
        return
    
    # Créer la grille de subplots
    cols = min(3, n_bins)
    rows = max(1, (n_bins + cols - 1) // cols)
    
    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    
    for idx, packer in enumerate(non_empty):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        
        # Récupérer les dimensions du bin
        bin_width = getattr(packer, 'width', 1200)
        bin_height = getattr(packer, 'height', 1200)
        bin_depth = getattr(packer, 'depth', 2000)
        
        # === DESSINER LE BIN (gris transparent) ===
        vertices = [
            [0, 0, 0], [bin_width, 0, 0], [bin_width, bin_height, 0], [0, bin_height, 0],
            [0, 0, bin_depth], [bin_width, 0, bin_depth], [bin_width, bin_height, bin_depth], [0, bin_height, bin_depth]
        ]
        
        edges = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]]
        ]
        
        bin_collection = Poly3DCollection(edges, linewidths=1, edgecolors='black', alpha=0.1, facecolors='lightgray')
        ax.add_collection3d(bin_collection)
        
        # === CALCULER BOUNDING BOX (boîte englobante des items) ===
        min_x, max_x = bin_width, 0
        min_y, max_y = bin_height, 0
        min_z, max_z = bin_depth, 0
        
        # Dessiner chaque item et calculer bounding box
        for item in packer.items:
            x, y, z = getattr(item, 'position', (0, 0, 0))
            
            # Essayer différentes méthodes pour obtenir les dimensions
            if hasattr(item, 'get_dimension') and callable(item.get_dimension):
                w, h, d = item.get_dimension()
            elif hasattr(item, 'width'):
                w = item.width
                h = getattr(item, 'height', 300)
                d = getattr(item, 'depth', 300)
            else:
                w, h, d = (300, 300, 300)  # dimensions par défaut
            
            # Mettre à jour bounding box
            min_x = min(min_x, x)
            max_x = max(max_x, x + w)
            min_y = min(min_y, y)
            max_y = max(max_y, y + h)
            min_z = min(min_z, z)
            max_z = max(max_z, z + d)
            
            # Créer les sommets de la boîte
            item_vertices = [
                [x, y, z], [x+w, y, z], [x+w, y+h, z], [x, y+h, z],
                [x, y, z+d], [x+w, y, z+d], [x+w, y+h, z+d], [x, y+h, z+d]
            ]
            
            item_faces = [
                [item_vertices[j] for j in [0, 1, 2, 3]],
                [item_vertices[j] for j in [4, 5, 6, 7]],
                [item_vertices[j] for j in [0, 1, 5, 4]],
                [item_vertices[j] for j in [2, 3, 7, 6]],
                [item_vertices[j] for j in [0, 3, 7, 4]],
                [item_vertices[j] for j in [1, 2, 6, 5]]
            ]
            
            # Couleur aléatoire pour chaque item
            color = np.random.rand(3,)
            item_collection = Poly3DCollection(item_faces, facecolors=color, linewidths=0.5, edgecolors='black', alpha=0.7)
            ax.add_collection3d(item_collection)
        
        # === DESSINER BOUNDING BOX EN ROUGE (délimitation du bloc compact) ===
        bbox_vertices = [
            [min_x, min_y, min_z], [max_x, min_y, min_z], [max_x, max_y, min_z], [min_x, max_y, min_z],
            [min_x, min_y, max_z], [max_x, min_y, max_z], [max_x, max_y, max_z], [min_x, max_y, max_z]
        ]
        
        bbox_edges = [
            [bbox_vertices[0], bbox_vertices[1], bbox_vertices[2], bbox_vertices[3]],  # bottom
            [bbox_vertices[4], bbox_vertices[5], bbox_vertices[6], bbox_vertices[7]],  # top
            [bbox_vertices[0], bbox_vertices[1], bbox_vertices[5], bbox_vertices[4]],  # front
            [bbox_vertices[2], bbox_vertices[3], bbox_vertices[7], bbox_vertices[6]],  # back
            [bbox_vertices[0], bbox_vertices[3], bbox_vertices[7], bbox_vertices[4]],  # left
            [bbox_vertices[1], bbox_vertices[2], bbox_vertices[6], bbox_vertices[5]]   # right
        ]
        
        # Boîte rouge avec transparence légère
        bbox_collection = Poly3DCollection(bbox_edges, linewidths=2, edgecolors='red', 
                                           alpha=0.15, facecolors='red')
        ax.add_collection3d(bbox_collection)
        
        # Configurer les axes
        ax.set_xlim([0, bin_width])
        ax.set_ylim([0, bin_height])
        ax.set_zlim([0, bin_depth])
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        
        # Ajouter infos compacité
        bbox_volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
        items_volume = sum(getattr(item, 'width', 300) * getattr(item, 'height', 300) * 
                          getattr(item, 'depth', 300) for item in packer.items)
        compacity = (items_volume / bbox_volume * 100) if bbox_volume > 0 else 0
        
        title = f'Bin {idx+1}: {len(packer.items)} items\nBBox: {max_x-min_x:.0f}×{max_y-min_y:.0f}×{max_z-min_z:.0f}mm\nCompacity: {compacity:.1f}%'
        ax.set_title(title, fontsize=10)
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[X] Visualization saved to {output_path}")