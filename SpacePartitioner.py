import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from geometry import Cuboid

class Renderer:
    def __init__(self, size=None):
        self.size = size or (32, 32, 32)
        self.fig = plt.figure(figsize=(6,6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self._drawn = []

    def clear(self):
        self.ax.cla()
        self._drawn = []

    def draw(self, box, color='gray', mode='fill'):
        # box: Cuboid(x,y,z,w,h,d)
        # convert to cuboid faces and add Poly3DCollection (simple, opaque)
        x0, y0, z0 = box.x, box.y, box.z
        w, h, d = box.width, box.height, box.depth
        verts = [
            [(x0,y0,z0), (x0+w,y0,z0), (x0+w,y0+d,z0), (x0,y0+d,z0)],  # bottom
            [(x0,y0,z0+h), (x0+w,y0,z0+h), (x0+w,y0+d,z0+h), (x0,y0+d,z0+h)],  # top
            # ... 4 side faces ...
        ]
        poly = Poly3DCollection(verts, alpha=0.6 if mode=='fill' else 0)
        poly.set_facecolor(color)
        self.ax.add_collection3d(poly)
        self._drawn.append(poly)

    def show(self):
        sx, sy, sz = self.size
        self.ax.set_xlim(0, sx)
        self.ax.set_ylim(0, sy)
        self.ax.set_zlim(0, sz)
        self.ax.set_box_aspect((sx, sy, sz))
        plt.tight_layout()
        return self.fig
        
def render(size, spaces, colors=None):
    r = Renderer(size=size)
    for box in spaces:
        r.draw(box, color='green', mode='fill')
    r.draw(Cuboid(0,0,0, *size), color='red', mode='stroke')
    return r.show()
    
class SpacePartitioner:
    def __init__(self, size):
        self.size = size
        self.reset()
        self._colors = {}
        
    def reset(self):
        w, h, d = self.size
        self.free_splits = [Cuboid(0, 0, 0, w, h, d)]
        self.splits = []
        self.height_map = np.zeros((d, w), dtype=int)
        
    def fit(self, cuboid):
        outer = Cuboid(0, 0, 0, *self.size)
        
        if not outer.contain(cuboid):
            return False
        
        # print(len(self.splits), len(self.free_splits))
        if len(self.splits) < len(self.free_splits):
            for split in self.splits:
                if split.intersect(cuboid):
                    return False
            return True
        
        for split in self.free_splits:
            if split.contain(cuboid):
                return True
        return False
    
    def add(self, cuboid):
        if not self.fit(cuboid):
            return False
            
        self.splits.append(cuboid)
        
        (left, bottom, back), (right, top, front) = cuboid.bounding_box()
        cover = np.maximum(self.height_map[back:front, left:right], top)
        self.height_map[back:front, left:right] = cover
        
        partitions = []
        new_partitions = []
        for partition in self.free_splits:
            if partition.intersect(cuboid):
                new_partitions.extend(partition.split(cuboid))
            else:
                partitions.append(partition)
                
        n_partitions = len(partitions)
        # only overlapped partitions create smaller partitions
        # no need to check new_partition.contain(non_overlapped)
        for i in range(len(new_partitions)):
            contained = False
            
            partition = new_partitions[i]
            
            # possible to have contained partitions in new partitions
            for j in range(len(new_partitions)):
                # impossible to have two identical partitions
                # no need to check both a.contain(b) and b.contain(a)
                if i != j and new_partitions[j].contain(partition):
                    contained = True
                    break
                    
            if not contained:
                for j in range(n_partitions):
                    if partitions[j].contain(partition):
                        contained = True
                        break
            
            # preserve order
            if not contained:
                partitions.append(partition)
                
        self.free_splits = partitions
        
        return True
    
    def space_utilization(self):
        used = np.sum([split.volume for split in self.splits])
        
        free = []
        for free_split in self.free_splits:
            new_splits = [free_split]
            for added in free:
                new_splits = [split for new_split in new_splits for split in new_split.split(added, False)]
            free.extend(new_splits)
        free = np.sum([split.volume for split in free])
        if used + free != np.prod(self.size):
            raise Exception('wtf')
        
        return used / np.prod(self.size)
        
    def render(self, free=False):
        if free:
            splits = self.free_splits
        else:
            splits = self.splits
        return render(self.size, splits, self._colors)
