import numpy as np
from split_gen import *
import itertools

class ItemGenerator:
    def __init__(self, k=1, scale=1.0):
        self.k = k
        self.scale = scale
        self.items = []
        self.index = 0

    def reset(self):
        """Réinitialise l'itérateur et retourne self (évite None)."""
        self.index = 0
        return self

    def peek(self):
        """Retourne une liste des prochaines k tailles (ne modifie pas index)."""
        if not self.items:
            return None
        res = []
        n = len(self.items)
        for i in range(self.k):
            res.append(self.items[(self.index + i) % n])
        return res

    def next_item(self):
        """Retourne l'item courant et avance l'index."""
        if not self.items:
            return None
        item = self.items[self.index % len(self.items)]
        self.index += 1
        return item

    def __iter__(self):
        return self

    def __next__(self):
        it = self.next_item()
        if it is None:
            raise StopIteration
        return it


class FileConveyor(ItemGenerator):
    def __init__(self, filepath, k=1):
        self.filepath = filepath
        self.items = []
        self.index = 0
        self.k = k
        self.debug = False
        self._load_items()
    
    def _load_items(self):
        """Load items from file."""
        try:
            with open(self.filepath, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            print(f"FileConveyor: read {len(lines)} lines from {self.filepath}")
            
            self.items = []
            for line in lines:
                parts = line.replace(',', ' ').split()
                if len(parts) >= 3:
                    try:
                        w, h, d = int(parts[0]), int(parts[1]), int(parts[2])
                        self.items.append((w, h, d))
                    except ValueError:
                        continue
            
            print(f"FileConveyor: loaded {len(self.items)} valid items.")
        except Exception as e:
            print(f"FileConveyor error: {e}")
            self.items = []
    
    def reset(self):
        """Reset to beginning (keep items loaded)."""
        self.index = 0
        if self.debug:
            print(f"FileConveyor.reset(): index reset to 0, {len(self.items)} items available")
    
    def peek(self, k=None):
        """Return next k items without removing them."""
        if k is None:
            k = self.k
        result = []
        for i in range(k):
            idx = self.index + i
            if idx < len(self.items):
                result.append(self.items[idx])
            else:
                result.append(None)
        
        # DEBUG
        if self.debug:
            print(f"  DEBUG FileConveyor.peek({k}): index={self.index}, total={len(self.items)}, returning {result}")
        return result
    
    def grab(self, item_idx=0):
        """Remove item at item_idx from peek window."""
        actual_idx = self.index + item_idx
        if actual_idx < len(self.items):
            self.items.pop(actual_idx)
            if self.debug:
                print(f"  DEBUG FileConveyor.grab({item_idx}): removed index {actual_idx}, {len(self.items)} remaining")
        else:
            if self.debug:
                print(f"  WARNING FileConveyor.grab({item_idx}): index {actual_idx} out of range")

def rotated_sizes(item, rotate=True, remove_duplicate=True):
    w, h, d = item
    rots = {(w, h, d)}
    if rotate:
        rots.update({(w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)})
    res = list(rots)
    if remove_duplicate:
        return res
    else:
        return list(itertools.permutations((w, h, d)))


class InputConveyor:
    """Simple conveyor for interactive/manual input (stub)."""
    def __init__(self, k=1):
        self.k = k
        self.items = []
        self.index = 0
    
    def reset(self):
        self.index = 0
        return self
    
    def peek(self, k=None):
        if k is None:
            k = self.k
        result = []
        for i in range(k):
            idx = self.index + i
            if idx < len(self.items):
                result.append(self.items[idx])
            else:
                result.append(None)
        return result
    
    def grab(self, item_idx=0):
        actual_idx = self.index + item_idx
        if actual_idx < len(self.items):
            self.items.pop(actual_idx)


class Conveyor:
    """Generic conveyor stub."""
    def __init__(self, k=1, **kwargs):
        self.k = k
        self.items = []
        self.index = 0
    
    def reset(self):
        self.index = 0
        return self
    
    def peek(self, k=None):
        if k is None:
            k = self.k
        return [None] * k
    
    def grab(self, item_idx=0):
        pass


# helper rotation function (simple placeholder)
def rotated_sizes(item, rotate=True, remove_duplicate=True):
    w, h, d = item
    rots = {(w, h, d)}
    if rotate:
        rots.update({(w, d, h), (h, w, d), (h, d, w), (d, w, h), (d, h, w)})
    res = list(rots)
    if remove_duplicate:
        return res
    else:
        return list(itertools.permutations((w, h, d)))

# lorsque tu instancies le conveyor pour deeppack3d :
# file_conv = FileConveyor(k=4, path='inputpack.txt', scale=1.0)  # scale=1.0 → mm inchangé

# === COMMENTER OU SUPPRIMER CES LIGNES (tout en bas du fichier) ===
# file_conv = FileConveyor(k=4, path='inputpack.txt', scale=1.0)
# OU remplacer par :
if __name__ == "__main__":
    # Code de test uniquement si exécuté directement
    file_conv = FileConveyor('inputpack.txt')
    file_conv.reset()
    print(f"Test: {len(file_conv.items)} items loaded")