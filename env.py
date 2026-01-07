import numpy as np

from conveyor import *

from SpacePartitioner import *

from geometry import *

from binpacker import *



class Env:

    def __init__(self, verbose):

        self.verbose = verbose

        self._state = None

        

    def reset(self):

        raise Exception('not implemented')

        

    def state(self, step=False):

        raise Exception('not implemented')

        

    def step(self, action):

        raise Exception('not implemented')

        

    def actions(self):

        raise Exception('not implemented')

        

def indices(actions):

    return [

        (i, j, k) 

        for i in range(len(actions)) 

        for j in range(len(actions[i])) 

        for k in range(len(actions[i][j]))

    ]



class MultiBinPackerEnv:

    def __init__(self, n_bins=1, size=(1200, 1200, 2000), bin_size=None, conveyor=None, **kwargs):

        self.n_bins = n_bins

        self.size = size

        self.bin_size = bin_size or size

        self.used_packers = []

        self.conveyor = conveyor

        # Suppression du fallback multi-bin: tout doit rester dans le premier bin

        # Compteurs d'échecs pour éviter boucles infinies

        self.failure_counts = {}

        self.max_failures_per_item = 10  # après 10 tentatives infructueuses -> stratégie de secours

        # Contraintes de hauteur (gripper)

        self.z_gripper_max = kwargs.get('z_gripper_max', 1650)

        self.z_gripper_offset = kwargs.get('z_gripper_offset', 745)

        self.z_local_max = max(0, self.z_gripper_max - self.z_gripper_offset)

        # Mode strict: aucune superposition XY même si empilement (vue top propre)

        self.strict_no_xy_overlap = kwargs.get('strict_no_xy_overlap', True)

        # Gap visuel minimal entre cartons (mm) pour éviter collage visuel

        self.min_xy_gap = kwargs.get('min_xy_gap', 50)

    

    def reset(self):

        """Reset environment and create initial bin."""

        print(f"[...] DEBUG env.reset() called")

        

        # === CRÉER LES CLASSES ITEM ET PACKER LOCALEMENT ===

        # (pas d'import externe)

        

        class Item:

            """Représente un carton avec dimensions et position."""

            def __init__(self, name, width, height, depth, rotation_type=0):

                self.name = name

                self.width = width

                self.height = height

                self.depth = depth

                self.rotation_type = rotation_type

                self.position = (0, 0, 0)

            

            def get_dimension(self):

                """Retourne (largeur, hauteur, profondeur)."""

                return (self.width, self.height, self.depth)

        

        class Packer:

            """Représente un bin (conteneur) pouvant contenir des items."""

            def __init__(self, width, height, depth):

                self.width = width

                self.height = height

                self.depth = depth

                self.items = []

            

            def add_item(self, item):

                """Ajoute un item au bin."""

                self.items.append(item)

                print(f"    [X] Item '{item.name}' added (now {len(self.items)} items in bin)")

                return True

        

        # Stocker les classes pour utilisation dans step()

        self.Item = Item

        self.Packer = Packer

        

        # Créer le premier bin

        self.used_packers = []

        initial_bin = Packer(*self.bin_size)

        self.used_packers.append(initial_bin)

        # Pas de compteurs de refus (on va explorer d'autres positions / empilement)

        self.failure_counts = {}

        

        print(f"  Created {len(self.used_packers)} initial bin(s)")

        print(f"  Bin size: {self.bin_size}")

        

        # Retourner l'état initial

        items = self.conveyor.peek() if self.conveyor else [None]

        print(f"  DEBUG: conveyor.peek() returned: {items}")

        

        h_maps = self.get_height_maps()

        actions = self.actions()

        

        print(f"  Initial state: {len(items)} items, {len(h_maps)} height maps, {len(actions)} action sets")

        

        return (items, h_maps, actions)

    

    def get_height_maps(self):

        """Return height maps for all bins."""

        h_maps = []

        for packer in self.used_packers:

            # Créer une height map 2D (width x height)

            h_map = np.zeros((self.bin_size[0], self.bin_size[1]), dtype=np.float32)

            

            # Remplir avec les hauteurs des items placés

            if hasattr(packer, 'items'):

                for item in packer.items:

                    if hasattr(item, 'position'):

                        x, y, z = item.position

                        if hasattr(item, 'get_dimension'):

                            w, h, d = item.get_dimension()

                        elif hasattr(item, 'width'):

                            w, h, d = item.width, item.height, item.depth

                        else:

                            continue

                        

                        # Remplir la zone occupée

                        x1, x2 = int(x), int(x + w)

                        y1, y2 = int(y), int(y + h)

                        x1 = max(0, min(x1, self.bin_size[0]-1))

                        x2 = max(0, min(x2, self.bin_size[0]))

                        y1 = max(0, min(y1, self.bin_size[1]-1))

                        y2 = max(0, min(y2, self.bin_size[1]))

                        

                        if x2 > x1 and y2 > y1:

                            h_map[x1:x2, y1:y2] = max(h_map[x1:x2, y1:y2].max(), z + d)

            

            h_maps.append(h_map)

        

        return h_maps

    

    def actions(self):

        """Génère positions avec BOTTOM-LEFT-FILL + grille multi-densité."""

        items = self.conveyor.peek() if self.conveyor else [None]

        

        all_actions = []

        for item in items:

            if item is None:

                all_actions.append([])

                continue

            

            item_w, item_h, item_d = item

            item_actions = []

            

            for packer in self.used_packers:

                bin_actions = []

                h_map = self._get_height_map(packer)

                

                # === BOTTOM-LEFT-FILL: grille 20mm pour plus de candidats sans trop pénaliser la compacité ===

                step = 20  # Grille 20mm : davantage de positions, compacité maintenue

                

                for x in range(0, self.bin_size[0], step):

                    for y in range(0, self.bin_size[1], step):

                        # === CONTRAINTE 1: Ne pas dépasser limites du bin ===

                        if x + item_w > self.bin_size[0] or y + item_h > self.bin_size[1]:

                            continue

                        

                        # === CONTRAINTE 2: Calculer hauteur minimale à cette position XY ===

                        x1 = max(0, int(x))

                        x2 = min(self.bin_size[0], int(x + item_w))

                        y1 = max(0, int(y))

                        y2 = min(self.bin_size[1], int(y + item_h))

                        

                        if x2 > x1 and y2 > y1:

                            z = float(h_map[x1:x2, y1:y2].max())

                        else:

                            continue

                        

                        # === CONTRAINTE 3: Vérifier limite hauteur gripper ===

                        if z + item_d > self.bin_size[2]:

                            continue

                        if z + self.z_gripper_offset > self.z_gripper_max:

                            continue

                        

                        # === CONTRAINTE 4: Vérifier collision AABB stricte ===

                        collision = False

                        for existing in getattr(packer, 'items', []):

                            if _aabb_overlap(

                                x, y, z, item_w, item_h, item_d,

                                existing.position[0], existing.position[1], existing.position[2],

                                existing.width, existing.height, existing.depth

                            ):

                                collision = True

                                break

                        

                        if collision:

                            continue

                        

                        # === CONTRAINTE 5: Vérifier gap XY (conflit visuel UNIQUEMENT même niveau) ===

                        gap_conflict = False

                        for existing in getattr(packer, 'items', []):

                            # Gap XY vérifié SEULEMENT si items à la même hauteur (tolérance 1mm)

                            if abs(z - existing.position[2]) < 1.0:  # Même niveau Z

                                if _xy_gap_conflict(

                                    x, y, item_w, item_h,

                                    existing.position[0], existing.position[1],

                                    existing.width, existing.height,

                                    self.min_xy_gap

                                ):

                                    gap_conflict = True

                                    break

                        

                        if gap_conflict:

                            continue

                        

                        # Position valide [X] ajouter aux candidats

                        bin_actions.append((x, y, z))

                

                # === TRI BOTTOM-LEFT: Priorité 1=Z bas, Priorité 2=Position proche (0,0) ===

                bin_actions.sort(key=lambda pos: (pos[2], pos[0] + pos[1]))

                

                # Garder les 100 meilleures positions (augmenté de 50 pour plus de choix)

                item_actions.append(bin_actions[:100])

            

            all_actions.append(item_actions)

        

        return all_actions

    

    def _get_height_map(self, packer):

        """Get 2D height map for a packer."""

        h_map = np.zeros((self.bin_size[0], self.bin_size[1]), dtype=np.float32)

        

        if not hasattr(packer, 'items'):

            return h_map

        

        for item in packer.items:

            if not hasattr(item, 'position'):

                continue

            

            x, y, z = item.position

            

            # Obtenir les dimensions

            if hasattr(item, 'get_dimension') and callable(item.get_dimension):

                w, h, d = item.get_dimension()

            elif hasattr(item, 'width'):

                w, h, d = item.width, item.height, item.depth

            else:

                continue

            

            # Mettre à jour la height map

            x1, x2 = max(0, int(x)), min(self.bin_size[0], int(x + w))

            y1, y2 = max(0, int(y)), min(self.bin_size[1], int(y + h))

            

            if x2 > x1 and y2 > y1:

                h_map[x1:x2, y1:y2] = np.maximum(h_map[x1:x2, y1:y2], z + d)

        

        return h_map

    

    def step(self, action):

        """Execute placement action."""

        print(f"\n[...] DEBUG env.step() called with action={action}")

        item_idx, bin_idx, pos_idx = action



        if bin_idx >= len(self.used_packers):

            print("[X] Invalid bin_idx")

            return self.reset(), -1.0, True



        items = self.conveyor.peek()

        if not items or item_idx >= len(items) or items[item_idx] is None:

            print("[X] Invalid item_idx")

            return self.reset(), -1.0, True



        item = items[item_idx]

        packer = self.used_packers[bin_idx]



        actions = self.actions()

        if item_idx >= len(actions):

            print("[X] No actions for item")

            return self.reset(), -1.0, True

        item_actions = actions[item_idx]

        if bin_idx >= len(item_actions):

            print("[X] No bin actions")

            return self.reset(), -1.0, True

        bin_actions = item_actions[bin_idx]

        if pos_idx >= len(bin_actions):

            print("[X] Invalid pos_idx")

            return self.reset(), -1.0, True



        position = bin_actions[pos_idx]



        item_obj = self.Item(

            name=f'item_{len(packer.items)+1}',

            width=item[0], height=item[1], depth=item[2], rotation_type=0

        )

        item_obj.position = position



        # === VÉRIFICATION STRICTE DE COLLISION ===

        collision_detected = False

        for existing in packer.items:

            # 1. Vérifier AABB volumétrique complet

            if _aabb_overlap(

                item_obj.position[0], item_obj.position[1], item_obj.position[2],

                item_obj.width, item_obj.height, item_obj.depth,

                existing.position[0], existing.position[1], existing.position[2],

                existing.width, existing.height, existing.depth

            ):

                print(f"[X] Collision AABB détectée avec item à {existing.position}")

                collision_detected = True

                break

            

            # 2. Vérifier gap XY (conflit visuel) SEULEMENT AU MÊME NIVEAU Z
            if abs(item_obj.position[2] - existing.position[2]) < 1.0:
                if _xy_gap_conflict(

                    item_obj.position[0], item_obj.position[1], item_obj.width, item_obj.height,

                    existing.position[0], existing.position[1], existing.width, existing.height,

                    self.min_xy_gap

                ):

                    print(f"[X] Conflit XY gap détecté avec item à {existing.position}")

                    collision_detected = True

                break

        

        # === SI COLLISION: Explorer les alternatives ===

        if collision_detected:

            alt_actions = self.actions()[item_idx][bin_idx]

            found_alternative = False

            

            for alt_idx, alt_pos in enumerate(alt_actions):

                if alt_idx == pos_idx:  # Ignorer position initiale

                    continue

                

                # Tester cette position alternative

                test_ok = True

                for existing in packer.items:

                    if _aabb_overlap(

                        alt_pos[0], alt_pos[1], alt_pos[2],

                        item_obj.width, item_obj.height, item_obj.depth,

                        existing.position[0], existing.position[1], existing.position[2],

                        existing.width, existing.height, existing.depth

                    ):

                        test_ok = False

                        break
                    if abs(alt_pos[2] - existing.position[2]) < 1.0:
                        if _xy_gap_conflict(
                            alt_pos[0], alt_pos[1], item_obj.width, item_obj.height,

                            existing.position[0], existing.position[1], existing.width, existing.height,

                            self.min_xy_gap

                        ):

                            test_ok = False

                        break

                

                if test_ok:

                    item_obj.position = alt_pos

                    print(f"[X] Position alternative trouvée (index {alt_idx}): {alt_pos}")

                    collision_detected = False

                    found_alternative = True

                    break

            

            if not found_alternative:

                print("[X] Pas d'alternative dans la liste [X] Recherche fine dense...")

                fine_pos = self._fine_search(item, packer)

                if fine_pos is not None:

                    item_obj.position = fine_pos

                    print(f"[X] Position trouvée via recherche fine: {fine_pos}")

                    collision_detected = False

                else:

                    print("[X] Recherche fine échouée [X] Item ignoré (skip)")

                    self.conveyor.grab(item_idx)

                    next_items = self.conveyor.peek()

                    done = all(x is None for x in next_items)

                    if done:

                        import copy

                        self.final_packers = copy.deepcopy(self.used_packers)

                        return (next_items, self.get_height_maps(), []), -0.1, True

                    return (next_items, self.get_height_maps(), self.actions()), -0.1, False

        

        # === PLACER L'ITEM SI PAS DE COLLISION ===

        success = packer.add_item(item_obj)

        if not success:

            print("[X] add_item failed")

            return self.reset(), -1.0, True



        # Retirer du convoyeur

        self.conveyor.grab(item_idx)



        # Reward (occupation)

        total_volume = self.bin_size[0] * self.bin_size[1] * self.bin_size[2]

        used_volume = sum(i.width * i.height * i.depth for p in self.used_packers for i in p.items)

        reward = used_volume / total_volume



        next_items = self.conveyor.peek()

        done = all(x is None for x in next_items)



        if done:

            import copy

            self.final_packers = copy.deepcopy(self.used_packers)

            terminal_items = next_items

            terminal_state = (terminal_items, self.get_height_maps(), [])

            return terminal_state, reward, True



        next_state = (next_items, self.get_height_maps(), self.actions())

        return next_state, reward, False



    def _fine_search(self, item, packer):

        """Recherche TRÈS dense (10mm) pour trouver une position sans collision.

        Utilise Bottom-Left-Fill: priorité z bas, puis coin (0,0).

        Retourne (x,y,z) ou None."""

        item_w, item_h, item_d = item

        h_map = self._get_height_map(packer)

        

        # Grille ultra-fine: 10mm pour vraie densité

        step = 10

        candidates = []

        

        for x in range(0, self.bin_size[0], step):

            for y in range(0, self.bin_size[1], step):

                # Vérifier limites bin

                if x + item_w > self.bin_size[0] or y + item_h > self.bin_size[1]:

                    continue

                

                # Hauteur minimale à cette position

                x1, x2 = int(x), min(self.bin_size[0], int(x + item_w))

                y1, y2 = int(y), min(self.bin_size[1], int(y + item_h))

                

                if x2 > x1 and y2 > y1:

                    z = float(h_map[x1:x2, y1:y2].max())

                else:

                    continue

                

                # Vérifier limites hauteur

                if z + item_d > self.bin_size[2] or z + self.z_gripper_offset > self.z_gripper_max:

                    continue

                

                test_pos = (x, y, z)

                

                # Vérifier collision AABB stricte

                collision = False

                for existing in packer.items:

                    if _aabb_overlap(

                        test_pos[0], test_pos[1], test_pos[2], item_w, item_h, item_d,

                        existing.position[0], existing.position[1], existing.position[2],

                        existing.width, existing.height, existing.depth

                    ):

                        collision = True

                        break

                    

                    # Vérifier gap XY SEULEMENT AU MÊME NIVEAU Z
                    if abs(test_pos[2] - existing.position[2]) < 1.0:
                        if _xy_gap_conflict(
                        test_pos[0], test_pos[1], item_w, item_h,

                        existing.position[0], existing.position[1], existing.width, existing.height,

                        self.min_xy_gap

                        ):

                            collision = True

                            break

                

                if not collision:

                    candidates.append(test_pos)

        

        if candidates:

            # Tri Bottom-Left: z bas, puis coin

            candidates.sort(key=lambda p: (p[2], p[0] + p[1]))

            print(f"  Fine search trouvé {len(candidates)} positions valides, meilleure: {candidates[0]}")

            return candidates[0]

        

        print(f"  Fine search: aucune position trouvée sur grille 10mm")

        return None



def _overlap_1d(a, la, b, lb):

    return a < b + lb and b < a + la



def _aabb_overlap(ax, ay, az, aw, ah, ad, bx, by, bz, bw, bh, bd):

    return (_overlap_1d(ax, aw, bx, bw) and

            _overlap_1d(ay, ah, by, bh) and

            _overlap_1d(az, ad, bz, bd))



def _xy_same_level_overlap(ax, ay, az, aw, ah, bx, by, bz, bw, bh, level_tol=1e-6):

    """Overlap strict sur XY si bases au même niveau (dans une tolérance)."""

    same_level = abs(az - bz) <= level_tol

    xy_overlap = _overlap_1d(ax, aw, bx, bw) and _overlap_1d(ay, ah, by, bh)

    return same_level and xy_overlap



def _xy_gap_conflict(ax, ay, aw, ah, bx, by, bw, bh, gap):

    """Retourne True si les deux rectangles XY se chevauchent OU sont séparés d'un espace < gap.

    Calcul de séparation minimale indépendante sur X et Y.

    """

    # Chevauchement direct

    overlap_x = _overlap_1d(ax, aw, bx, bw)

    overlap_y = _overlap_1d(ay, ah, by, bh)

    if overlap_x and overlap_y:

        return True

    # Séparations (0 si overlap)

    sep_x = 0 if overlap_x else (bx - (ax + aw) if ax + aw <= bx else (ax - (bx + bw)))

    sep_x = abs(sep_x)

    sep_y = 0 if overlap_y else (by - (ay + ah) if ay + ah <= by else (ay - (by + bh)))

    sep_y = abs(sep_y)

    # Conflit visuel si les deux axes sont à moins du gap

    return sep_x < gap and sep_y < gap