import os

import argparse

from pathlib import Path

import numpy as np

import pandas as pd

from pallet_utils import PalletConfig



# --- Chemins ---


REPO_DIR = Path(r"C:\Users\RTA15\Documents\EKL DELMIA\SIMPAC-2024-311 - Copy V10")


INPUT_PATH = REPO_DIR / "inputpack.txt"

XLSX_PATH = REPO_DIR / "position_optimized.xlsx"



# --- rotation Y imposée pour DELMIA ---

RY_FOR_DELMIA = -3.141592654



# --- Configuration de la palette ---

PALLET_CONFIG = PalletConfig(

    pallet_x=1295.889166,

    pallet_y=9096.996345,

    pallet_z=0,

    pallet_length=1200,

    pallet_width=1200,

    pallet_height=145,

    bin_size=(1200, 1200, 2000)

)



# === OFFSET MANUEL (à ajuster selon DELMIA) ===

OFFSET_X = 850.0

OFFSET_Y = -2650.0

OFFSET_Z = 300



# Petite marge verticale anti-interpénétration (mm)

EPS_Z = 0.0



def to_py_scalar(v):

    """Conversion numpy scalar [X] python."""

    try:

        return v.item() if isinstance(v, (np.integer, np.floating)) else v

    except Exception:

        return v



def run_packing(input_file, output_file, method='rl', lookahead=32, n_iterations=-1, verbose=0):

    """Exécute DeepPack3D et génère Excel avec coordonnées absolues + HAUTEUR CORRIGÉE."""

    if not input_file.exists():

        print(f"Erreur : {input_file} introuvable.")

        return 1

    

    rows = []

    placed_all = []  # Pour validation chevauchement



    print(f"Démarrage DeepPack3D (méthode={method}, lookahead={lookahead})...")

    

    # === IMPORTER LES BONNES CLASSES ===

    from env import MultiBinPackerEnv

    from conveyor import FileConveyor

    from agent import Agent

    

    # === INITIALISER LE CONVEYOR CORRECTEMENT ===

    print(f"[...] Chargement du fichier : {input_file}")

    # FileConveyor avec k=lookahead pour que peek() retourne tous les items

    conv = FileConveyor(str(input_file), k=lookahead)

    conv.reset()

    print(f"[X] Conveyor initialisé (k={lookahead}) : {len(conv.items)} items chargés")

    

    if not conv.items or len(conv.items) == 0:

        print(f"[X] Aucun item trouvé dans {input_file}")

        return 1

    

    # Initialiser l'environnement avec le conveyor

    env = MultiBinPackerEnv(n_bins=10, size=(1200, 1200, 2000), conveyor=conv)

    env.k = lookahead  # Force l'environnement à utiliser le bon lookahead

    

    # Charger le modèle

    model_path = f'./models/k={lookahead}.h5'

    agent = Agent(env, visualize=False)

    

    try:

        from tensorflow.keras.models import load_model

        agent.q_net = load_model(model_path, compile=False)

        print(f"[X] Modèle chargé : {model_path}")

    except Exception as e:

        print(f"[X] Erreur chargement modèle : {e}")

        return 1

    

    # Exécuter le placement

    print("[...] Placement des items...")

    for episode_id, reward, utils in agent.run(n_episodes=1, verbose=(verbose > 0)):

        print(f"  Episode {episode_id}: reward={reward:.4f}, bins={len(utils)}")

    

    # === UTILISER env.final_packers ===

    print(f"\n[...] Extraction des positions...")

    

    if not hasattr(env, 'final_packers') or not env.final_packers:

        print("[X] Aucun packer sauvegardé dans env.final_packers")

        return 1

    

    print(f"  DEBUG: env.final_packers = {len(env.final_packers)} bins")

    item_count = 0

    

    for bin_idx, packer in enumerate(env.final_packers):

        if not hasattr(packer, 'items'):

            print(f"  [X][X] Bin {bin_idx}: pas d'attribut 'items'")

            continue



        print(f"  Bin {bin_idx}: {len(packer.items)} items")



        # Mémoire des cartons déjà validés dans ce bin (ordre de pose)

        placed_local = []  # éléments: {"x","y","z_base","w","h","d"}



        for item in packer.items:

            if not hasattr(item, 'position'):

                print(f"    [X][X] Item sans position, ignoré")

                continue



            x, y, z = item.position

            

            # === OBTENIR LES DIMENSIONS (CRITIQUE POUR LA HAUTEUR) ===

            if hasattr(item, 'get_dimension') and callable(item.get_dimension):

                w, h, d = item.get_dimension()

            elif hasattr(item, 'width'):

                w = item.width

                h = item.height

                d = item.depth  # [X] HAUTEUR DU CARTON

            else:

                w, h, d = 300, 300, 300

            

            x, y, z = to_py_scalar(x), to_py_scalar(y), to_py_scalar(z)

            w, h, d = to_py_scalar(w), to_py_scalar(h), to_py_scalar(d)



            # --- Correction anti-entrelacement (valide pour TOUS les cartons) ---

            z_required = 0.0

            for prev in placed_local:

                overlap_x = (x < prev["x"] + prev["w"]) and (prev["x"] < x + w)

                overlap_y = (y < prev["y"] + prev["h"]) and (prev["y"] < y + h)

                # Empile seulement si chevauchement XY

                if overlap_x and overlap_y and z < prev["z_base"] + prev["d"]:

                    z_required = max(z_required, prev["z_base"] + prev["d"])



            z_corr = max(z, z_required)



            # Borne à la hauteur du bin si nécessaire

            _, _, BIN_Z = PALLET_CONFIG.bin_size

            if z_corr + d > BIN_Z:

                if verbose > 0:

                    print(f"    [X][X] Clamp hauteur: z+d={z_corr+d:.1f}>{BIN_Z}, on borne.")

                z_corr = max(0.0, BIN_Z - d)



            # Conversion local [X] absolu avec z corrigé

            abs_x, abs_y, abs_z_bas = PALLET_CONFIG.local_to_absolute(x, y, z_corr)



            # Le gripper va au sommet du carton

            abs_z_gripper = abs_z_bas + d



            # Offsets DELMIA

            abs_x += OFFSET_X

            abs_y += OFFSET_Y

            abs_z_gripper += OFFSET_Z



            rows.append({

                "x": float(abs_x),

                "y": float(abs_y),

                "z": float(abs_z_gripper),

                "rx": 0.0,

                "ry": RY_FOR_DELMIA,

                "rz": 0.0

            })



            if verbose > 0:

                print(f"    Item {item_count+1}: local=({x:.0f},{y:.0f},{z_corr:.0f}) "

                      f"dim=({w:.0f}×{h:.0f}×{d:.0f}) [X] z_gripper={abs_z_gripper:.1f}mm")



            # Mémoriser ce carton pour les suivants

            placed_local.append({"x": x, "y": y, "z_base": z_corr, "w": w, "h": h, "d": d})

            placed_all.append({"x": x, "y": y, "z_base": z_corr, "w": w, "h": h, "d": d})

            item_count += 1

    if not rows:

        print("[X] Aucun résultat (packer.items vide après placement).")

        return 1



    df = pd.DataFrame(rows, columns=["x", "y", "z", "rx", "ry", "rz"])



    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_excel(output_file, index=False)

    

    print(f"\n[X] Excel créé : {output_file}")

    print(f"  - Total cartons : {len(df)}")

    print(f"\n[...] STATISTIQUES:")

    print(f"  z_min (plus bas gripper) : {df['z'].min():.1f} mm")

    print(f"  z_max (plus haut gripper): {df['z'].max():.1f} mm")

    print(f"  Hauteur span             : {df['z'].max() - df['z'].min():.1f} mm")



    # Validation chevauchement 3D

    print("\n[...] Validation chevauchements 3D (locale avant offsets):")

    _validate_no_overlap(placed_all)



    # === RELOCALISATION OPTIONNELLE POUR CHEVAUCHEMENTS XY (tous niveaux) ===

    MIN_GAP = 50  # mm écart visuel minimal

    def _xy_overlap(a, b):

        # Séparations sur X/Y (0 si chevauchement)

        sep_x = 0

        if a['x'] + a['w'] <= b['x']:

            sep_x = b['x'] - (a['x'] + a['w'])

        elif b['x'] + b['w'] <= a['x']:

            sep_x = a['x'] - (b['x'] + b['w'])

        sep_y = 0

        if a['y'] + a['h'] <= b['y']:

            sep_y = b['y'] - (a['y'] + a['h'])

        elif b['y'] + b['h'] <= a['y']:

            sep_y = a['y'] - (b['y'] + b['h'])

        # Conflit si chevauchement (sep=0) OU si les deux séparations sont < MIN_GAP

        return sep_x < MIN_GAP and sep_y < MIN_GAP



    level_tol = 1e-6

    bin_w, bin_h, bin_z = PALLET_CONFIG.bin_size

    adjusted = 0

    # Trier items par surface descendante pour stabiliser grands cartons

    order = sorted(range(len(placed_all)), key=lambda i: placed_all[i]['w']*placed_all[i]['h'], reverse=True)

    placed_ordered = [placed_all[i] for i in order]



    for i in range(len(placed_ordered)):

        ai = placed_ordered[i]

        moved = True

        # Répéter tentative tant qu'il y a overlap avec un précédent

        for _ in range(3):  # limiter cycles

            conflict_prev = None

            for j in range(i):

                aj = placed_ordered[j]

                if _xy_overlap(ai, aj):

                    conflict_prev = aj

                    break

            if conflict_prev is None:

                break

            shifted = False

            # Déplacement horizontal vers la droite

            for dx in range(10, 301, 10):

                new_x = ai['x'] + dx

                if new_x + ai['w'] > bin_w:

                    break

                test_ai = {**ai, 'x': new_x}

                if all(not _xy_overlap(test_ai, placed_ordered[k]) for k in range(i)):

                    ai['x'] = new_x

                    shifted = True

                    break

            # Si pas de place à droite, essayer vers le bas (y+)

            if not shifted:

                for dy in range(10, 301, 10):

                    new_y = ai['y'] + dy

                    if new_y + ai['h'] > bin_h:

                        break

                    test_ai = {**ai, 'y': new_y}

                    if all(not _xy_overlap(test_ai, placed_ordered[k]) for k in range(i)):

                        ai['y'] = new_y

                        shifted = True

                        break

            if shifted:

                adjusted += 1

            else:

                # Essayer translation diagonale combinée

                for delta in range(10, 301, 10):

                    new_x = ai['x'] + delta

                    new_y = ai['y'] + delta

                    if new_x + ai['w'] > bin_w or new_y + ai['h'] > bin_h:

                        break

                    test_ai = {**ai, 'x': new_x, 'y': new_y}

                    if all(not _xy_overlap(test_ai, placed_ordered[k]) for k in range(i)):

                        ai['x'] = new_x

                        ai['y'] = new_y

                        adjusted += 1

                        shifted = True

                        break

                if not shifted:

                    # Dernier recours: laisser la position (overlap résiduel) mais ne pas boucler

                    print("[X][X] Overlap résiduel non résolu pour un item (pas d'espace libre).")

                    break



    # Repropager dans placed_all selon ordre initial

    for idx, ord_idx in enumerate(order):

        placed_all[ord_idx] = placed_ordered[idx]



    if adjusted > 0:

        print(f"\n[...] Relocalisation effectuée sur {adjusted} item(s) pour éviter chevauchements XY.")

        # Recalculer lignes Excel pour cohérence des z gripper

        new_rows = []

        for rec in placed_all:

            abs_x, abs_y, abs_z_bas = PALLET_CONFIG.local_to_absolute(rec['x'], rec['y'], rec['z_base'])

            abs_x += OFFSET_X; abs_y += OFFSET_Y

            abs_z_gripper = abs_z_bas + rec['d'] + OFFSET_Z

            new_rows.append({

                "x": abs_x, "y": abs_y, "z": abs_z_gripper,

                "rx": 0.0, "ry": RY_FOR_DELMIA, "rz": 0.0

            })

        df = pd.DataFrame(new_rows, columns=["x","y","z","rx","ry","rz"])

        df.to_excel(output_file, index=False)

        print(f"[X] Excel mis à jour après relocalisation: {output_file}")

    

    return 0



def parse_args():

    p = argparse.ArgumentParser(description="DeepPack3D [X] Excel (coordonnées DELMIA)")

    p.add_argument("--input", "-i", type=Path, default=INPUT_PATH)

    p.add_argument("--output", "-o", type=Path, default=XLSX_PATH)

    p.add_argument("--method", "-m", choices=["rl", "bl", "baf", "bssf", "blsf"], default="rl")

    p.add_argument("--lookahead", "-l", type=int, default=10)

    p.add_argument("--iterations", type=int, default=-1)

    p.add_argument("--verbose", "-v", type=int, default=1)

    return p.parse_args()



def _validate_no_overlap(placed):

    ok = True

    for i, a in enumerate(placed):

        for j, b in enumerate(placed):

            if j <= i: 

                continue

            overlap_x = (a['x'] < b['x'] + b['w']) and (b['x'] < a['x'] + a['w'])

            overlap_y = (a['y'] < b['y'] + b['h']) and (b['y'] < a['y'] + a['h'])

            overlap_z = (a['z_base'] < b['z_base'] + b['d']) and (b['z_base'] < a['z_base'] + a['d'])

            if overlap_x and overlap_y and overlap_z:

                print(f"[X] Chevauchement détecté entre cartons {i+1} et {j+1}")

                ok = False

    if ok:

        print("[X] Validation: aucun chevauchement 3D détecté.")

    return ok



if __name__ == "__main__":

    args = parse_args()

    try:

        exit_code = run_packing(

            args.input, args.output,

            method=args.method,

            lookahead=args.lookahead,

            n_iterations=args.iterations,

            verbose=args.verbose

        )

    except Exception as e:

        print(f"[X] Erreur : {e}")

        import traceback

        traceback.print_exc()

        exit_code = 1

    

    raise SystemExit(exit_code)



