import re
import argparse
from pathlib import Path

PAT = re.compile(r"(np\.meshgrid\()\s*(.*?)\)", re.DOTALL)

def process_text(text):
    changed = False
    def repl(m):
        nonlocal changed
        args = m.group(2)
        if "indexing=" in args:
            return m.group(0)
        changed = True
        return f"{m.group(1)}{args}, indexing='ij')"
    new = PAT.sub(repl, text)
    return new, changed

def scan_files(root):
    py_files = list(Path(root).rglob("*.py"))
    report = []
    for p in py_files:
        txt = p.read_text(encoding='utf-8', errors='ignore')
        new, changed = process_text(txt)
        if changed:
            report.append(p)
    return report

def apply_patch(path):
    p = Path(path)
    txt = p.read_text(encoding='utf-8', errors='ignore')
    new, changed = process_text(txt)
    if changed:
        p.write_text(new, encoding='utf-8')
    return changed

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="project root")
    ap.add_argument("--apply", action="store_true", help="apply changes")
    args = ap.parse_args()

    files = scan_files(args.root)
    if not files:
        print("Aucune occurrence modifiable de np.meshgrid trouvée.")
    else:
        print("Fichiers concernés:")
        for f in files:
            print(" -", f)
    if args.apply and files:
        for f in files:
            ok = apply_patch(f)
            print(("MODIFIE" if ok else "RIEN A FAIRE"), f)
        print("Patch appliqué. Fais un commit si OK.")
    else:
        print("Mode dry-run (aucun fichier modifié). Pour appliquer: python tools\\fix_meshgrid_indexing.py --apply")