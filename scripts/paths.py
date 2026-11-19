"""Folder locations. By default the code expects this layout (both Zenodo zips unpacked side by side):
    <root>/Mocha_code/scripts/      (this folder)
    <root>/Mocha_processed_data/data/   inputs and gridded intermediates
    <root>/Mocha_processed_data/out/    results (re-created when scripts are rerun)
Override with the environment variables MOCHA_DATA and MOCHA_OUT."""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "Mocha_processed_data"))
DATA = os.environ.get("MOCHA_DATA", os.path.join(_ROOT, "data"))
OUTB = os.environ.get("MOCHA_OUT", os.path.join(_ROOT, "out"))
for _d in ("", "step2", "step3", "step4", "final"):
    os.makedirs(os.path.join(OUTB, _d), exist_ok=True)
