#!/bin/bash
set -euo pipefail
if [ "$#" -ne 1 ]; then echo 'Usage: sudo bash export_results.sh /actual/USB/path/OilSpillResults'; exit 1; fi
DEST="$1"
if [ -e "$DEST" ]; then echo 'Destination already exists. Choose a new name; no existing results overwritten.'; exit 1; fi
test -f /var/lib/oil-spill/scene-latest.json
mkdir -p -- "$(dirname -- "$DEST")"
mkdir -- "$DEST"
source /opt/oil-spill/runtime.env
"$PYTHON_BIN" - "$DEST" <<'PY'
from pathlib import Path
import json,shutil,sys
source=Path('/var/lib/oil-spill')
destination=Path(sys.argv[1])
def validation_scene(path):
    for name in ('summary.json','input_scene.json'):
        metadata=path/name
        if metadata.is_file():
            try:
                if json.loads(metadata.read_text()).get('source_name','').lower()=='gm03.mat':
                    return True
            except (OSError,ValueError):
                pass
    return False
for item in source.iterdir():
    if item.name.endswith('.lock'): continue
    target=destination/item.name
    if item.name=='scenes' and item.is_dir():
        target.mkdir()
        for scene in item.iterdir():
            if scene.is_dir() and validation_scene(scene):
                print('Excluded validation scene from final export:',scene.name)
                continue
            scene_target=target/scene.name
            if scene.is_dir():
                shutil.copytree(str(scene),str(scene_target),ignore=shutil.ignore_patterns('cube.npy','.quantized_cube.npy','.quantized_cube.tmp.npy'))
            else:
                shutil.copy2(str(scene),str(scene_target))
    elif item.is_dir():
        shutil.copytree(str(item),str(target),ignore=shutil.ignore_patterns('cube.npy','.quantized_cube.npy','.quantized_cube.tmp.npy'))
    else:
        shutil.copy2(str(item),str(target))
PY
cp -- /opt/oil-spill/release.json "$DEST/release.json"
cp -- /opt/oil-spill/scene_input/scene.json "$DEST/input_scene.json"
journalctl -u oil-spill.service -b --no-pager > "$DEST/service-current-boot.log" || echo 'Journal export failed; JSON results were copied.'
echo "Results copied to $DEST. On your laptop open scenes, then its job folder, then report.html."
