#!/bin/bash
# Single board entry point. Python files are required by the actual XIR/VART runtime.
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP=/opt/oil-spill
STATE=/var/lib/oil-spill
MODE="${1:-run}"
STAGE='initial checks'; LAST_COMMAND=''; CHILD=''; MONITOR=''
fail() {
  trap - ERR
  echo
  echo "STOPPED at: $STAGE"
  echo "Command: $LAST_COMMAND"
  echo "Reason/exit: $*"
  echo 'Photograph this error and the lines just above. Later stages were not run.'
  echo 'Keep the files. Correct the error, then rerun the same RUN.sh command.'
  exit 1
}
trap 'rc=$?; LAST_COMMAND="$BASH_COMMAND"; fail "line $LINENO, exit $rc"' ERR
interrupt() {
  trap - INT TERM ERR
  [ -z "$CHILD" ] || kill "$CHILD" 2>/dev/null || true
  [ -z "$MONITOR" ] || kill "$MONITOR" 2>/dev/null || true
  echo 'Interrupted. Existing checkpoints are retained; rerun RUN.sh to resume.'
  exit 130
}
trap interrupt INT TERM
stage() { STAGE="$1"; printf '\n========== %s ==========\n' "$STAGE"; }
run() {
  local rc=0
  printf -v LAST_COMMAND '%q ' "$@"
  echo "+ $LAST_COMMAND"
  "$@" <&0 &
  CHILD=$!
  (
    while sleep 10; do
      kill -0 "$CHILD" 2>/dev/null || exit 0
      echo "[working] $STAGE - process $CHILD is still running"
    done
  ) &
  MONITOR=$!
  wait "$CHILD" || rc=$?
  kill "$MONITOR" 2>/dev/null || true
  wait "$MONITOR" 2>/dev/null || true
  CHILD=''; MONITOR=''
  [ "$rc" -eq 0 ] || fail "$rc"
  echo "[OK] $STAGE"
}
case "$MODE" in
  help|--help|-h)
    echo 'On FPGA Linux: bash /actual/USB/path/finaldeployment/RUN.sh [mode]'
    echo 'Default: run. Verify, install/update, actual sample, benchmark, all images and export.'
    echo 'run: checks/inference/export without changing automatic startup configuration.'
    echo 'fresh: run again, scanning all pixels instead of reusing completed maps.'
    echo 'autostart: explicitly run the same workflow, then enable future automatic boots.'
    echo 'boot: internal service mode; actual sample baseline check and image queue resume.'
    echo 'status: display saved status. export DEST: copy results into a NEW folder.'
    exit 0;;
  run|fresh|autostart|boot|status|export) ;;
  *) echo "Unknown mode: $MODE. Use RUN.sh help"; exit 64;;
esac
[ "$(uname -s)" = Linux ] || fail 'Run on the FPGA Linux terminal, not Windows.'
[ "$(id -u)" -eq 0 ] || fail 'Root required. Use sudo -E bash /actual/path/RUN.sh'
case "$SOURCE/" in */inference-new/*) fail 'Protected inference-new folder';; esac
if [[ "$MODE" =~ ^(boot|status|export)$ ]]; then
  [ -f "$APP/runtime.env" ] || fail 'Install first: run RUN.sh from the USB.'
  source "$APP/runtime.env"
  PYTHON="${PYTHON_BIN:?}"
else
  PYTHON="${2:-$(command -v python3)}"
fi
[[ "$PYTHON" = /* && -x "$PYTHON" ]] || fail 'An executable absolute Python path is required.'
command -v timeout >/dev/null
if [ "$MODE" = status ]; then
  stage 'saved status'
  run "$PYTHON" "$APP/deploy.py" status
  run "$PYTHON" "$APP/queue_app.py" status
  exit 0
fi
if [ "$MODE" = export ]; then
  DEST="${2:?Supply a new destination folder after export}"
  case "$DEST/" in */inference-new/*) fail 'Protected destination';; esac
  stage 'export existing results'
  run bash "$APP/export_results.sh" "$DEST"
  exit 0
fi
stage '1/9 - runtime and storage'
run mkdir -p /opt /var/lib "$STATE/runs"
LOG="$STATE/runs/run-$(date -u +%Y%m%dT%H%M%S)-$$.log"
exec > >(tee -a "$LOG") 2>&1
echo "Full terminal log: $LOG"
echo 'Board calendar may be incorrect; inference durations use monotonic time.'
run "$PYTHON" -c 'import sys,numpy,xir,vart; assert sys.version_info >= (3,8); print("Real board runtime:",sys.executable,sys.version,"NumPy",numpy.__version__,flush=True)'
run "$PYTHON" - "$SOURCE" "$APP" "$STATE" <<'PY'
import json,shutil,sys
from pathlib import Path
root,app,state=map(Path,sys.argv[1:])
manifest=json.loads((root/'release.json').read_text())
queue=json.loads((root/'scene_queue.json').read_text())
copy_bytes=0 if root.resolve()==app.resolve() else sum((root/n).stat().st_size for n in manifest['sha256'])
output_bytes=100*1024*1024
for item in queue['scenes']:
    directory=(root/item['directory']).resolve(); directory.relative_to(root.resolve())
    h,w=json.loads((directory/'scene.json').read_text())['prepared_shape'][:2]
    output_bytes+=h*w*200
    print('Queued image:',item['source_name'],h,w,flush=True)
if not queue['scenes']: raise RuntimeError('No prepared images')
same=app.parent.stat().st_dev==state.stat().st_dev
if shutil.disk_usage(app.parent).free < copy_bytes+(output_bytes if same else 0):
    raise RuntimeError('Insufficient application storage')
if not same and shutil.disk_usage(state).free < output_bytes:
    raise RuntimeError('Insufficient result storage')
print('Persistent storage budget checked.',flush=True)
PY
stage '2/9 - verify packaged files and real images'
run "$PYTHON" "$SOURCE/deploy.py" verify
stage '3/9 - install or repair this deployment, preserving model and images'
if [ "$MODE" != boot ] && command -v systemctl >/dev/null && systemctl is-active --quiet oil-spill.service; then
  DESCRIPTION="$(systemctl show oil-spill.service -p Description --value)"
  [[ "$DESCRIPTION" = 'PlanetSpark hyperspectral'* ]] || fail 'Unrelated service already uses oil-spill.service'
  run systemctl stop oil-spill.service
fi
run "$PYTHON" - "$SOURCE" "$APP" <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
source,app=map(Path,sys.argv[1:]); source=source.resolve(); app=app.resolve()
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
if source==app:
    print('Installed deployment; no files copied.',flush=True); sys.exit(0)
incoming=json.loads((source/'release.json').read_text())
if app.exists() and any(app.iterdir()):
    old=json.loads((app/'release.json').read_text())
    if old.get('application')!='whole-scene-v2': raise RuntimeError('Unrecognized installation; nothing replaced')
    # A reviewed release may change queue membership; model and prepared-scene bytes remain immutable.
    for name,expected in old['sha256'].items():
        if name.endswith('.xmodel') or name.startswith(('test_patch.','scene_input/','scene_jobs/')):
            if incoming['sha256'].get(name)!=expected: raise RuntimeError('Refusing to replace model/input: '+name)
app.mkdir(parents=True,exist_ok=True)
for name,expected in incoming['sha256'].items():
    src=(source/name).resolve(); src.relative_to(source)
    dst=(app/name).resolve(); dst.relative_to(app)
    if dst.is_file() and digest(dst)==expected:
        print('Already verified:',name,flush=True); continue
    dst.parent.mkdir(parents=True,exist_ok=True)
    print('Copying:',name,flush=True)
    shutil.copyfile(str(src),str(dst)); dst.chmod(0o644)
    if digest(dst)!=expected: raise RuntimeError('Copy verification failed: '+name)
for name in ('README.md','LAB_GUIDE.md','VALIDATION.md'):
    if (source/name).is_file(): shutil.copyfile(str(source/name),str(app/name))
shutil.copyfile(str(source/'release.json'),str(app/'release.json'))
print('Installation synchronized; no files deleted.',flush=True)
PY
if [ "$MODE" != boot ]; then
  run "$PYTHON" - "$PYTHON" <<'PY'
import os,shlex,sys
from pathlib import Path
lines=['export PYTHON_BIN='+shlex.quote(sys.argv[1])]
for n in ('PATH','LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME','XILINX_XRT','XILINX_VART_FIRMWARE'):
    if n in os.environ: lines.append('export '+n+'='+shlex.quote(os.environ[n]))
path=Path('/opt/oil-spill/runtime.env'); path.write_text('\n'.join(lines)+'\n'); path.chmod(0o644)
PY
fi
source "$APP/runtime.env"
if [ "$SOURCE" != "$APP" ]; then run "$PYTHON" "$APP/deploy.py" verify; fi
stage '4/9 - find or load the actual DPU'
run timeout 180 "$PYTHON" - "$APP" <<'PY'
import runpy,sys
from pathlib import Path
ns=runpy.run_path(str(Path(sys.argv[1])/'deploy.py'))
hardware=ns['boot_query'](True)
print(hardware,flush=True)
if not ns['fingerprints'](hardware): raise RuntimeError('No real DPU fingerprint')
PY
stage '5/9 - execute the real known sample on the FPGA'
if [ -f "$STATE/reference.json" ]; then
  run timeout 180 "$PYTHON" "$APP/deploy.py" boot --load-dpu
else
  [ "$MODE" != boot ] || fail 'Boot baseline absent; finish a manual run first.'
  run timeout 180 "$PYTHON" "$APP/deploy.py" run --load-dpu
fi
stage '6/9 - repeatability and real latency measurement'
if [ ! -f "$STATE/reference.json" ]; then
  run "$PYTHON" "$APP/deploy.py" approve-reference
  run timeout 180 "$PYTHON" "$APP/deploy.py" boot --load-dpu
fi
if [ "$MODE" != boot ]; then
  run timeout 180 "$PYTHON" "$APP/deploy.py" benchmark --load-dpu --warmup 20 --iterations 200
else
  echo 'Boot sample checked against the saved board baseline; benchmark is a manual step.'
fi
stage '7/9 - real oil/water inference for every queued image'
echo 'Watch IMAGE n/total and row progress. Large images take time; predictions require the actual DPU.'
if [ "$MODE" = fresh ]; then
  run "$PYTHON" "$APP/queue_app.py" run --load-dpu --fresh
else
  run "$PYTHON" "$APP/queue_app.py" run --load-dpu
fi
stage '8/9 - automatic boot configuration'
if [ "$MODE" = autostart ]; then
  run timeout 180 env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /bin/bash -c 'set -euo pipefail; source /opt/oil-spill/runtime.env; "$PYTHON_BIN" /opt/oil-spill/deploy.py boot --load-dpu'
  command -v systemctl >/dev/null
  test -d /run/systemd/system
  if systemctl cat oil-spill.service >/dev/null 2>&1; then
    DESCRIPTION="$(systemctl show oil-spill.service -p Description --value)"
    [[ "$DESCRIPTION" = 'PlanetSpark hyperspectral'* ]] || fail 'Refusing to replace an unrelated service'
  fi
  run chmod 0755 "$APP/start.sh"
  if command -v systemd-analyze >/dev/null; then run systemd-analyze verify "$APP/oil-spill.service"; fi
  run install -m 0644 "$APP/oil-spill.service" /etc/systemd/system/oil-spill.service
  DPU_UNITS=''
  for unit in dfx-mgr.service dfx-mgrd.service; do
    if systemctl cat "$unit" >/dev/null 2>&1; then DPU_UNITS="$DPU_UNITS $unit"; fi
  done
  if [ -n "$DPU_UNITS" ]; then
    run mkdir -p /etc/systemd/system/oil-spill.service.d
    printf '# finaldeployment detected DPU-manager ordering\n[Unit]\nAfter=%s\nWants=%s\n' "$DPU_UNITS" "$DPU_UNITS" > /etc/systemd/system/oil-spill.service.d/20-dpu-manager.conf
  fi
  run systemctl daemon-reload
  LOADED="$(systemctl show oil-spill.service -p LoadState --value)"
  [ "$LOADED" = loaded ] || fail "Service load state is $LOADED"
  run systemctl enable oil-spill.service
  echo 'Autostart enabled for NEXT boot. A real cold-boot test is still required.'
else
  echo 'Autostart configuration unchanged. Use RUN.sh autostart after a successful manual run.'
fi
stage '9/9 - final status and results'
run "$PYTHON" "$APP/queue_app.py" status
if [ "$MODE" != boot ] && [ "$SOURCE" != "$APP" ]; then
  DEST="$SOURCE/results-$(date -u +%Y%m%dT%H%M%S)-$$"
  run bash "$APP/export_results.sh" "$DEST"
  echo "USB results: $DEST"
  echo 'On the laptop open scenes, then the scene job folder, then report.html.'
else
  echo "Results retained on the board: $STATE/scenes"
fi
echo 'ALL REQUIRED STEPS PASSED for this run. Repeatability is not independent held-out accuracy.'
echo "Log: $LOG"
