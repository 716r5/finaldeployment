#!/usr/bin/env python3
"""PlanetSpark deployment; actual XIR/VART inference only. Python 3.8+."""
import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ('oil_spill_cnn_138.xmodel', 'test_patch.npy', 'test_patch.json')
CODE = ('RUN.sh', 'deploy.py', 'install.sh', 'uninstall.sh', 'oil-spill.service',
        'scene.py', 'queue_app.py', 'prepare_scene.py', 'prepare_queue.py', 'hsi_contract.py', 'regions.py', 'start.sh', 'export_results.sh')
SCENE = ('scene_input/scene.json','scene_input/cube.npy','scene_input/preview.png','scene_input/input_manifest.json')

def sha(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(str(temp), str(path))

def now():
    return datetime.now(timezone.utc).isoformat()

def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()

def shell(argv):
    p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=15)
    if p.returncode:
        raise RuntimeError('{} returned {}: {}'.format(argv, p.returncode, p.stdout))
    return p.stdout

def verify(include_scene=True):
    manifest = json.loads((ROOT/'release.json').read_text(encoding='utf-8'))
    for name in ARTIFACTS + CODE + (SCENE if include_scene else ()):
        if name not in manifest['sha256']:
            raise RuntimeError('Required file absent from release manifest: '+name)
    for name, expected in manifest['sha256'].items():
        if not include_scene and (name=='scene_queue.json' or name.startswith(('scene_input/','scene_jobs/'))):
            continue
        if sha(ROOT/name) != expected:
            raise RuntimeError('Missing/changed release file: '+name)
    return manifest

def seal():
    """User explicitly assembles their reviewed files; no USB is accessed automatically."""
    target = ROOT/'release.json'
    if target.exists():
        raise RuntimeError('release.json already exists; do not silently reseal a changed release')
    queue=json.loads((ROOT/'scene_queue.json').read_text())
    if not queue['scenes']:
        raise RuntimeError('No real HSI scenes were prepared')
    scene_files=['scene_queue.json']
    for item in queue['scenes']:
        relative=Path(item['directory'])
        directory=(ROOT/relative).resolve()
        directory.relative_to(ROOT.resolve())
        scene_files.extend(str((relative/name).as_posix()) for name in ('cube.npy','preview.png','scene.json','input_manifest.json'))
        if (directory/'labels.npy').exists(): scene_files.append((relative/'labels.npy').as_posix())
    names=ARTIFACTS+CODE+tuple(scene_files)
    manifest = {'created_utc':now(), 'application':'whole-scene-v2', 'model_target':'DPUCZDX8G_ISA1_B4096',
                'scene_files':scene_files,'sha256':{name:sha(ROOT/name) for name in names}}
    write_json(target, manifest)
    return manifest

def query(load):
    output = ''
    try:
        output = shell(['xdputil','query'])
    except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired):
        if not load:
            raise
    if fingerprints(output):
        return output
    if not load:
        raise RuntimeError('No nonzero DPU fingerprint found in xdputil query')
    shell(['dfx-mgr-client','-load','dpu'])
    output = shell(['xdputil','query'])
    if not fingerprints(output):
        raise RuntimeError('DPU still unavailable after loading application dpu')
    return output

def fingerprints(text):
    # xdputil may prefix its JSON output with informational lines.
    values = re.findall(r'fingerprint["\s]*[:=]\s*["\s]*(0x[0-9a-fA-F]+|[0-9]+)', text, re.I)
    return {int(v,16) if v.lower().startswith('0x') else int(v) for v in values} - {0}

def boot_query(load):
    for attempt in range(3):
        try:
            return query(load)
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            if attempt == 2:
                raise
            time.sleep(2)

def load_patch(path, expected, np):
    patch = np.load(str(path), allow_pickle=False)
    if patch.dtype.kind not in 'fiu' or not np.isfinite(patch).all():
        raise ValueError('Patch must contain finite numeric values')
    if patch.shape == (138,11,11):
        patch = patch[None, ...]
    if tuple(patch.shape) not in ((1,138,11,11),(1,11,11,138)):
        raise ValueError('Expected prepared patch shape (138,11,11), (1,138,11,11) or (1,11,11,138)')
    if patch.min() < 0 or patch.max() > 1:
        raise ValueError('Patch must already be normalized to [0,1]; this runner never renormalizes it')
    if tuple(patch.shape) != expected:
        patch = np.transpose(patch, (0,2,3,1) if expected[-1] == 138 else (0,3,1,2))
    return np.ascontiguousarray(patch, dtype=np.float32)

def validate_output_conversion(model_sha256, attributes, output_fix_point):
    """Approve only the terminal conversion inspected in this exact compiled model.

    The known graph ends with signed INT8 -> float conversion. execute() performs
    that same multiplication on the host. This is not a general CPU-op bypass.
    """
    known_model = '6985462767744b40c760ed1fff075c3c350d281aff4d705253e52fb010a8c3e4'
    if model_sha256 != known_model:
        raise RuntimeError('Terminal fix2float graph has not been inspected for this model hash')
    if int(attributes['bit_width']) != 8 or not bool(attributes['if_signed']):
        raise RuntimeError('Terminal fix2float must use signed 8-bit input')
    if int(attributes['fix_point']) != int(output_fix_point):
        raise RuntimeError('Terminal fix2float scale differs from the DPU output tensor')
    if int(attributes['fix_point']) != 3:
        raise RuntimeError('Terminal fix2float differs from the inspected compiled graph')

class Engine:
    def __init__(self, model, hardware):
        import numpy as np
        import xir
        import vart
        self.np = np
        self.graph = xir.Graph.deserialize(str(model))
        children = self.graph.get_root_subgraph().toposort_child_subgraph()
        selected = []
        output_conversions = []
        self.coverage = []
        for child in children:
            device = str(child.get_attr('device')).upper() if child.has_attr('device') else 'UNKNOWN'
            types = sorted(set(op.get_type() for op in child.get_ops()))
            self.coverage.append({'name':child.get_name(),'device':device,'op_types':types})
            if device == 'DPU':
                selected.append(child)
            elif device == 'CPU' and types == ['fix2float'] and len(child.get_ops()) == 1:
                output_conversions.append((next(iter(child.get_ops())), self.coverage[-1]))
            # XIR data/data-fix are caller-fed input placeholders, not CPU layers.
            elif not types or any(t not in ('data', 'data-fix') for t in types):
                raise RuntimeError('Unexecuted non-DPU operations: '+str(self.coverage[-1]))
        if len(selected) != 1:
            raise RuntimeError('Expected one DPU subgraph; found '+str(len(selected)))
        sg = selected[0]
        if not sg.has_attr('dpu_fingerprint'):
            raise RuntimeError('Missing model dpu_fingerprint; cannot verify compatibility')
        fingerprint = int(sg.get_attr('dpu_fingerprint'))
        if fingerprint not in fingerprints(hardware):
            raise RuntimeError('Model fingerprint {} not present in board query {}'.format(hex(fingerprint),hardware))
        self.fingerprint = hex(fingerprint)
        self.runner = vart.Runner.create_runner(sg,'run')
        inputs, outputs = self.runner.get_input_tensors(), self.runner.get_output_tensors()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError('Expected one input and one output')
        self.input, self.output = inputs[0], outputs[0]
        self.shape = tuple(int(x) for x in self.input.dims)
        if self.shape not in ((1,138,11,11),(1,11,11,138)):
            raise RuntimeError('Unexpected model input shape '+str(self.shape))
        if math.prod(int(x) for x in self.output.dims) != 1:
            raise RuntimeError('Expected exactly one output logit')
        for tensor in (self.input,self.output):
            if not tensor.has_attr('fix_point'):
                raise RuntimeError('Missing fix_point on '+tensor.name)
            dtype = str(tensor.dtype).upper()
            if 'INT8' not in dtype or 'UINT' in dtype:
                raise RuntimeError('Expected signed INT8 tensor, found '+dtype)
        self.output_buffer = np.empty(tuple(self.output.dims),dtype=np.int8,order='C')
        self.output_step = 2.0**(-int(self.output.get_attr('fix_point')))
        if len(output_conversions) > 1:
            raise RuntimeError('Expected at most one terminal fix2float conversion')
        for operation, coverage in output_conversions:
            keys = ('fix_point', 'bit_width', 'if_signed')
            if not all(operation.has_attr(key) for key in keys):
                raise RuntimeError('Terminal fix2float is missing conversion metadata')
            attributes = {key: operation.get_attr(key) for key in keys}
            validate_output_conversion(sha(model), attributes, self.output.get_attr('fix_point'))
            coverage['handled_by'] = 'Host signed-INT8 dequantization, checked against the exact model graph'

    def prepare(self, patch):
        np = self.np
        values = load_patch(patch,self.shape,np)
        scaled = np.rint(values * 2.0**int(self.input.get_attr('fix_point')))
        return np.ascontiguousarray(np.clip(scaled,-128,127),dtype=np.int8)

    def execute(self, values):
        begin = time.perf_counter()
        job = self.runner.execute_async([values],[self.output_buffer])
        status = self.runner.wait(job)
        elapsed = (time.perf_counter()-begin)*1000
        if status != 0:
            raise RuntimeError('VART wait returned '+str(status))
        logit = float(self.output_buffer.reshape(-1)[0])*self.output_step
        prob = 1/(1+math.exp(-logit)) if logit >= 0 else math.exp(logit)/(1+math.exp(logit))
        return {'logit':logit,'oil_probability':prob,'prediction':'oil' if logit>=0 else 'water','execution_ms':elapsed}

def baseline_check(result, reference):
    for key in ('model_sha256','input_sha256'):
        if result[key] != reference[key]:
            raise RuntimeError('Reference identity mismatch: '+key)
    if result['prediction'] != reference['prediction'] or abs(result['logit']-reference['logit']) > reference['tolerance']:
        raise RuntimeError('Output differs from saved board baseline')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['seal','verify','doctor','run','boot','benchmark','status','approve-reference','evidence'])
    parser.add_argument('--patch',type=Path,default=ROOT/'test_patch.npy')
    parser.add_argument('--state-dir',type=Path,default=Path('/var/lib/oil-spill'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--load-dpu',action='store_true')
    parser.add_argument('--warmup',type=int,default=20)
    parser.add_argument('--iterations',type=int,default=200)
    args = parser.parse_args()
    state = args.state_dir
    result = {'started_utc':now(),'action':args.action,'status':'FAIL'}
    started = time.perf_counter()
    output = None if args.action=='evidence' else args.output
    if output is None and args.action in ('run','boot','benchmark','doctor'):
        output = state/('benchmark.json' if args.action=='benchmark' else 'doctor.json' if args.action=='doctor' else 'latest.json')
    try:
        if args.action=='seal':
            result['manifest'] = seal()
        elif args.action=='status':
            result = json.loads((state/'latest.json').read_text())
            result['current_boot_id'] = boot_id()
            if result.get('boot_id') != result['current_boot_id']:
                raise RuntimeError('Latest result belongs to a previous boot')
            if result.get('status') != 'PASS':
                raise RuntimeError('Latest inference failed')
        else:
            manifest = verify(include_scene=args.action in ('verify','evidence'))
            result['release_manifest_sha256'] = sha(ROOT/'release.json')
            if args.action=='approve-reference':
                previous = json.loads((state/'latest.json').read_text())
                if (state/'reference.json').exists():
                    raise RuntimeError('Reference already exists; preserve it before deliberately replacing')
                if previous['status']!='PASS' or not previous.get('ground_truth_match'):
                    raise RuntimeError('Require successful packaged-patch inference agreeing with its ground truth')
                for key,name in (('model_sha256',ARTIFACTS[0]),('input_sha256',ARTIFACTS[1])):
                    if previous[key] != manifest['sha256'][name]:
                        raise RuntimeError('Latest result belongs to different artifacts')
                reference = {key:previous[key] for key in ('model_sha256','input_sha256','prediction','logit')}
                reference.update(tolerance=previous['output_step'],created_utc=now(),
                    meaning='Repeatability baseline from real board execution; not held-out accuracy or independent numerical validation')
                write_json(state/'reference.json',reference)
                result['reference'] = reference
            elif args.action=='evidence':
                import tarfile
                if not args.output:
                    raise ValueError('Supply --output /path/evidence.tar.gz')
                args.output.parent.mkdir(parents=True,exist_ok=True)
                with tarfile.open(str(args.output),'x:gz') as archive:
                    archive.add(str(ROOT/'release.json'),arcname='release.json')
                    for path in sorted(state.rglob('*')):
                        if path.is_file() and path.resolve()!=args.output.resolve():
                            archive.add(str(path),arcname='state/'+str(path.relative_to(state)))
                output = None
                result['archive'] = str(args.output)
            elif args.action!='verify':
                if os.environ.get('XLNX_ENABLE_FINGERPRINT_CHECK')=='0' or os.environ.get('DEBUG_DPU_RUNNER_DRY_RUN','0')!='0':
                    raise RuntimeError('Disabled fingerprint checks or DPU dry-run settings are prohibited')
                reference = None
                if args.action=='boot':
                    reference = json.loads((state/'reference.json').read_text())
                    for key,name in (('model_sha256',ARTIFACTS[0]),('input_sha256',ARTIFACTS[1])):
                        if reference[key]!=manifest['sha256'][name]:
                            raise RuntimeError('Boot reference identity mismatch')
                hardware = boot_query(args.load_dpu) if args.action=='boot' else query(args.load_dpu)
                result.update(dpu_query=hardware,python=sys.version,platform=platform.platform())
                engine = Engine(ROOT/ARTIFACTS[0],hardware)
                result.update(model_sha256=sha(ROOT/ARTIFACTS[0]),input_sha256=sha(args.patch),
                    fingerprint=engine.fingerprint,input_shape=list(engine.shape),
                    output_shape=[int(x) for x in engine.output.dims],output_step=engine.output_step,
                    graph_coverage=engine.coverage,numpy_version=engine.np.__version__)
                prep = time.perf_counter()
                values = engine.prepare(args.patch)
                result['patch_preparation_ms']=(time.perf_counter()-prep)*1000
                if args.action=='benchmark':
                    if args.warmup<0 or args.iterations<1:
                        raise ValueError('Require warmup >= 0, iterations >= 1')
                    for _ in range(args.warmup): engine.execute(values)
                    times=[]
                    begin=time.perf_counter()
                    for _ in range(args.iterations): times.append(engine.execute(values)['execution_ms'])
                    duration=time.perf_counter()-begin
                    result.update(warmup=args.warmup,iterations=args.iterations,
                        execution_median_ms=float(engine.np.median(times)),
                        execution_p95_ms=float(engine.np.percentile(times,95)),
                        sustained_patches_per_second=args.iterations/duration,
                        timing_scope='One prepared patch; reused single runner; throughput includes loop and score decoding')
                elif args.action!='doctor':
                    result.update(engine.execute(values))
                    result['reference_check']='NOT_ESTABLISHED'
                    if reference is not None:
                        baseline_check(result,reference)
                        result['reference_check']='PASS'
                    if result['input_sha256']==manifest['sha256']['test_patch.npy']:
                        metadata=json.loads((ROOT/'test_patch.json').read_text())
                        result['ground_truth']=metadata['ground_truth_name']
                        result['ground_truth_match']=result['prediction']==metadata['ground_truth_name']
                        if not result['ground_truth_match']:
                            raise RuntimeError('Packaged sample prediction disagrees with ground truth; inspect before installation')
        result['status']='PASS'
    except Exception as exc:
        result.update(status='FAIL',error='{}: {}'.format(type(exc).__name__,exc))
    result['finished_utc']=now()
    result['process_elapsed_ms']=(time.perf_counter()-started)*1000
    try: result['boot_id']=boot_id()
    except OSError: pass
    if output:
        write_json(output,result)
        if args.action in ('run','boot'):
            write_json(state/'history'/('run-'+str(time.time_ns())+'.json'),result)
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='PASS' else 1

if __name__=='__main__':
    sys.exit(main())
