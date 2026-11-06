#!/usr/bin/env python3
"""Run/import queues of real HSI scenes. Model and buffers are reused across scenes."""
import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from deploy import ROOT,boot_id,now,sha,verify,write_json
from hsi_contract import CONTRACT
from scene import run as run_scene,verify_input

def read_queue(path,check_inputs=True):
    path=Path(path).resolve()
    queue=json.loads(path.read_text())
    if queue['contract']!=CONTRACT or not queue['scenes']:
        raise ValueError('Empty or unsupported HSI queue')
    entries=[]
    for item in queue['scenes']:
        relative=Path(item['directory'])
        if relative.is_absolute():raise ValueError('Input queue paths must be relative')
        directory=(path.parent/relative).resolve()
        directory.relative_to(path.parent)
        if check_inputs:verify_input(directory)
        entries.append((item['source_name'],directory))
    return queue,entries

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','import','validate','status'])
    parser.add_argument('--input-queue',type=Path)
    parser.add_argument('--state-dir',type=Path,default=Path('/var/lib/oil-spill'))
    parser.add_argument('--load-dpu',action='store_true')
    parser.add_argument('--fresh',action='store_true')
    args=parser.parse_args()
    state=args.state_dir
    state.mkdir(parents=True,exist_ok=True)
    progress={'status':'RUNNING','started_utc':now()}
    try:
        if args.action=='status':
            progress=json.loads((state/'queue-latest.json').read_text())
            print(json.dumps(progress,indent=2));return 1 if progress['status']=='FAIL' else 0
        release=verify(include_scene=False)
        if args.action=='import':
            if not args.input_queue:raise ValueError('Supply --input-queue /USB/path/queue.json')
            payload,entries=read_queue(args.input_queue)
            identity=sha(args.input_queue)
            target=state/'inbox'/('queue-'+identity[:16])
            if target.exists():
                read_queue(target/'queue.json')
                print('This exact prepared queue is already imported:',target);return 0
            needed=100*1024*1024
            for _,directory in entries:
                metadata=json.loads((directory/'scene.json').read_text())
                h,w=metadata['prepared_shape'][:2]
                needed+=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())+h*w*200
            if shutil.disk_usage(str(state)).free<needed:
                raise RuntimeError('Insufficient persistent storage for all imported inputs and their output caches/maps')
            temporary=target.with_name(target.name+'.partial')
            if temporary.exists():raise FileExistsError('Incomplete import remains at '+str(temporary)+'; preserve it and choose a fresh prepared queue')
            temporary.mkdir(parents=True)
            imported=[]
            for index,(name,directory) in enumerate(entries,1):
                destination=temporary/('scene-'+str(index).zfill(4))
                shutil.copytree(str(directory),str(destination))
                verify_input(destination)
                imported.append({'directory':destination.name,'source_name':name})
            write_json(temporary/'queue.json',{'contract':CONTRACT,'scenes':imported})
            temporary.rename(target)
            print('Imported {} image(s) onto persistent storage. Restart the service to analyse them.'.format(len(entries)))
            return 0
        if args.input_queue:
            paths=[args.input_queue]
        else:
            default=ROOT/'scene_queue.json'
            if sha(default)!=release['sha256']['scene_queue.json']:
                raise ValueError('Default boot queue differs from sealed release')
            paths=[default]+sorted((state/'inbox').glob('queue-*/queue.json'))
        entries=[]
        for path in paths:
            _,items=read_queue(path,check_inputs=args.action=='validate')
            entries.extend(items)
        if args.action=='validate':
            print('Validated {} actual prepared HSI input(s); inference not performed.'.format(len(entries)));return 0
        progress.update(boot_id=boot_id(),total_scenes=len(entries),completed_scenes=0,results=[])
        engine=None
        for index,(name,directory) in enumerate(entries,1):
            progress.update(current_scene=name,current_scene_number=index,updated_utc=now())
            write_json(state/'queue-latest.json',progress)
            print('IMAGE {}/{}: {}'.format(index,len(entries),name),flush=True)
            engine=run_scene(SimpleNamespace(input_dir=directory,state_dir=state,output=None,
                stride=1,load_dpu=args.load_dpu,fresh=args.fresh),engine=engine)
            result=json.loads((state/'scene-latest.json').read_text())
            progress['results'].append({'source_name':name,'mode':result['mode'],'output_directory':result['output_directory']})
            progress['completed_scenes']=index
            write_json(state/'queue-latest.json',progress)
        progress.update(status='PASS',finished_utc=now())
        write_json(state/'queue-latest.json',progress)
        print('ALL INPUT IMAGES COMPLETE. Separate maps/reports are saved for each scene.');return 0
    except Exception as exc:
        progress.update(status='FAIL',error='{}: {}'.format(type(exc).__name__,exc),finished_utc=now())
        write_json(state/'queue-latest.json',progress)
        print(json.dumps(progress,indent=2));return 1

if __name__=='__main__':sys.exit(main())
