#!/usr/bin/env python3
"""Choose one or many real HSI images; prepare inputs without running a model."""
import argparse
import sys
import uuid
from pathlib import Path
from deploy import ROOT,write_json
from hsi_contract import CONTRACT
from prepare_scene import prepare

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--additional',action='store_true')
    args=parser.parse_args()
    import tkinter as tk
    from tkinter import filedialog
    root=tk.Tk();root.withdraw()
    filenames=filedialog.askopenfilenames(title='Select one or multiple REAL HSI MAT images (Ctrl-click selects several)',filetypes=[('HSI MATLAB images','*.mat')])
    root.destroy()
    if not filenames:
        print('No images chosen. Nothing changed.');return 1
    try:
        if args.additional:
            destination=ROOT/'additional_inputs'/('queue-'+uuid.uuid4().hex[:12])
            queue_path=destination/'queue.json'
        else:
            if (ROOT/'release.json').exists() or (ROOT/'scene_queue.json').exists():
                raise RuntimeError('Initial inputs already prepared. Use ADD_IMAGES.cmd for additional images')
            destination=ROOT;queue_path=ROOT/'scene_queue.json'
        scenes=[]
        for index,filename in enumerate(filenames):
            if not args.additional and index==0:
                relative=Path('scene_input')
            elif not args.additional:
                relative=Path('scene_jobs')/('scene-'+uuid.uuid4().hex[:12])
            else:
                relative=Path('scene-'+str(index+1).zfill(4))
            print('Preparing real image {}/{}: {}'.format(index+1,len(filenames),Path(filename).name),flush=True)
            metadata=prepare(Path(filename),destination/relative)
            scenes.append({'directory':relative.as_posix(),'source_name':metadata['source_name']})
        write_json(queue_path,{'contract':CONTRACT,'scenes':scenes})
        print('PREPARED {} real HSI image(s). No inference performed.'.format(len(scenes)))
        print('Prepared queue:',queue_path.resolve())
        return 0
    except Exception as exc:
        print('PREPARATION FAILED: {}: {}'.format(type(exc).__name__,exc));return 1

if __name__=='__main__':sys.exit(main())
