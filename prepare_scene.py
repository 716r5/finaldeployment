#!/usr/bin/env python3
"""Prepare a real hyperspectral MAT scene on the laptop. Never runs inference."""
import argparse
import json
import sys
import uuid
from pathlib import Path
import numpy as np
from PIL import Image
from deploy import now, sha, write_json
from hsi_contract import CONTRACT, USABLE_BANDS, normalise_band

ROOT=Path(__file__).resolve().parent

def open_source(path):
    import scipy.io
    try:
        payload=scipy.io.loadmat(str(path),variable_names=['img','map'])
    except NotImplementedError:
        import h5py
        handle=h5py.File(str(path),'r')
        try:
            source=handle['img']
            if source.ndim!=3 or 'MATLAB_class' not in source.attrs:
                raise ValueError('Expected MATLAB v7.3 img with MATLAB dimension metadata')
            shape=tuple(reversed(source.shape))
            labels=np.asarray(handle['map']).T if 'map' in handle else None
            return shape, lambda index:np.asarray(source[index,:,:]).T, labels, handle.close
        except Exception:
            handle.close()
            raise
    if 'img' not in payload:
        raise ValueError('MAT file must contain img (height,width,spectral bands)')
    source=payload['img']
    return source.shape, lambda index:source[:,:,index], payload.get('map'), lambda:None

def prepare(path, output):
    path=Path(path).resolve()
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError('Scene folder already exists; preserve it and use a fresh extracted application folder')
    shape, get_band, labels, close=open_source(path)
    try:
        if len(shape)!=3 or shape[2]<=max(USABLE_BANDS) or min(shape[:2])<11:
            raise ValueError('Expected img shape H,W,C, H/W >=11 and at least 199 original bands; got '+str(shape))
        height,width=shape[:2]
        if labels is not None:
            labels=np.asarray(labels)
            if labels.shape!=(height,width) or not np.isin(labels,[0,1]).all():
                raise ValueError('Optional map must match image dimensions and use only water=0, oil=1')
        output.mkdir(parents=True)
        write_json(output/'preparation.json',{'status':'RUNNING','source':str(path),'started_utc':now()})
        cube=np.lib.format.open_memmap(str(output/'cube.npy'),mode='w+',dtype=np.float32,shape=(height,width,138))
        ranges=[]
        for position,index in enumerate(USABLE_BANDS):
            cube[:,:,position],bounds=normalise_band(get_band(index),np)
            ranges.append(bounds)
            if (position+1)%10==0:
                print('Prepared spectral bands {}/138'.format(position+1),flush=True)
        cube.flush()
        # Display only: a real three-band composite, not a predicted map.
        preview=np.rint(cube[:,:,[100,60,20]]*255).astype(np.uint8)
        Image.fromarray(preview,'RGB').save(str(output/'preview.png'))
        del preview,cube
        if labels is not None:
            np.save(str(output/'labels.npy'),labels.astype(np.uint8),allow_pickle=False)
        source_hash=sha(path)
        metadata={'status':'PREPARED','prepared_utc':now(),'contract':CONTRACT,
            'scene_id':path.stem+'-'+source_hash[:12],'source_name':path.name,'source_sha256':source_hash,
            'original_shape':list(shape),'prepared_shape':[height,width,138],
            'band_indices_zero_based':USABLE_BANDS,'normalization':'per-scene per-band 1st/99th percentile, epsilon=1e-8',
            'normalization_ranges':ranges,'has_labels':labels is not None,
            'label_mapping':{'0':'water','1':'oil'},'preview_selected_channel_positions':[100,60,20],
            'preview_note':'Three-band composite of real normalized scene; not natural RGB or model prediction',
            'test_split_status':'NOT_CONFIRMED_HELD_OUT',
            'border_policy':'5-pixel border unclassified; no padding or interpolation'}
        write_json(output/'scene.json',metadata)
        files=['cube.npy','preview.png','scene.json']+(['labels.npy'] if labels is not None else [])
        write_json(output/'input_manifest.json',{'contract':CONTRACT,'sha256':{name:sha(output/name) for name in files}})
        write_json(output/'preparation.json',{'status':'PASS','finished_utc':now(),'scene_id':metadata['scene_id']})
        return metadata
    except Exception as exc:
        if output.exists():
            write_json(output/'preparation.json',{'status':'FAIL','error':str(exc),'finished_utc':now()})
        raise
    finally:
        close()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'scene_input')
    parser.add_argument('--choose',action='store_true')
    parser.add_argument('--new-job',action='store_true',help='Prepare another scene in a new scene_jobs subfolder')
    args=parser.parse_args()
    if args.choose:
        import tkinter as tk
        from tkinter import filedialog
        root=tk.Tk();root.withdraw()
        filename=filedialog.askopenfilename(title='Choose your real hyperspectral GMxx.mat scene',filetypes=[('MATLAB scene','*.mat')])
        root.destroy()
        if not filename:
            print('No scene selected; preparation cancelled.');return 1
        args.input=Path(filename)
    if args.input is None:
        parser.error('Supply --input actual-scene.mat or --choose')
    if args.new_job:
        args.output=ROOT/'scene_jobs'/('scene-'+uuid.uuid4().hex[:12])
    try:
        metadata=prepare(args.input,args.output)
        print('Real scene prepared:',metadata['source_name'],metadata['prepared_shape'])
        print('No inference performed. The model will classify it on the FPGA.')
        print('Prepared scene folder:',args.output.resolve())
        return 0
    except Exception as exc:
        print('PREPARATION FAILED: {}: {}'.format(type(exc).__name__,exc));return 1

if __name__=='__main__':
    sys.exit(main())
