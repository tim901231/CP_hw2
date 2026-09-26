"""Command-line entry points for HW2 Parts 1-5; run with --help."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

import main as algorithms
from cp_hw2 import readHDR, writeHDR
from tiled_merge import merge_tiff_stack

ROOT = Path(__file__).resolve().parents[1]


def camera_times(folder):
    files=[folder/f'exposure{i}.nef' for i in range(1,17)]
    if not all(p.exists() for p in files):
        raise FileNotFoundError('Room merging requires exposure1.nef through exposure16.nef for shutter metadata')
    rows=json.loads(subprocess.check_output(['exiftool','-j','-n','-ExposureTime']+[str(p) for p in files],text=True))
    times={Path(r['SourceFile']).stem:float(r['ExposureTime']) for r in rows}
    return [times[f'exposure{i}'] for i in range(1,17)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='task',required=True)
    m=sub.add_parser('merge',help='Merge a door/room TIFF or JPEG bracket')
    m.add_argument('--scene',choices=['door','room'],default='door')
    m.add_argument('--format',choices=['tiff','jpg'],default='tiff')
    m.add_argument('--domain',choices=['linear','log'],default='log')
    m.add_argument('--weight',choices=['uniform','tent','gaussian','photon'],default='gaussian')
    m.add_argument('--all-variants',action='store_true',help='Both domains and all four weights for this format')
    m.add_argument('--tile-size',type=int,default=256)
    m.add_argument('--sample-step',type=int,default=200)
    m.add_argument('--smoothness',type=float,default=5.)
    m.add_argument('--preview-scale',type=float,default=.5)
    m.add_argument('--output-dir',type=Path)
    c=sub.add_parser('color',help='Apply the saved door ColorChecker correction')
    c.add_argument('--input',type=Path,required=True)
    c.add_argument('--output',type=Path,required=True)
    c.add_argument('--preview-scale',type=float,default=.012)
    t=sub.add_parser('tonemap',help='Photographic tonemapping and gamma-encoded JPEG')
    t.add_argument('--input',type=Path,required=True)
    t.add_argument('--output',type=Path,required=True)
    t.add_argument('--method',choices=['rgb','xyY'],default='rgb')
    t.add_argument('--key',type=float,default=.05)
    t.add_argument('--burn',type=float,default=.95)
    o=sub.add_parser('optimal-room',help='Existing Part 5 fixed-white-balance RGB workflow')
    o.add_argument('--tile-size',type=int,default=256)
    n=sub.add_parser('noise-fit',help='Part 5 G1 mean-variance plot in scaled TIFF DN')
    n.add_argument('--ramp-dir',type=Path,default=ROOT/'data/calibration_iso_200_20260925_220351/raw_tiff')
    n.add_argument('--dark-dir',type=Path,default=ROOT/'data/black_iso_200_20260925_165853/raw_tiff')
    n.add_argument('--output',type=Path,default=ROOT/'output/noise_fit/g_iso200.png')
    args=p.parse_args()
    if hasattr(args,'tile_size') and args.tile_size<=0:p.error('--tile-size must be positive')
    if args.task=='merge':
        if args.sample_step<1 or args.smoothness<0:p.error('Invalid sample step or smoothness')
        folder=ROOT/'data'/f'{args.scene}_stack'
        times=camera_times(folder) if args.scene=='room' else [2**i/2048. for i in range(16)]
        files=[folder/f'exposure{i}.{args.format}' for i in range(1,17)]
        if not all(f.exists() for f in files):raise FileNotFoundError(f'Expected all 16 {args.format} inputs in {folder}')
        out=args.output_dir or ROOT/'output/reproduced'/f'{args.scene}_{args.format}'
        out.mkdir(parents=True,exist_ok=True)
        weights=['uniform','tent','gaussian','photon'] if args.all_variants else [args.weight]
        modes=['linear','log'] if args.all_variants else [args.domain]
        for weight in weights:
            g=None
            if args.format=='jpg':
                # Only calibration samples are stacked; merging streams frames.
                samples=np.stack([imread(f)[::args.sample_step,::args.sample_step].reshape(-1) for f in files])
                g=algorithms.solve_g(samples,args.smoothness,weight,exposure_times=times)
                np.save(out/f'g_{weight}.npy',g)
                fig,ax=plt.subplots();ax.plot(np.arange(256),g)
                ax.set(xlabel='JPEG intensity',ylabel='g(I)',title=f'Recovered response: {weight}')
                ax.grid(alpha=.3);fig.tight_layout();fig.savefig(out/f'g_{weight}.png',dpi=160);plt.close(fig)
                del samples
            for mode in modes:
                if args.format=='tiff':
                    hdr=merge_tiff_stack(files,algorithms.merge,tile_size=args.tile_size,
                                        exposure_times=times,config=mode,weight=weight)
                else:
                    lut=np.exp(g).astype(np.float32)
                    observed=(imread(f).astype(np.float32)/255. for f in files)
                    linear=(lut[imread(f)] for f in files)
                    hdr=algorithms.merge(observed,linear,config=mode,weight=weight,exposure_times=times)
                stem=out/f'{args.scene}_{mode}_{weight}'
                writeHDR(str(stem.with_suffix('.hdr')),hdr)
                algorithms.save_jpg(str(stem.with_suffix('.jpg')),hdr,scale=args.preview_scale)
                del hdr
        (out/'settings.json').write_text(json.dumps({**vars(args), 'output_dir':str(out), 'exposure_times':times},default=str,indent=2)+'\n')
    elif args.task=='color':
        args.output.parent.mkdir(parents=True,exist_ok=True)
        hdr=algorithms.color_correction(readHDR(str(args.input)))
        writeHDR(str(args.output),hdr)
        algorithms.save_jpg(str(args.output.with_suffix('.jpg')),hdr,args.preview_scale)
    elif args.task=='tonemap':
        if args.key<=0 or args.burn<=0:p.error('--key and --burn must be positive')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        hdr=np.maximum(readHDR(str(args.input)),0)
        mapped=algorithms.tonemapping(hdr,args.method,K=args.key,B=args.burn)
        algorithms.save_jpg(str(args.output),mapped,scale=1)
    elif args.task=='optimal-room':
        algorithms.merge_room_rgb(tile_size=args.tile_size)
    elif args.task=='noise-fit':
        args.output.parent.mkdir(parents=True,exist_ok=True)
        crop=(829,1419,2792,4502);scale=65535./(4095.-150.)
        dark,dark_var=algorithms.load_black_img(args.dark_dir,'g1',crop,scale)
        mean,var=algorithms.load_calibrated_img(args.ramp_dir,dark,'g1',crop,scale)
        # Save instead of requiring an interactive plot window.
        show=plt.show
        plt.show=lambda: None
        try:
            g,intercept=algorithms.fit_data(mean,var,x_threshold=5000)
            plt.savefig(args.output,dpi=160);plt.close()
        finally:plt.show=show
        args.output.with_suffix('.json').write_text(json.dumps({'gain_scaled_DN_per_electron':g,
            'ramp_intercept_scaled_DN2':intercept,'measured_black_variance_scaled_DN2':float(dark_var.mean()),
            'scale':scale},indent=2)+'\n')


if __name__=='__main__':main()
