"""Save measured per-ISO/channel gain and black-frame additive variance.

Uses native undemosaiced RAW DN. Temporal variances have N-1 normalization.
Only a Bayer-phase-preserving sample of the calibration crop is held in RAM.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
PHASES = {'r':(0,0), 'g1':(0,1), 'g2':(1,0), 'b':(1,1)}
SETS = {
    200: ('calibration_iso_200_20260925_220351','black_iso_200_20260925_165853'),
    400: ('calibration_iso_400_20260925_222517','black_iso_400_20260925_160633'),
    800: ('calibration_iso_800_20260925_223045','black_iso_800_20260925_161032'),
    1600: ('calibration_iso_1600_20260925_223725','black_iso_1600_20260925_163736'),
    3200: ('calibration_iso_3200_20260925_224409','black_iso_3200_20260925_164745'),
}


def stats(folder, crop, stride):
    files = sorted(folder.glob('*.tiff'))
    if len(files) < 2:
        raise ValueError(f'Need at least two frames in {folder}')
    y1,x1,y2,x2 = crop
    means, m2 = {}, {}
    for n,path in enumerate(files,1):
        raw = tifffile.memmap(path,mode='r')
        try:
            if raw.shape != (4016,6016) or raw.dtype != np.uint16:
                raise ValueError(f'Unexpected RAW layout: {path}')
            for c,(dy,dx) in PHASES.items():
                a = raw[y1+(dy-y1)%2:y2:stride, x1+(dx-x1)%2:x2:stride].astype(float)
                if n==1:
                    means[c]=np.zeros_like(a);m2[c]=np.zeros_like(a)
                delta=a-means[c]
                means[c]+=delta/n
                m2[c]+=delta*(a-means[c])
        finally:
            raw._mmap.close()
    return means,{c:m2[c]/(len(files)-1) for c in PHASES},len(files)


def fit_gain(mean,var,max_mean,scale):
    # Match main.py: group in rounded scaling-only DN, fit group averages.
    m,v=mean.ravel()*scale,var.ravel()*scale**2
    keep=np.isfinite(m)&np.isfinite(v)&(m>0)&(v>=0)
    _,inv=np.unique(np.rint(m[keep]).astype(np.int64),return_inverse=True)
    count=np.bincount(inv)
    x=np.bincount(inv,weights=m[keep])/count
    y=np.bincount(inv,weights=v[keep])/count
    keep=x<=max_mean
    x,y=x[keep],y[keep]
    if len(x)<3 or np.ptp(x)==0: raise ValueError('Insufficient mean bins')
    g,b=np.polyfit(x,y,1)
    pred=g*x+b
    if not np.isfinite(g) or g<=0: raise ValueError('Nonpositive gain fit')
    return float(g/scale),{'gain_scaled_DN_per_electron':float(g),
          'fitted_intercept_raw_DN2_NOT_USED':float(b/scale**2),
          'bins':len(x),'R_squared':float(1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'data/noise_calibration/iso_noise_profile.json')
    p.add_argument('--stride',type=int,default=4,help='Even RAW sampling stride (2 uses every pixel of each phase)')
    args=p.parse_args()
    if args.stride<2 or args.stride%2:p.error('--stride must be positive and even, >=2')
    crop=(829,1419,2792,4502);scale=65535/(4095-150)
    profile={'schema_version':1,'camera':'Nikon D3500','bayer_pattern':'RG/GB',
             'units':{'gain':'native RAW DN/electron','additive_variance':'native RAW DN^2','black_mean':'native RAW DN'},
             'method':'Gain from ramp mean-variance slope; additive variance directly from temporal black variance, NOT ramp-fit intercept.',
             'crop_y1_x1_y2_x2':crop,'sample_stride':args.stride,'fit_scaling':scale,
             'max_fit_mean_scaled_DN':5000,'saturation_raw_DN':4000,
             'limitations':['Scalar per-channel dark offset, not a pixelwise dark map.',
                            'Short-exposure dark calibration; long-exposure dark-current changes not modeled.',
                            '4000 DN saturation cutoff is a conservative assumption, not calibrated per ISO.'],
             'isos':{}}
    for iso,(ramp,dark) in SETS.items():
        print(f'Calibrating ISO {iso}: black frames',flush=True)
        dm,dv,nd=stats(ROOT/'data'/dark/'raw_tiff',crop,args.stride)
        print(f'Calibrating ISO {iso}: ramp frames',flush=True)
        rm,rv,nr=stats(ROOT/'data'/ramp/'raw_tiff',crop,args.stride)
        row={'ramp_directory':str(ROOT/'data'/ramp/'raw_tiff'),
             'dark_directory':str(ROOT/'data'/dark/'raw_tiff'),
             'ramp_frames':nr,'dark_frames':nd,'channels':{}}
        for c in PHASES:
            gain,fit=fit_gain(rm[c]-dm[c],rv[c],5000,scale)
            additive=float(dv[c].mean())
            if not np.isfinite(additive) or additive<=0:raise ValueError('Invalid dark variance')
            row['channels'][c]={'gain_DN_per_electron':gain,'additive_variance_DN2':additive,
                                'black_mean_DN':float(dm[c].mean()),'fit':fit}
            print(f'  {c}: gain={gain:.6f} DN/e-, additive={additive:.6f} DN^2',flush=True)
        profile['isos'][str(iso)]=row
        del dm,dv,rm,rv
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(profile,indent=2)+'\n')
    with args.output.with_suffix('.csv').open('w', newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['ISO','channel','gain_DN_per_electron','additive_variance_DN2','black_mean_DN'])
        for iso,row in profile['isos'].items():
            for c,q in row['channels'].items():
                writer.writerow([iso,c,q['gain_DN_per_electron'],q['additive_variance_DN2'],q['black_mean_DN']])
    print(f'Saved {args.output}',flush=True)


if __name__=='__main__':main()
