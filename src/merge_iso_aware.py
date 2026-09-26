"""Patchwise ISO-aware optimal merging using saved empirical calibration.

Gain is DN/electron. Inputs remain native Bayer RAW, without WB/demosaicing.
Variance = gain * max(dark_corrected_DN, 0) + measured_black_variance.
Outputs preserve a signed full-resolution Bayer flux map and a common camera-RGB
rendering for both brackets. No fitted ramp intercept is used as additive noise.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy.ndimage import convolve

from tiled_merge import patches, read_patch, validate_stack

ROOT=Path(__file__).resolve().parents[1]
PHASES={'r':(0,0),'g1':(0,1),'g2':(1,0),'b':(1,1)}


def iso_weights(signal,t,gain,additive,valid):
    if not np.isfinite([t,gain,additive]).all() or min(t,gain,additive)<=0:
        raise ValueError('Exposure, gain, and additive variance must be finite and positive')
    variance=gain*np.maximum(signal,0.)+additive
    return np.where(valid,(gain*t)**2/variance,0.)


def merge_records(records,profile,tile_size=256):
    files,shape=validate_stack([r['tiff'] for r in records],tile_size)
    if len(shape)!=2:raise ValueError('Expected RAW Bayer mosaics')
    if not records:raise ValueError('No input frames')
    times=np.array([r['ExposureTime'] for r in records])
    if not np.isfinite(times).all() or np.any(times<=0) or np.any(np.diff(times)<0):
        raise ValueError('Frames must have positive, increasing exposure times')
    hdr=np.empty(shape,np.float32);usable=np.empty(shape,bool)
    max_dn=float(profile['saturation_raw_DN'])
    total=((shape[0]+tile_size-1)//tile_size)*((shape[1]+tile_size-1)//tile_size)
    for index,(ys,xs) in enumerate(patches(shape,tile_size),1):
        size=(ys.stop-ys.start,xs.stop-xs.start)
        num=np.zeros(size,float);den=np.zeros(size,float);fallback=np.empty(size,np.float32)
        for i,(record,file) in enumerate(zip(records,files)):
            raw=read_patch(file,ys,xs);t=float(record['ExposureTime'])
            params=profile['isos'][str(record['ISO'])]['channels']
            for c,(dy,dx) in PHASES.items():
                phase=(slice((dy-ys.start)%2,None,2),slice((dx-xs.start)%2,None,2))
                q=params[c];gain=float(q['gain_DN_per_electron'])
                observed=raw[phase]
                z=observed.astype(float)-q['black_mean_DN']
                flux=z/(gain*t)
                valid=(observed>0)&(observed<max_dn)
                weight=iso_weights(z,t,gain,q['additive_variance_DN2'],valid)
                num[phase]+=weight*flux;den[phase]+=weight
                if i==0:fallback[phase]=flux
        valid=den>0
        np.divide(num,den,out=fallback,where=valid,casting='unsafe')
        hdr[ys,xs]=fallback;usable[ys,xs]=valid
        if index==1 or index%50==0 or index==total:print(f'Patch {index}/{total}',flush=True)
    return hdr,usable


def decode(nef,cache):
    existing=nef.parent/'raw_tiff'/f'{nef.stem}.tiff'
    if existing.exists():return existing
    target=cache/nef.parent.name/f'{nef.stem}.tiff'
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        temporary=target.with_suffix('.partial')
        try:
            with temporary.open('wb') as f:
                subprocess.run(['dcraw','-D','-4','-j','-t','0','-T','-c',str(nef)],stdout=f,check=True)
            temporary.replace(target)
        finally:
            if temporary.exists():temporary.unlink()
    return target


def metadata(nefs,profile,cache):
    records=json.loads(subprocess.check_output(['exiftool','-j','-n','-ISO','-ExposureTime',
        '-FNumber','-Model','-DateTimeOriginal','-RedBalance','-BlueBalance']+[str(p) for p in nefs],text=True))
    for r in records:
        r['ISO']=int(r['ISO'])
        if str(r['ISO']) not in profile['isos']:raise ValueError(f"No calibration for ISO {r['ISO']}")
        if r.get('Model')!='NIKON D3500':raise ValueError('Calibration is only for D3500')
        r['tiff']=str(decode(Path(r['SourceFile']),cache))
    records.sort(key=lambda r:r['ExposureTime'])
    if len({r['FNumber'] for r in records})!=1:raise ValueError('Aperture changed')
    return records


def camera_rgb(flux,profile,wb):
    """Float32 bilinear demosaicing, converted to a shared ISO200 DN/s scale.

    Camera RGB only: no camera-to-sRGB color matrix, no clipping before filtering.
    """
    h,w=flux.shape
    rgb=np.empty((h,w,3),np.float32)
    for channel,phases in enumerate([['r'],['g1','g2'],['b']]):
        signal=np.zeros((h,w),np.float32);mask=np.zeros((h,w),np.float32)
        for c in phases:
            dy,dx=PHASES[c]
            gain=profile['isos']['200']['channels'][c]['gain_DN_per_electron']
            signal[dy::2,dx::2]=flux[dy::2,dx::2]*gain*wb[channel]
            mask[dy::2,dx::2]=1
        kernel=np.array([[1,2,1],[2,4,2],[1,2,1]],np.float32)
        convolve(signal,kernel,output=signal,mode='mirror')
        convolve(mask,kernel,output=mask,mode='mirror')
        np.divide(signal,mask,out=rgb[:,:,channel],where=mask>0)
    return rgb


def encode(a,scale):
    a=np.clip(a*scale,0,1)
    return np.round(np.where(a<=.0031308,12.92*a,1.055*a**(1/2.4)-.055)*255).astype(np.uint8)



def render_saved(out,target_mean=.225):
    """Shared scalar from the baseline mean, then main.py's sRGB encoding."""
    out=Path(out)
    report=json.loads((out/'merge_report.json').read_text())
    previews={};crops={};scale=None;stats={}
    for name in ['fixed_iso','varying_iso']:
        rgb=tifffile.memmap(out/f'{name}_camera_rgb_signed.tiff',mode='r')
        try:
            total=0.
            for y in range(0,rgb.shape[0],256):
                total+=np.maximum(rgb[y:y+256],0).sum(dtype=np.float64)
            mean=total/rgb.size
            if not np.isfinite(mean) or mean<=0:raise ValueError('Cannot scale a zero/nonfinite image mean')
            if scale is None:scale=target_mean/mean
            encoded=np.empty(rgb.shape,np.uint8)
            clipped_sum=0.;clipped_count=0
            for y in range(0,rgb.shape[0],256):
                block=rgb[y:y+256]
                linear=block*scale
                clipped_sum+=np.clip(linear,0,1).sum(dtype=np.float64)
                clipped_count+=np.count_nonzero(linear>1)
                encoded[y:y+256]=encode(block,scale)
            im=Image.fromarray(encoded)
            im.save(out/f'{name}_preview.jpg',quality=95,subsampling=0)
            im.save(out/f'{name}_preview.png')
            preview=im.copy();preview.thumbnail((900,600));previews[name]=preview
            crops[name]=Image.fromarray(encode(rgb[1900:2412,1500:2012],scale*8))
            crops[name].save(out/f'{name}_shadow_8x.png')
            stats[name]={'source_nonnegative_linear_mean':mean,'scaled_linear_mean_before_clipping':mean*scale,
                         'clipped_linear_mean':clipped_sum/rgb.size,'clipped_channel_fraction':clipped_count/rgb.size}
            print(f'{name}: scalar={scale:.8g}, linear mean before clipping={mean*scale:.4f}',flush=True)
            del encoded,im
        finally:rgb._mmap.close()
    report['display']={'method':'shared scalar from fixed-ISO linear mean, clip [0,1], sRGB gamma encode',
                       'target_linear_mean_before_clipping':target_mean,'shared_linear_scale':scale,
                       'encoding':'sRGB transfer function on camera RGB','shadow_boost':8,
                       'crop_xyxy':[1500,1900,2012,2412],'statistics':stats}
    for filename,images,width,height in [('comparison.png',previews,900,600),('shadow_comparison_100pct.png',crops,512,512)]:
        canvas=Image.new('RGB',(width*2+20,height+45),'white');draw=ImageDraw.Draw(canvas)
        for i,(name,im) in enumerate(images.items()):
            draw.text((i*(width+20)+5,12),name+' | identical display settings',fill='black')
            canvas.paste(im,(i*(width+20),40))
        canvas.save(out/filename)
        if filename=='comparison.png':canvas.save(out/'comparison.jpg',quality=95,subsampling=0)
    (out/'merge_report.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,default=ROOT/'data/noise_calibration/iso_noise_profile.json')
    p.add_argument('--fixed',type=Path,default=ROOT/'data/room_fix_iso')
    p.add_argument('--varying',type=Path,default=ROOT/'data/room_vary_iso')
    p.add_argument('--reuse-fixed-first-three',action='store_true',help='Explicitly reuse fixed bracket frames 1-3')
    p.add_argument('--tile-size',type=int,default=256)
    p.add_argument('--output',type=Path,default=ROOT/'output/iso_aware_merge')
    p.add_argument('--target-mean',type=float,default=.225,help='Mean linear intensity before display clipping/gamma')
    p.add_argument('--render-only',action='store_true',help='Render existing signed RGB TIFFs without merging again')
    args=p.parse_args()
    if not np.isfinite(args.target_mean) or not 0 < args.target_mean <= 1:
        p.error('--target-mean must be in (0,1]')
    if args.render_only:
        render_saved(args.output,args.target_mean)
        return
    if args.tile_size<=0:p.error('--tile-size must be positive')
    profile=json.loads(args.profile.read_text());out=args.output;out.mkdir(parents=True,exist_ok=True)
    fixed=[args.fixed/f'{i}.NEF' for i in range(1,9)]
    varying=[(args.fixed if args.reuse_fixed_first_three and i<=3 else args.varying)/f'{i}.NEF' for i in range(1,9)]
    for f in fixed+varying:
        if not f.exists():raise FileNotFoundError(f)
    stacks={name:metadata(files,profile,out/'raw_cache') for name,files in [('fixed_iso',fixed),('varying_iso',varying)]}
    if {r['ISO'] for r in stacks['fixed_iso']}!={200}:raise ValueError('Expected fixed ISO200 bracket')
    if len({r['FNumber'] for rows in stacks.values() for r in rows})!=1:raise ValueError('Aperture differs across stacks')
    a,b=stacks.values()
    if not np.allclose([r['ExposureTime'] for r in a],[r['ExposureTime'] for r in b],rtol=1e-6,atol=1e-9):
        raise ValueError('Comparison requires matching shutter times')
    wb=[float(a[0]['RedBalance']),1.,float(a[0]['BlueBalance'])]
    report={'profile':str(args.profile),'gain_convention':'DN/electron',
            'equations':{'flux':'z/(gain*t)','variance':'gain*max(z,0)+black_variance','weight':'(gain*t)^2/variance'},
            'shared_white_balance':wb,'reused_fixed_frames':[1,2,3] if args.reuse_fixed_first_three else [],
            'radiance_units':'electrons/second in RAW mosaic; ISO200-equivalent camera DN/second in RGB',
            'limitations':profile['limitations']+['No image registration or motion correction.',
                'Camera-RGB HDR uses bilinear demosaicing, no camera-to-sRGB matrix.',
                'Equal integration time; manually timed captures are not equal elapsed capture time.',
                'Dark-offset estimation uncertainty is neglected; shared calibration may induce correlated errors.'],
            'stacks':{}}
    shutil.copy2(args.profile,out/'calibration_used.json')
    import cv2
    for name,records in stacks.items():
        print(f'Merging {name}',flush=True)
        hdr,valid=merge_records(records,profile,args.tile_size)
        if not np.isfinite(hdr).all():raise ValueError('Nonfinite result')
        tifffile.imwrite(out/f'{name}_bayer_electrons_per_second.tiff',hdr,photometric='minisblack')
        tifffile.imwrite(out/f'{name}_valid_mask.tiff',valid.astype(np.uint8)*255)
        invalid=float(1-valid.mean());del valid
        print('Demosaicing for common RGB preview',flush=True)
        rgb=camera_rgb(hdr,profile,wb);del hdr
        # Signed float TIFF is authoritative. Radiance RGBE cannot store negatives.
        tifffile.imwrite(out/f'{name}_camera_rgb_signed.tiff',rgb,photometric='rgb')
        if not cv2.imwrite(str(out/f'{name}_camera_rgb.hdr'),np.maximum(rgb[:,:,::-1],0)):
            raise OSError('HDR export failed')
        report['stacks'][name]={'frames':records,'total_exposure_seconds':sum(r['ExposureTime'] for r in records),
                                'fallback_fraction':invalid,'fallback':'Shortest exposure, converted by its gain and time'}
        del rgb
    (out/'merge_report.json').write_text(json.dumps(report,indent=2)+'\n')
    render_saved(out,args.target_mean)
    print(f'Saved both merges to {out}',flush=True)


if __name__=='__main__':main()
