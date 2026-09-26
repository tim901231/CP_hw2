"""Visualize real ISO-merge differences without labeling them as pure noise."""
import json
from pathlib import Path
import numpy as np
import tifffile
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

ROOT=Path(__file__).resolve().parents[1]/'output/iso_aware_merge'
OUT=ROOT/'difference_analysis'
ROIS={'Dark wall':(650,1850,906,2106),'Dark clothing':(1700,1900,1956,2156)}


def gamma(a):
    a=np.clip(a,0,1)
    return np.where(a<=.0031308,12.92*a,1.055*a**(1/2.4)-.055)


def main():
    OUT.mkdir(exist_ok=True)
    report=json.loads((ROOT/'merge_report.json').read_text())
    profile=json.loads((ROOT/'calibration_used.json').read_text())
    a=tifffile.memmap(ROOT/'fixed_iso_camera_rgb_signed.tiff',mode='r')
    b=tifffile.memmap(ROOT/'varying_iso_camera_rgb_signed.tiff',mode='r')
    flux=tifffile.memmap(ROOT/'fixed_iso_bayer_electrons_per_second.tiff',mode='r')
    try:
        x=np.log1p(np.maximum(a[::4,::4].mean(axis=2),0)).astype(np.float32)
        y=np.log1p(np.maximum(b[::4,::4].mean(axis=2),0)).astype(np.float32)
        score,matrix=cv2.findTransformECC(x,y,np.eye(2,3,dtype=np.float32),cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-6))
        summary={'translation_full_pixels':(matrix[:,2]*4).tolist(),'ECC_correlation':float(score),
                 'registration_applied':False,'note':'No resampling of crops. Differences contain noise, scene/lighting changes, calibration errors and residual alignment. First three source frames are shared.',
                 'prediction':'Ideal inverse-variance model at common estimated G1 scene flux, using actual saturation masks; not measured temporal SNR.',
                 'rois':{}}
        fig,axes=plt.subplots(2,3,figsize=(13,9),constrained_layout=True)
        residual_fig,residual_axes=plt.subplots(2,2,figsize=(9,9),constrained_layout=True)
        for row,(label,(x1,y1,x2,y2)) in enumerate(ROIS.items()):
            pa=a[y1:y2,x1:x2].copy();pb=b[y1:y2,x1:x2].copy()
            scale=.25/np.maximum(pa,0).mean()
            for col,img in enumerate([pa,pb]):
                axes[row,col].imshow(gamma(img*scale),interpolation='nearest')
                axes[row,col].set_title(f'{label} | '+['Fixed ISO','Varying ISO'][col])
                axes[row,col].axis('off')
            # G1 raw estimator: native same-channel samples, no demosaicing.
            phi=np.maximum(flux[y1:y2:2,x1+1:x2:2].astype(float),0)
            weights={}
            for name,stack in report['stacks'].items():
                den=np.zeros_like(phi)
                for record in stack['frames']:
                    params=profile['isos'][str(record['ISO'])]['channels']['g1']
                    t=record['ExposureTime'];gain=params['gain_DN_per_electron']
                    raw=tifffile.memmap(record['tiff'],mode='r')
                    try:observed=raw[y1:y2:2,x1+1:x2:2].copy()
                    finally:raw._mmap.close()
                    variance=gain**2*t*phi+params['additive_variance_DN2']
                    den+=np.where((observed>0)&(observed<profile['saturation_raw_DN']),(gain*t)**2/variance,0)
                weights[name]=den
            snr=np.sqrt(weights['varying_iso']/weights['fixed_iso'])
            green_mean=float(pa[:,:,1].mean())
            difference=(pb[:,:,1]-pa[:,:,1])/green_mean*100
            heat=axes[row,2].imshow(difference,cmap='RdBu_r',vmin=-10,vmax=10,interpolation='nearest')
            axes[row,2].set_title('Amplified signed difference\nVarying minus fixed (green channel)')
            axes[row,2].axis('off')
            fig.colorbar(heat,ax=axes[row,2],shrink=.78,label='% of fixed crop mean (clipped at ±10%)')
            residuals=[p[:,:,1]-gaussian_filter(p[:,:,1],2) for p in [pa,pb]]
            limit=float(np.percentile(np.abs(residuals[0]),99))
            for col,residual in enumerate(residuals):
                residual_axes[row,col].imshow(residual,cmap='gray',vmin=-limit,vmax=limit,interpolation='nearest')
                residual_axes[row,col].set_title(label+' | '+['Fixed ISO','Varying ISO'][col])
                residual_axes[row,col].axis('off')
            summary['rois'][label]={'crop_xyxy':[x1,y1,x2,y2],
                'shared_crop_display_scalar':float(scale),'median_electrons_in_8s':float(np.median(phi)*8),
                'predicted_median_SNR_ratio_varying_over_fixed':float(np.median(snr)),
                'predicted_median_noise_std_ratio_varying_over_fixed':float(np.median(1/snr)),
                'mean_green_difference_percent':float(difference.mean())}
            # 4x nearest-neighbor enlargements avoid interpolation smoothing.
            for name,patch in [('fixed',pa),('varying',pb)]:
                im=Image.fromarray(np.round(gamma(patch*scale)*255).astype(np.uint8))
                im.resize((1024,1024),Image.Resampling.NEAREST).save(OUT/f'{label.lower().replace(" ","_")}_{name}_4x.png')
            print(label,summary['rois'][label])
        fig.suptitle('Shadow comparison | 256 × 256 source-pixel crops, matched brightness within each row',fontsize=14)
        fig.savefig(OUT/'shadow_difference_comparison.png',dpi=160)
        plt.close(fig)
        residual_fig.suptitle('Amplified fine-scale residuals | same contrast within each row\nContains texture and noise; not an isolated noise measurement',fontsize=13)
        residual_fig.savefig(OUT/'fine_scale_residuals.png',dpi=160)
        plt.close(residual_fig)
        # Locator uses exactly the coordinates of the original full-resolution image.
        fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True)
        preview=Image.open(ROOT/'fixed_iso_preview.jpg');preview.thumbnail((1504,1004))
        ax.imshow(preview,extent=[0,a.shape[1],a.shape[0],0])
        from matplotlib.patches import Rectangle
        for label,(x1,y1,x2,y2) in ROIS.items():
            ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='#00ffff',linewidth=2))
            ax.text(x1,y1-45,label,color='white',bbox=dict(facecolor='black',alpha=.7,pad=3))
        ax.set_title('Locations of the shadow crops');ax.axis('off')
        fig.savefig(OUT/'crop_locations.png',dpi=150);plt.close(fig)
        (OUT/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    finally:
        a._mmap.close();b._mmap.close();flux._mmap.close()


if __name__=='__main__':main()
