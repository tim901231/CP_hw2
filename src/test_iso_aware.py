import tempfile
import unittest
from pathlib import Path
import numpy as np
import tifffile
from merge_iso_aware import merge_records, iso_weights, camera_rgb, PHASES


class ISOAwareTests(unittest.TestCase):
    def test_gain_normalization_and_inverse_variance(self):
        with tempfile.TemporaryDirectory() as tmp:
            shape=(7,9);profile={'saturation_raw_DN':4000,'isos':{}};records=[]
            for iso,gain,t in [(200,1.,1.),(800,4.,.5)]:
                params={c:{'gain_DN_per_electron':gain,'black_mean_DN':150.,'additive_variance_DN2':2.}
                        for c in PHASES}
                profile['isos'][str(iso)]={'channels':params}
                path=Path(tmp)/f'{iso}.tiff'
                tifffile.imwrite(path,np.full(shape,150+100*gain*t,np.uint16))
                records.append({'tiff':str(path),'ISO':iso,'ExposureTime':t})
            records.sort(key=lambda r:r['ExposureTime'])
            for tile in [1,3,16]:
                hdr,valid=merge_records(records,profile,tile)
                np.testing.assert_allclose(hdr,100)
                self.assertTrue(valid.all())
        z=np.array([-2.,0.,8.]);w=iso_weights(z,.5,4.,2.,np.array([True,False,True]))
        np.testing.assert_allclose(w,[2.,0.,4/34])

    def test_phase_edges_signed_signal_and_saturation_fallback(self):
        rng=np.random.default_rng(10)
        with tempfile.TemporaryDirectory() as tmp:
            profile={'saturation_raw_DN':4000,'isos':{}};records=[];arrays=[]
            for i,(iso,t) in enumerate([(200,.25),(400,1.)]):
                params={c:{'gain_DN_per_electron':(j+1)*(i+1)*.2,
                            'black_mean_DN':150.+j,'additive_variance_DN2':1.+j+i}
                        for j,c in enumerate(PHASES)}
                profile['isos'][str(iso)]={'channels':params}
                a=rng.integers(140,4100,size=(7,9),dtype=np.uint16)
                a[0,0]=4095;a[2,2]=149
                path=Path(tmp)/f'{iso}.tiff';tifffile.imwrite(path,a)
                records.append({'tiff':str(path),'ISO':iso,'ExposureTime':t});arrays.append(a)
            num=np.zeros((7,9));den=np.zeros((7,9));fallback=np.zeros((7,9))
            for i,(r,a) in enumerate(zip(records,arrays)):
                for c,(dy,dx) in PHASES.items():
                    p=profile['isos'][str(r['ISO'])]['channels'][c]
                    z=a[dy::2,dx::2].astype(float)-p['black_mean_DN'];g=p['gain_DN_per_electron'];t=r['ExposureTime']
                    f=z/(g*t);v=g*np.maximum(z,0)+p['additive_variance_DN2']
                    w=np.where(a[dy::2,dx::2]<4000,(g*t)**2/v,0.)
                    num[dy::2,dx::2]+=w*f;den[dy::2,dx::2]+=w
                    if i==0:fallback[dy::2,dx::2]=f
            np.divide(num,den,out=fallback,where=den>0)
            actual,valid=merge_records(records,profile,3)
            np.testing.assert_allclose(actual,fallback,rtol=1e-6)
            np.testing.assert_array_equal(valid,den>0)
            self.assertLess(actual[2,2],0)

    def test_demosaic_constant_flux(self):
        params={c:{'gain_DN_per_electron':1.} for c in PHASES}
        rgb=camera_rgb(np.full((8,10),100,np.float32),{'isos':{'200':{'channels':params}}},[2,1,3])
        np.testing.assert_allclose(rgb[:,:,0],200)
        np.testing.assert_allclose(rgb[:,:,1],100)
        np.testing.assert_allclose(rgb[:,:,2],300)


if __name__=='__main__':unittest.main()
