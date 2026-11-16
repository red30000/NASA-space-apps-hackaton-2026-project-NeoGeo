"""Conservative NISAR/other HDF5 geocoded raster reader.
Only accepts geocoded datasets with coordinate vectors and identifiable CRS.
Does not pretend radar-coordinate SLC arrays are map rasters.
"""
import io
import numpy as np
import h5py
from affine import Affine
from rasterio.crs import CRS


def inspect_hdf5(blob):
    items=[]
    with h5py.File(io.BytesIO(blob),'r') as f:
        def visit(name,obj):
            if isinstance(obj,h5py.Dataset) and obj.ndim==2 and np.issubdtype(obj.dtype,np.number):
                items.append({'path':'/'+name,'shape':tuple(obj.shape),'dtype':str(obj.dtype)})
        f.visititems(visit)
    return items

def _decode(v):
    if isinstance(v,bytes):return v.decode(errors='ignore')
    if isinstance(v,np.ndarray) and v.size==1:return _decode(v.flat[0])
    return v

def load_geocoded_hdf5(blob,path):
    with h5py.File(io.BytesIO(blob),'r') as f:
        ds=f[path]
        a=ds[()]
        if a.ndim!=2 or np.iscomplexobj(a):
            raise ValueError('Select a 2D real-valued geocoded backscatter layer, not complex SLC data.')
        # NISAR GCOV products commonly store coordinate vectors in the dataset parent.
        group=ds.parent
        xx=next((group[k][()] for k in ('xCoordinates','x_coordinates','x') if k in group),None)
        yy=next((group[k][()] for k in ('yCoordinates','y_coordinates','y') if k in group),None)
        if xx is None or yy is None:
            raise ValueError('No x/y coordinate vectors found beside the selected dataset. This may be radar-coordinate data; use a geocoded GCOV layer or georeference externally.')
        xx=np.asarray(xx).ravel(); yy=np.asarray(yy).ravel()
        if len(xx)!=a.shape[1] or len(yy)!=a.shape[0] or len(xx)<2 or len(yy)<2:
            raise ValueError('Coordinate vector lengths do not match the raster dimensions.')
        dx=float(np.median(np.diff(xx)));dy=float(np.median(np.diff(yy)))
        if not np.allclose(np.diff(xx),dx,atol=max(1e-6,abs(dx)*1e-4)) or not np.allclose(np.diff(yy),dy,atol=max(1e-6,abs(dy)*1e-4)):
            raise ValueError('Coordinates are not regularly spaced. Reproject the dataset before analysis.')
        epsg=None
        for obj in (group,ds,f):
            for key in ('epsg','EPSG','epsg_code'):
                if key in obj.attrs:
                    try:epsg=int(_decode(obj.attrs[key]));break
                    except (TypeError,ValueError):pass
            if epsg:break
        if not epsg:
            for key in ('projection','projectionInformation'):
                if key in group and hasattr(group[key],'attrs'):
                    for field in ('epsg','epsg_code'):
                        if field in group[key].attrs:
                            try:epsg=int(_decode(group[key].attrs[field]));break
                            except (TypeError,ValueError):pass
        if not epsg:
            raise ValueError('EPSG code could not be verified from this HDF5 product. Export a georeferenced GeoTIFF using an official NISAR reader first.')
        transform=Affine.translation(float(xx[0]-dx/2),float(yy[0]-dy/2))*Affine.scale(dx,dy)
        arr=np.asarray(a,dtype='float32')
        fill=ds.attrs.get('_FillValue',ds.attrs.get('fill_value',None))
        if fill is not None:
            try:arr[arr==float(_decode(fill))]=np.nan
            except (TypeError,ValueError):pass
        arr[~np.isfinite(arr)]=np.nan
        return arr,{'driver':'GTiff','height':arr.shape[0],'width':arr.shape[1],'count':1,'dtype':'float32','crs':CRS.from_epsg(epsg),'transform':transform},transform,CRS.from_epsg(epsg)
