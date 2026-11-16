# Chatgpt helped with getting the list of imports and useful tools(I still had to put it in by hand)
import io
import json
import numpy as np
import streamlit as st
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import transform_bounds
from scipy.ndimage import median_filter, binary_opening, binary_closing, binary_dilation, label
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from nisar_io import inspect_hdf5, load_geocoded_hdf5
from rasterio.transform import rowcol
from pyproj import Transformer
import pandas as pd
#configure the page and front/end
st.set_page_config(page_title='Geonova | SAR Response Explorer', layout='wide')
st.title('GEONOVA | SAR Response Explorer — Integrated')
st.caption('Two-date SAR backscatter analysis • exploratory flood screening • spatial uncertainty indicators')
st.warning('Screening tool only. Radar backscatter change is NOT confirmed flooding, water depth, ground deformation, or a safe rescue route. Validate with field reports, weather, optical imagery, and additional SAR acquisitions.')
#had to ask Chatgpt for help with the side-bar
with st.sidebar:
    st.header('Analysis controls')
    units=st.selectbox('Input backscatter units',['linear power','dB'])
    band=st.selectbox('Polarization / channel label',['VV','VH','Other'])
    threshold=st.slider('Change threshold |Δ| (dB)',1.0,10.0,3.0,0.5)
    smooth=st.select_slider('Speckle-reduction median window (pixels)',options=[1,3,5,7],value=3)
    noise_mult=st.slider('Noise-screening strength (× robust noise)',0.0,4.0,1.5,0.25)
    min_patch=st.slider('Minimum connected patch (pixels)',1,250,25)
    water_db=st.slider('Radar-dark candidate threshold (dB)',-30.0,-8.0,-18.0,0.5)
    proximity=st.slider('Near pre-existing dark-water pixels (pixels)',0,40,12)
    opacity=st.slider('Hotspot overlay opacity',0.2,1.0,0.8,0.05)
    st.caption('NISAR support: geocoded, real-valued HDF5 raster layers with verifiable coordinates and EPSG only.')
#never worked with GeoTIFF, had to use ai help for it
st.subheader('1 · Import radar acquisitions')
mode=st.radio('Source format',['GeoTIFF (Sentinel-1 or exported NISAR)','Geocoded NISAR HDF5 (experimental)'],horizontal=True)
before_file=st.file_uploader('Before acquisition',type=['tif','tiff'] if mode.startswith('GeoTIFF') else ['h5','hdf5'],key='before')
after_file=st.file_uploader('After acquisition',type=['tif','tiff'] if mode.startswith('GeoTIFF') else ['h5','hdf5'],key='after')
before_path=after_path=None
if mode.startswith('Geocoded'):
    for file,label_name in [(before_file,'Before'),(after_file,'After')]:
        if file:
            try:
                layers=inspect_hdf5(file.getvalue())
                if not layers:st.error(f'{label_name}: no 2D numerical layers found.')
                else:
                    selection=st.selectbox(f'{label_name} — geocoded backscatter layer',[e['path'] for e in layers],key=label_name)
                    if label_name=='Before':before_path=selection
                    else:after_path=selection
            except Exception as e:st.error(f'{label_name} HDF5 inspection: {e}')
else:
    before_band=st.number_input('Before GeoTIFF band (1=VV for your existing clipped images)',1,20,1)
    after_band=st.number_input('After GeoTIFF band',1,20,1)

@st.cache_data(show_spinner=False)
def read_raster(blob,band_index=1):
    with MemoryFile(blob) as m:
        with m.open() as ds:
            if band_index>ds.count: raise ValueError(f'Band {band_index} unavailable: file has {ds.count} bands.')
            a=ds.read(band_index,masked=True).filled(np.nan).astype('float32')
            return a,ds.profile.copy(),ds.transform,ds.crs,ds.bounds

def median_valid(arr, size):
    if size==1:return arr
    good=np.isfinite(arr)
    # Nearest-neighbor fill invalid pixels locally, without allowing zeros to contaminate the median.
    from scipy.ndimage import distance_transform_edt
    if not np.any(good): return arr
    idx=distance_transform_edt(~good,return_distances=False,return_indices=True)
    filled=arr[tuple(idx)]
    result=median_filter(filled,size=size,mode='nearest')
    result[~good]=np.nan
    return result
#check changes within files
def remove_small(mask,n):
    if n<=1:return mask
    labs,k=label(mask)
    counts=np.bincount(labs.ravel())
    keep=counts>=n
    keep[0]=False
    return keep[labs]

def stretched(arr,lo,hi):
    z=np.clip((arr-lo)/max(hi-lo,0.001),0,1)
    return np.nan_to_num(z,nan=0).astype('float32')

def png_bytes(fig):
    b=io.BytesIO();fig.savefig(b,format='png',dpi=170,bbox_inches='tight',facecolor='white');return b.getvalue()
#had to use Chat gpt for this section
def tif_bytes(array,profile,description,dtype='float32'):
    prof=profile.copy();prof.update(count=1,dtype=dtype,nodata=-9999 if dtype=='float32' else 255,compress='deflate')
    with MemoryFile() as m:
        with m.open(**prof) as d:
            if dtype=='float32':d.write(np.where(np.isfinite(array),array,-9999).astype('float32'),1)
            else:d.write(array.astype('uint8'),1)
            d.set_band_description(1,description)
        return m.read()

if before_file and after_file:
    try:
        with st.spinner('Calculating change, robust noise statistics and mapped screening zones...'):
            
            if mode.startswith('Geocoded'):
                
                if not before_path or not after_path:st.stop()
                x,profile,t1,crs1=load_geocoded_hdf5(before_file.getvalue(),before_path)
                y,_,t2,crs2=load_geocoded_hdf5(after_file.getvalue(),after_path)
                bounds=rasterio.transform.array_bounds(x.shape[0],x.shape[1],t1)
            else:
                
                
                x,profile,t1,crs1,bounds=read_raster(before_file.getvalue(),before_band)
                y,_,t2,crs2,_=read_raster(after_file.getvalue(),after_band)
            if x.shape!=y.shape or t1!=t2 or crs1!=crs2:
                
                st.error('Images must have identical grid, transform and CRS.');st.stop()
            if units=='linear power':
                x=np.where(x>0,10*np.log10(np.maximum(x,1e-12)),np.nan)
                y=np.where(y>0,10*np.log10(np.maximum(y,1e-12)),np.nan)
            x=median_valid(x,smooth);y=median_valid(y,smooth)
            valid=np.isfinite(x)&np.isfinite(y)
            if valid.sum()<100: st.error('Not enough shared valid pixels. Check units and nodata.');st.stop()
            delta=np.where(valid,y-x,np.nan)
            values=delta[valid]
            bias=float(np.median(values))
            mad=float(np.median(np.abs(values-bias)))
            robust_sigma=1.4826*mad
            # Descriptive noise screen: global scene heterogeneity includes real change, not pure instrument error.
            effective=max(threshold,noise_mult*robust_sigma)
            changed=valid&(np.abs(delta)>=effective)
            hotspots=remove_small(changed,min_patch)
            decreases=hotspots&(delta<0)
            increases=hotspots&(delta>0)
# Darkening near previously "radar-dark" areas is only a flood screening heuristics.
            old_dark=valid&(x<water_db)
            near_water=binary_dilation(old_dark,iterations=proximity) if proximity else old_dark
            flood_candidates=remove_small(decreases&(y<water_db)&(~old_dark)&near_water,min_patch)
# Relative confidence proxy: margin above threshold, it's not calibrated statistical probability.
            strength=np.where(valid,np.maximum(0,np.abs(delta)-effective)/max(robust_sigma,0.1),0)
            strong=hotspots&(strength>=1.5)
# Per-pixel geodesic surface area in km2, used center latitude of each row.
            from pyproj import Geod
            geod=Geod(ellps='WGS84')
            h,w=x.shape
            row_areas=np.zeros(h,dtype='float64')
            
            if crs1 and crs1.is_geographic:
                for r in range(h):
                    a,b=t1*(0,r);c,d=t1*(1,r);e,f=t1*(1,r+1);g,j=t1*(0,r+1)
                    row_areas[r]=abs(geod.polygon_area_perimeter([a,c,e,g],[b,d,f,j])[0])/1e6
            else:
                # Projected CRS may use non-meter units; use pyproj CRS for axis conversion, had to ask Claude for this.
                from pyproj import CRS
                c=CRS.from_user_input(crs1)
                factor=c.axis_info[0].unit_conversion_factor if c.axis_info else 1
                row_areas[:]=abs(t1.a*t1.e-t1.b*t1.d)*factor**2/1e6
            def area(mask):return float(np.sum(np.count_nonzero(mask,axis=1)*row_areas))
            changed_area=area(hotspots);flood_area=area(flood_candidates)
            vv=np.concatenate([x[valid],y[valid]])
            low,high=np.percentile(vv,[2,98])
            before_img=stretched(x,low,high)
            after_img=stretched(y,low,high)
            classes=np.zeros(x.shape,dtype=np.uint8)
            classes[decreases]=1;classes[increases]=2;classes[flood_candidates]=3
            classes[~valid]=255
            display=np.dstack([after_img]*3)
            display[~valid]=0
            overlay_colors={1:np.array([0.08,0.73,1.0]),2:np.array([1.0,0.30,0.06]),3:np.array([1.0,0.95,0.12])}
            for code,color in overlay_colors.items():
                m=classes==code
                display[m]=display[m]*(1-opacity)+color*opacity
            display=(np.clip(display,0,1)*255).astype('uint8')
# Pixel-based area is calculated geodesically; no false precision implied.
        st.success('Analysis complete')
        c1,c2,c3,c4=st.columns(4)
        c1.metric('Flagged change pixels',f'{hotspots.sum():,}',f'{hotspots.sum()/valid.sum()*100:.1f}% of valid pixels')
        c2.metric('Screened change area',f'{changed_area:,.1f} km²')
        c3.metric('Possible water expansion',f'{flood_area:,.1f} km²',help='Exploratory darkening near pre-existing radar-dark areas; not verified flooding')
        c4.metric('Effective change threshold',f'{effective:.2f} dB')
        st.subheader('Response map — hotspots that stand out')
        st.image(display,caption='Yellow = possible water expansion; cyan = lower backscatter; orange = higher backscatter. Base = after-image radar texture.',use_container_width=True)
        st.subheader('Before / after / change')
        fig,axes=plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
        axes[0].imshow(x,cmap='gray',vmin=low,vmax=high);axes[0].set_title('Before (dB)')
        axes[1].imshow(y,cmap='gray',vmin=low,vmax=high);axes[1].set_title('After (dB)')
        im=axes[2].imshow(delta,cmap='RdBu_r',vmin=-8,vmax=8);axes[2].set_title('After − before (dB)')
        fig.colorbar(im,ax=axes[2],fraction=0.035,pad=0.02,label='dB')
        for ax in axes:ax.axis('off')
        st.pyplot(fig)
        st.subheader('3 · Error and uncertainty analysis')
        st.write('**Robust scene-wide difference spread (MAD-based):**',f'{robust_sigma:.2f} dB', ' • **Median scene-wide shift:**',f'{bias:+.2f} dB')
        st.write('**High-strength flagged pixels:**',f'{strong.sum():,}',f'({strong.sum()/max(hotspots.sum(),1)*100:.1f}% of hotspots).')
        st.info('The MAD-based spread is a descriptive measure of variability across this scene, **not** independently measured sensor noise or a confidence interval. It includes genuine surface changes, speckle, acquisition differences and calibration effects. The high-strength indicator is a relative signal score, not a probability of flooding.')
        fig2,ax=plt.subplots(figsize=(9,3))
        sample=values[::max(1,len(values)//100000)]
        ax.hist(np.clip(sample,-12,12),bins=100,color='#587f9c')
        ax.axvline(-effective,color='#087dc1',ls='--');ax.axvline(effective,color='#d35b36',ls='--')
        ax.set_xlabel('Radar backscatter difference (dB)');ax.set_ylabel('Sampled pixels');ax.set_title('Distribution of radar changes and screening thresholds')
        st.pyplot(fig2)
        st.caption('Screening assumptions: co-registered acquisitions, same polarization, comparable geometry and calibration. No field validation; water classification may confuse smooth soil, shadows, roads and other low-backscatter surfaces. Near-water distance is in pixels, not meters.')
# Coordinate-aware optional map overlay via image extent in latitude/longitude. see actual change as its overlayed
        if crs1:
            try:
                from rasterio.warp import reproject, Resampling
                from rasterio.transform import from_bounds
                import folium
                from streamlit_folium import st_folium
                wgs=transform_bounds(crs1,'EPSG:4326',*bounds,densify_pts=21)
                left,bottom,right,top=wgs
                st.subheader('Geographic locator')
                m = folium.Map(
                    location=[(bottom + top) / 2, (left + right) / 2],
                    zoom_start=10,
                    tiles="OpenStreetMap",
                    control_scale=True
                )   
#had to ask chatgot for this part as I had never done it myself
 # Input raster in this example is EPSG:4326, north-up. For other CRS, omit image overlay rather than misplace it.
                if crs1.to_epsg()==4326 and abs(t1.b)<1e-9 and abs(t1.d)<1e-9:
                    img=Image.fromarray(display).resize((min(w,1500),max(1,int(h*min(w,1500)/w))))
                    folium.raster_layers.ImageOverlay(image=np.array(img),bounds=[[bottom,left],[top,right]],opacity=0.82,name='SAR screening',interactive=True).add_to(m)
                    folium.LayerControl().add_to(m)
                st_folium(m,width=None,height=430,returned_objects=[])
            except Exception as exc:st.caption(f'Geographic map unavailable: {exc}')
        st.subheader('4 · Field validation (optional)')
        st.caption('Upload independent field observations with columns latitude, longitude, flooded (0 or 1). These must be independently verified, not generated from this SAR result.')
        validation=st.file_uploader('Field validation CSV',type=['csv'],key='validation')
        if validation:
            try:
                truth=pd.read_csv(validation)
                needed={'latitude','longitude','flooded'}
                if not needed.issubset(truth.columns):
                    
                    st.error('CSV requires latitude, longitude, flooded (0 or 1).')
                elif not set(truth['flooded'].dropna().unique()).issubset({0,1}):
                    st.error('The flooded column must contain only 0 and 1.')
                else:
                    transformer=Transformer.from_crs('EPSG:4326',crs1,always_xy=True)
                    xx,yy=transformer.transform(truth.longitude.to_numpy(),truth.latitude.to_numpy())
                    rr,cc=rowcol(t1,xx,yy)
                    rr=np.asarray(rr);cc=np.asarray(cc)
                    inside=(rr>=0)&(rr<h)&(cc>=0)&(cc<w)
                    keep=inside.copy();keep[inside]=valid[rr[inside],cc[inside]]
                    
                    if not np.any(keep):st.warning('No validation points overlap valid imagery.')
                    else:
                        observed=truth.flooded.to_numpy()[keep].astype(int)
                        predicted=flood_candidates[rr[keep],cc[keep]].astype(int)
                        tp=int(((observed==1)&(predicted==1)).sum());tn=int(((observed==0)&(predicted==0)).sum())
                        fp=int(((observed==0)&(predicted==1)).sum());fn=int(((observed==1)&(predicted==0)).sum())
                        c1,c2,c3=st.columns(3)
                        c1.metric('Validation points',len(observed))
                        c2.metric('Precision',f'{tp/(tp+fp):.1%}' if tp+fp else 'N/A')
                        c3.metric('Recall',f'{tp/(tp+fn):.1%}' if tp+fn else 'N/A')
                        st.write({'true_positive':tp,'true_negative':tn,'false_positive':fp,'false_negative':fn})
                        st.caption('Accuracy estimates apply only to these independently labeled points; spatial sampling bias may be substantial.')
# MODULE 4 — validation results(input a file .csv) and check for data validation
                        st.subheader("Module 4 — Validation Diagnostics")
                        test_mode = st.checkbox
                        "These observations are synthetic test data",
                        value=True,
                        help="Uncheck only when observations are independently verified."
                        precision = tp / (tp + fp) if (tp + fp) else None
                        recall = tp / (tp + fn) if (tp + fn) else None
                        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
                        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else None
                        if test_mode:
                            st.warning(
                                "SYNTHETIC TEST: These results verify software behavior, "
                                "not real-world flood detection accuracy."
                            )

                        c1, c2, c3, c4 = st.columns(4)
#check precision
                        c1.metric("Precision", f"{precision:.1%}" if precision is not None else "N/A")
                        c2.metric("Recall", f"{recall:.1%}" if recall is not None else "N/A")
                        c3.metric("F1 Score", f"{f1:.1%}" if f1 is not None else "N/A")
                        c4.metric("Accuracy", f"{accuracy:.1%}" if accuracy is not None else "N/A")

                        matrix = pd.DataFrame(
                            [
                                [tn, fp],
                                [fn, tp],
                            ],
                            index=["Actually not flooded", "Actually flooded"],
                            columns=["Predicted not flooded", "Predicted flooded"]
                        )

                        st.markdown("#### Confusion Matrix")
                        st.dataframe(matrix, use_container_width=True)

                        if tp + fp == 0:
                            st.info(
                                "No flood-positive predictions occurred at the validation points. "
                                "Precision is undefined."
                            )

                        if fn > 0:
                            st.warning(
                                f"{fn} labeled flooded locations were not flagged by the current "
                                "water-expansion screening settings."
                            )
            except Exception as exc:st.error(f'Could not validate CSV: {exc}')
        st.subheader('5 · Download results')
        c1,c2,c3=st.columns(3)
        overlay_buffer=io.BytesIO()
#export 
        Image.fromarray(display).save(overlay_buffer,format='PNG')
        c1.download_button('Download response overlay (PNG)',overlay_buffer.getvalue(),file_name='geonova_response_overlay.png',mime='image/png')
        c2.download_button('Download change (dB) GeoTIFF',tif_bytes(delta,profile,'After minus before (dB)'),file_name='geonova_change_db.tif',mime='image/tiff')
        c3.download_button('Download classification GeoTIFF',tif_bytes(classes,profile,'0 unflagged; 1 decrease; 2 increase; 3 possible water expansion; 255 nodata','uint8'),file_name='geonova_screening_classes.tif',mime='image/tiff')
        summary={'source':('NISAR HDF5 (user-provided; metadata-validated)' if mode.startswith('Geocoded') else 'User-supplied GeoTIFF (verify mission from filename/metadata)'),'polarization':band,'input_units':units,'valid_pixels':int(valid.sum()),'change_threshold_db':threshold,'effective_threshold_db':round(effective,3),'robust_scene_spread_db':round(robust_sigma,3),'median_scene_shift_db':round(bias,3),'flagged_pixels':int(hotspots.sum()),'flagged_area_km2':round(changed_area,3),'potential_water_expansion_km2':round(flood_area,3),'note':'Unvalidated radar change screening. Water expansion is a candidate only, not confirmed flooding.'}
        st.download_button('Download analysis summary (JSON)',json.dumps(summary,indent=2),file_name='geonova_analysis.json',mime='application/json')
    except Exception as exc:
        st.exception(exc)
else:
    st.info('Upload two comparable SAR acquisitions. For your Sentinel-1 test, use September 6 as Before and September 30 as After.')
