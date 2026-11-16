# Geonova SAR Response Explorer — integrated experimental build

## Launch
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Test with your existing files
Upload the September 6 and September 30 VV single-band GeoTIFFs; choose **GeoTIFF** and **linear power**. Start with 3 dB threshold, median 3, minimum patch 25. The application generates before/after, dB difference, bright change overlays, exploratory water-darkening candidates, area estimates, robust scene-spread metrics, map overlays and exports.

## NISAR HDF5 support (experimental)
The importer lists numeric 2D datasets in a supplied HDF5 file. It only accepts real-valued *geocoded* layers with x/y coordinate vectors alongside the selected layer and a verified EPSG attribute. Raw radar-coordinate SLC and complex layers are **not** supported. Not all NISAR product structures are covered; actual NISAR ingestion is unverified without a real sample. NISAR GCOV geocoded backscatter is the intended starting point. Choose matching acquisition geometry, polarization, grid and units. Cross-mission subtraction is not scientifically valid without harmonization.

## Independent validation
Optional CSV: `latitude,longitude,flooded` where flooded is independently observed 0/1. Computes TP/TN/FP/FN, precision and recall. Do not label SAR-derived guesses as field observations.

## Important limits
- Radar darkening near existing water is only a candidate screening heuristic; not confirmed flood, depth, rescue route, or ground movement.
- MAD-based variation includes real changes and is not a calibrated error bar or probability.
- The application requires identical pixel grids for the two inputs; it will refuse misaligned images rather than silently subtracting them.
- Area estimates are approximate and require valid georeferencing.
- NISAR HDF5 importer needs testing on actual GCOV products. This build has not been validated for operational emergency response.
