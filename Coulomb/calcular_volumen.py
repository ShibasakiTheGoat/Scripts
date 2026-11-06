"""Compute the complete bounded 3D Coulomb field for A/A and B/B.

The native fields are retained as float32; a documented 8-bit signed-log
encoding is used only for the interactive volume renderer.
"""
from pathlib import Path
import sys
import csv
import json
import gzip
import base64
import hashlib
import numpy as np
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
DEST = HERE.parent if HERE.name == 'codigo' else HERE.parent.parent / 'outputs/Coulomb_evento28_Mw4_ds5p3'
sys.argv = [sys.argv[0], str(DEST)]
import calcular_modelo as model

OUT = DEST / 'Volumen_3D_completo'
OUT.mkdir(exist_ok=True)
X = np.linspace(-2, 2, 65)
Y = np.linspace(-2, 2, 65)
DEPTHS = np.linspace(4.5, 13.5, 145)
E, N = np.meshgrid(X, Y)
arrays = dict(east_km=X, north_km=Y, depth_km_msl=DEPTHS)
payload = dict(shape=[len(X), len(Y), len(DEPTHS)],
    bounds_min=[float(X[0]), float(Y[0]), -float(DEPTHS[-1])],
    bounds_max=[float(X[-1]), float(Y[-1]), -float(DEPTHS[0])],
    encoding='uint8: 0=masked; 128=zero; u=round(127*clip(sign(c)*log10(1+abs(c)/.01)/log10(1001),-1,1))+128',
    planes={},source_depth_km=model.core.DEPTH,
    model=dict(Mw=4.0,M0_Nm=model.core.M0,fc_Hz=model.core.FC_HZ,stress_drop_spectral_MPa=model.DIM['spectral_circular_stress_drop_MPa']),
    origin=dict(latitude=model.core.LAT,longitude=model.core.LON))
checks = {}

nominal = list(csv.DictReader((model.DATA/'Coulomb_13_eventos_hipocentros.csv').open(encoding='utf-8-sig')))
for source, angles in model.core.PLANES.items():
    values = np.empty((len(DEPTHS),len(Y),len(X)),dtype=np.float32)
    excluded = 0
    for iz,z in enumerate(DEPTHS):
        tensor,bad,_ = model.field(E,N,float(z),angles)
        values[iz] = model.core.resolve(tensor,angles)[0]/1e5
        excluded += int(np.sum(bad))
        if iz%36==0:print(f'Volumen {source}: {iz+1}/{len(DEPTHS)} niveles',flush=True)
    arrays[f'dCFS_bar_{source}'] = values
    finite = np.isfinite(values)
    assert np.sum(finite)>values.size*.995
    # In a texture the z coordinate increases upward; reverse depth axis.
    v = values[::-1].copy()
    if not np.isfinite(v).all():
        nearest=distance_transform_edt(~np.isfinite(v),return_distances=False,return_indices=True)
        v[~np.isfinite(v)]=v[tuple(nearest[:,~np.isfinite(v)])]
    transformed = np.sign(v)*np.log10(1+np.abs(v)/.01)/np.log10(1001)
    encoded = np.where(np.isfinite(v),np.rint(127*np.clip(np.nan_to_num(transformed),-1,1))+128,0).astype(np.uint8)
    compressed = gzip.compress(encoded.tobytes(order='C'),compresslevel=9,mtime=0)
    events=[]
    rows=[r for r in nominal if r['source_plane']==source and r['evaluation_plane']==source]
    for row in rows:
        e,n,z = float(row['east_km']),float(row['north_km']),float(row['depth_km_msl'])
        tensor,_,_ = model.field(np.array([e]),np.array([n]),z,angles)
        exact=float(model.core.resolve(tensor[0],angles)[0]/1e5)
        assert np.isclose(exact,float(row['dCFS_bar']),rtol=1e-12,atol=1e-12)
        events.append(dict(id=int(row['event_id']),p=[e,n,-z],depth=z,cfs=exact))
    _,_,along,up = model.core.vectors(angles)
    half = model.DIM['length_km']/2
    corners=[]
    for a,b in [(-1,-1),(1,-1),(1,1),(-1,1)]:
        q=a*half*along+b*half*up
        corners.append([float(q[0]),float(q[1]),float(-model.core.DEPTH+q[2])])
    # Independent point checks against the exported volume, beyond texture encoding.
    maximum_relative = 0
    for ix,iy,iz in [(5,7,8),(20,25,40),(45,41,75),(52,55,130),(30,34,60)]:
        sig,mask,_=model.field(np.array([X[ix]]),np.array([Y[iy]]),float(DEPTHS[iz]),angles)
        exact=float(model.core.resolve(sig[0],angles)[0]/1e5)
        assert np.isclose(values[iz,iy,ix],exact,rtol=1e-6,atol=1e-8)
        maximum_relative=max(maximum_relative,abs(float(values[iz,iy,ix])-exact)/(abs(exact)+1e-12))
    payload['planes'][source]=dict(gzip=base64.b64encode(compressed).decode('ascii'),half_side_km=half,
        events=events,source_corners=corners,angles=list(angles),positive_count=sum(e['cfs']>0 for e in events))
    checks[source]=dict(grid_nodes=int(values.size),finite_nodes=int(finite.sum()),masked_edge_nodes=excluded,
        maximum_relative_error_float32_samples=maximum_relative,hypocenters_recomputed=13,
        exact_positive_count=sum(e['cfs']>0 for e in events),
        minimum_bar=float(values[finite].min()),maximum_bar=float(values[finite].max()),
        texture_raw_bytes=encoded.nbytes,texture_gzip_bytes=len(compressed),
        texture_sha256=hashlib.sha256(encoded.tobytes()).hexdigest())
    print(json.dumps({source:checks[source]},indent=2),flush=True)

np.savez_compressed(OUT/'Campo_Coulomb_3D_dos_planos.npz',**arrays)
(HERE/'volumen_payload.json').write_text(json.dumps(payload,separators=(',',':'),allow_nan=False),encoding='utf-8')
metadata=dict(bounds_east_km=[-2,2],bounds_north_km=[-2,2],bounds_depth_km_msl=[4.5,13.5],
    grid_shape_zyx=[145,65,65],spacing_m=[62.5,62.5,62.5],source_scenarios=['A/A','B/B'],
    points_per_source=int(len(X)*len(Y)*len(DEPTHS)),checks=checks,
    render_type='direct 3D volume ray casting, no horizontal or vertical slicing',
    render_color='signed logarithmic diverging scale saturated at +/-10 bar; original numerical fields are not clipped',
    render_mask='native NPZ retains NaN nodes; texture fills masked nodes by nearest finite neighbor for interpolation; renderer applies exact analytic <=10m edge exclusion',
    render_interpolation='trilinear interpolation of quantized signed-log texture; hypocentral values evaluated directly',
    spatial_scope='bounded computational volume, not the infinite extent of the elastic field',
    source_geometry='finite uniform-slip square with area from spectral corner frequency; sources independent',
    source_M0_Nm=model.core.M0,fc_Hz=model.core.FC_HZ,Vs_km_s=3.5,
    spectral_stress_drop_MPa=model.DIM['spectral_circular_stress_drop_MPa'],
    classification_scope='nominal locations; full location sensitivity remains in parent data package')
(OUT/'Parametros_y_verificacion_volumen.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print('JSON renderer bytes', (HERE/'volumen_payload.json').stat().st_size,flush=True)
