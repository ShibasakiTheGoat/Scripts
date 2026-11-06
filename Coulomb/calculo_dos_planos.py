"""Mw4.0, adopted circular spectral drop5.3MPa: A/A and B/B only."""
from pathlib import Path
from datetime import datetime
import ast
import csv
import json
import hashlib
import shutil
import sys
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

HERE=Path(__file__).resolve().parent
DEST=(HERE.parent if HERE.name=='codigo' else HERE.parent.parent/'outputs/Coulomb_evento28_Mw4_ds5p3')
sys.argv=[sys.argv[0],str(DEST)]
import calcular_modelo as model
core=model.core
OUT=DEST/'Mapas_dos_planos'
OUT.mkdir(parents=True,exist_ok=True)
DIM=model.DIM
assert np.isclose(DIM['spectral_circular_stress_drop_MPa'],5.3,rtol=1e-13)
checks=core.validate()
assert np.isclose(core.MU*DIM['area_km2']*1e6*DIM['slip_m'],core.M0,rtol=1e-14)

nominal=[];locations=[];samples=[];friction_rows=[];summary=[]
source_xy=model.stencil(float(core.EVENT['horizontal_se_conditional_km']),4)
source_z_scale=float(core.EVENT['depth_se_conditional_km'])
for source,angles in core.PLANES.items():
    # Each tensor is resolved ONLY on its own source orientation.
    for row in model.POST:
        event=int(row['id']);e,n,z=model.coordinates(row)
        tensor,bad,edge=model.field(np.array([e]),np.array([n]),z,angles)
        c,t,u=core.resolve(tensor[0],angles)
        assert not bad[0] and np.isfinite([c,t,u]).all()
        assert np.isclose(c,t+.4*u)
        nominal.append(dict(event_id=event,source_plane=source,evaluation_plane=source,
            origin_UTC=row['origin_UTC'],longitude_WGS84=float(row['lon_WGS84']),latitude_WGS84=float(row['lat_WGS84']),
            depth_km_msl=z,east_km=e,north_km=n,dCFS_MPa=float(c/1e6),dCFS_bar=float(c/1e5),
            dTau_bar=float(t/1e5),dSigma_n_tension_bar=float(u/1e5),edge_distance_km=float(edge[0]),
            distance_to_source_patch_km=float(core.distance_patch(e,n,z,angles,DIM)),
            excluded_edge_10m=bool(bad[0]),sign=model.sign(c,bad[0])))
        for friction in [.2,.4,.6]:
            value=float(core.resolve(tensor[0],angles,friction)[0]/1e5)
            friction_rows.append(dict(event_id=event,source_plane=source,evaluation_plane=source,
                effective_friction=friction,dCFS_bar=value,sign=model.sign(value)))
        receiver_xy=model.stencil(float(row['horizontal_se_conditional_km']),8)
        relative=(receiver_xy[:,None,:]-source_xy[None,:,:]).reshape(-1,2)
        receiver_z_scale=float(row['depth_se_conditional_km'])
        values=[]
        for zr in [-1,0,1]:
            for zs in [-1,0,1]:
                rd=z+zr*receiver_z_scale;sd=core.DEPTH+zs*source_z_scale
                tensor,bad,edge=model.field(e+relative[:,0],n+relative[:,1],rd,angles,sd)
                cfs=core.resolve(tensor,angles)[0]/1e5
                values.extend(cfs.tolist())
                for idx,value in enumerate(cfs):
                    samples.append(dict(event_id=event,source_plane=source,evaluation_plane=source,
                        receiver_depth_step=zr,source_depth_step=zs,receiver_horizontal_step=idx//5,
                        source_horizontal_step=idx%5,dCFS_bar=float(value),excluded_edge_10m=bool(bad[idx])))
        v=np.array(values);finite=v[np.isfinite(v)];assert len(v)==405
        stability=('positive_all_sampled' if len(finite)==405 and min(finite)>0 else
                   'negative_all_sampled' if len(finite)==405 and max(finite)<0 else 'variable_or_edge')
        locations.append(dict(event_id=event,source_plane=source,evaluation_plane=source,
            sampled_configurations=len(v),finite_configurations=len(finite),
            positive_configurations=int(np.sum(finite>0)),negative_configurations=int(np.sum(finite<0)),
            excluded_configurations=int(np.sum(~np.isfinite(v))),minimum_dCFS_bar=float(min(finite)),
            maximum_dCFS_bar=float(max(finite)),sign_stability=stability,
            event_horizontal_conditional_scale_km=float(row['horizontal_se_conditional_km']),
            event_depth_conditional_scale_km=receiver_z_scale,
            source_horizontal_conditional_scale_km=float(core.EVENT['horizontal_se_conditional_km']),
            source_depth_conditional_scale_km=source_z_scale))
    rows=[r for r in nominal if r['source_plane']==source]
    loc=[r for r in locations if r['source_plane']==source]
    summary.append(dict(source=source,evaluation=source,angles=angles,
        positive_ids=[r['event_id'] for r in rows if r['sign']=='positive'],
        negative_ids=[r['event_id'] for r in rows if r['sign']=='negative'],
        location_stable_positive_ids=[r['event_id'] for r in loc if r['sign_stability']=='positive_all_sampled'],
        location_stable_negative_ids=[r['event_id'] for r in loc if r['sign_stability']=='negative_all_sampled'],
        location_variable_ids=[r['event_id'] for r in loc if r['sign_stability']=='variable_or_edge']))
    print(f'Plano {source}: {len(summary[-1]["positive_ids"])}/13 positivos',flush=True)
assert len(nominal)==26 and len(samples)==10530
assert all(r['source_plane']==r['evaluation_plane'] for r in nominal+samples)
for filename,rows in [('Coulomb_13_eventos_hipocentros.csv',nominal),('Sensibilidad_localizacion_resumen.csv',locations),
                      ('Sensibilidad_localizacion_10530_muestras.csv',samples),('Sensibilidad_friccion.csv',friction_rows)]:
    model.save_csv(filename,rows)
wide=[]
for event in range(29,42):
    a=next(r for r in nominal if r['event_id']==event and r['source_plane']=='A')
    b=next(r for r in nominal if r['event_id']==event and r['source_plane']=='B')
    wide.append(dict(evento=event,profundidad_km=a['depth_km_msl'],plano_1_dCFS_bar=a['dCFS_bar'],
                     signo_plano_1=a['sign'],plano_2_dCFS_bar=b['dCFS_bar'],signo_plano_2=b['sign']))
model.save_csv('Tabla_13_eventos.csv',wide)

# Reuse exactly the established layout in the user's reference, without running its selection code.
layout=(HERE/'seleccionar_presentacion.py').read_text(encoding='utf-8')
tree=ast.parse(layout)
function=ast.get_source_segment(layout,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='figure'))
function=function.replace("receiver = model.core.PLANES['B']","receiver = angles")
function=function.replace("r['evaluation_plane'] == 'B'","r['evaluation_plane'] == source")
function=function.replace("title='Escenario principal: plano A como fuente' if source=='A' else 'Alternativa: plano B como fuente'","title='Primer plano nodal' if source=='A' else 'Segundo plano nodal'")
function=function.replace("'Orientación de evaluación única B: rumbo 249.44°, buzamiento 25.89°, rake −68.69°.'","f'Fuente y referencia: rumbo {angles[0]:.2f}°, buzamiento {angles[1]:.2f}°, rake {angles[2]:.2f}°.'")
function=function.replace("'Mw 4.0 · M₀ = 1.259 × 10¹⁵ N·m · f₀ = 3.40 Hz · Δσ espectral = 10.03 MPa · μ′ = 0.4'","f'Mw 4.0 · M₀ = 1.259 × 10¹⁵ N·m · f₀ = {core.FC_HZ:.2f} Hz · Δσ espectral = 5.30 MPa · μ′ = 0.4'")
function=function.replace("'Rectángulo: proyección de la fuente finita. Orientación de evaluación hipotética; fuente A y fuente B son alternativas.'","'Rectángulo: proyección de la ruptura. El esfuerzo se resuelve con el mismo rumbo, buzamiento y rake de la fuente.'")
function=function.replace("'Selección exploratoria por coincidencia de signo con los mismos 13 eventos. La incertidumbre de localización se conserva en el análisis.'","'Los dos planos son fuentes alternativas, calculadas por separado. Se conserva la sensibilidad a la incertidumbre de localización.'")
function=function.replace('dpi=270','dpi=300')
assert "receiver = angles" in function and '3.40 Hz' not in function
exec(compile(function,str(HERE/'seleccionar_presentacion.py'),'exec'))
for plane in ['A','B']:
    rows=figure(plane,f'Mapa_plano_{plane}_Mw4_ds5p3')
    with (OUT/f'Valores_13_eventos_plano_{plane}.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

metadata=dict(event_id=28,Mw_adopted=core.MW,M0_input_Nm=core.M0,stress_drop_adopted_MPa=5.3,
    fc_derived_Hz=core.FC_HZ,**DIM,Vs_source_km_s=core.VS_KM_S,
    fc_status='Derived from adopted Mw/M0 and stress drop; not a new measured spectral fit',
    spectral_formulas='r=(0.44*M0/Delta_sigma_Pa)^(1/3); fc=0.37*Vs/r',
    source_dimensions_method='Uniform-slip square with same area as equivalent spectral circle',
    stress_drop_scope='Circular spectral sizing proxy, not exact mean shear-stress drop of uniform-slip square',
    source_latitude_WGS84=core.LAT,source_longitude_WGS84=core.LON,source_depth_km_msl=core.DEPTH,
    source_origin_UTC=core.EVENT['origin_UTC'],nodal_planes_strike_dip_rake=core.PLANES,
    scenarios='A source / A reference; B source / B reference, independent alternatives; no crossed orientations',
    shear_modulus_Pa=core.MU,poisson_ratio=core.NU,effective_friction=core.FRICTION,
    model='Okada 1992 homogeneous isotropic elastic halfspace, uniform-slip finite square',
    free_surface='Mean sea level, no topography',source_centroid='At adjusted hypocenter',
    equation='dCFS=dTau+mu_effective*dSigma_n; normal tension positive',
    count='At individual 3D hypocenters; background is a horizontal cut at source depth',
    edge_exclusion_km=model.EDGE_KM,source_interior='Included, symmetric limits +/-0.1m on plane',
    location_screening='405 deterministic conditional location configurations per event and scenario',
    location_screening_limits='Not probabilities or confidence intervals; covariance unavailable',
    colormap='RdBu_r, symlog linear threshold0.01bar; limits +/-10bar; numerical values not clipped')
checks.update(nominal_evaluations=26,location_samples=10530,posterior_events=13,
    source_reference_always_same=True,all_nominal_events_outside_edge_mask=True,
    moment_reconstruction_Nm=core.MU*DIM['area_km2']*1e6*DIM['slip_m'])
for filename,obj in [('Parametros_modelo.json',metadata),('Resumen_resultados.json',summary),('Verificacion.json',checks)]:
    (model.DATA/filename).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

text=['# Evento28: Mw4.0 y caída espectral adoptada5.3MPa','',
    'Se adoptan Mw4.0, M0=1.259×10¹⁵ N·m, Vs=3.5km/s y Δσ=5.3MPa. La frecuencia se deriva de esos valores: '
    f'{core.FC_HZ:.12f}Hz, redondeada a2.75Hz en las figuras. No corresponde a un nuevo ajuste de sismogramas.',
    f'Radio equivalente {DIM["radius_equivalent_m"]:.6f}m; área {DIM["area_km2"]:.9f}km²; '
    f'cuadrado de {DIM["length_km"]*1000:.6f}m de lado; deslizamiento {DIM["slip_m"]*100:.6f}cm.',
    'La caída circular espectral se usa para dimensionar la fuente; no se identifica con la caída cortante media exacta del rectángulo uniforme.','',
    'Cada plano nodal se calcula por separado como fuente y como referencia para resolver esfuerzos. '
    'No se suman fuentes, no se usan combinaciones cruzadas ni se añaden orientaciones. '
    'Se supone deslizamiento potencial paralelo a la geometría de la fuente en cada escenario; no se conocen las fallas de los eventos posteriores.',
    'Círculos amarillos: epicentros. El fondo es un corte a8.166km, mientras la tabla y el conteo usan las profundidades individuales.',
    'Se conserva el interior de la ruptura. Únicamente se excluye una franja de10m en los bordes singulares; ningún hipocentro nominal cae en ella.',
    'Los colores se saturan en±10bar, sin recortar valores numéricos. 1bar=0.1MPa.','']
for s in summary:
    text += [f'Plano {s["source"]}: {len(s["positive_ids"])}/13 positivos. Positivos: {s["positive_ids"]}. Negativos: {s["negative_ids"]}.',
             f'Sensibilidad: positivos en las405 configuraciones: {s["location_stable_positive_ids"]}; negativos estables: {s["location_stable_negative_ids"]}; signo variable o muestras en borde: {s["location_variable_ids"]}.','']
text+=['El muestreo de localización es determinista, con escalas condicionales del catálogo. No proporciona probabilidades ni intervalos de confianza. '
       'La concordancia de signo indica compatibilidad con el modelo, no demuestra desencadenamiento causal.','',
       'Referencias: SEISAN https://web.mst.edu/sgao/manuals/seisan.pdf ; Coulomb3.3 https://pubs.usgs.gov/of/2011/1060/ ; '
       'orientación paralela a la fuente https://pubs.usgs.gov/publication/70192219']
(DEST/'Metodo_y_resultados.md').write_text('\n\n'.join(text),encoding='utf-8')
code=DEST/'codigo';code.mkdir(exist_ok=True)
for name in ['coulomb_core.py','calcular_modelo.py','seleccionar_presentacion.py','recalcular_dos_planos.py',
             'source_data.csv','LICENSE_OkadaPy_GPLv3.txt']:
    if (HERE/name).resolve()!=(code/name).resolve():shutil.copy2(HERE/name,code/name)
if (HERE/'pydeps').resolve()!=(code/'pydeps').resolve():
    shutil.copytree(HERE/'pydeps',code/'pydeps',dirs_exist_ok=True)
(code/'requirements.txt').write_text('numpy\nmatplotlib\npyproj\n',encoding='utf-8')
(code/'README.txt').write_text('Python Windows x64. Instalar requirements.txt y ejecutar recalcular_dos_planos.py. '
    'Los otros módulos son dependencias; no ejecutar seleccionar_presentacion.py ni calcular_modelo.py directamente.\n',encoding='utf-8')
manifest={str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='SHA256.json'}
(model.DATA/'SHA256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
with zipfile.ZipFile(DEST.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(DEST.rglob('*')):
        if p.is_file():z.write(p,p.relative_to(DEST.parent))
with zipfile.ZipFile(DEST.with_suffix('.zip')) as z:assert z.testzip() is None
print(json.dumps({'parameters':metadata,'summary':summary},indent=2,ensure_ascii=False))
