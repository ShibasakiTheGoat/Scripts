"""Reproducible event-28 scenario: supplied Mw/M0/fc, independent nodal sources.

Run with Python, numpy, matplotlib and pyproj. The bundled Okada C library
is used directly through ctypes. See LICENSE_OkadaPy_GPLv3.txt.
"""
from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import shutil
import sys
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Polygon, Rectangle
from pyproj import CRS, Transformer
import coulomb_core as core

HERE = Path(__file__).resolve().parent
DEST = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent
DATA = DEST / 'datos'
FIG = DEST / 'figuras'
for folder in [DATA, FIG]:
    folder.mkdir(parents=True, exist_ok=True)
DIM = core.dimensions()
EDGE_KM = .01
with core.SOURCE_CSV.open(encoding='utf-8-sig', newline='') as stream:
    CAT = list(csv.DictReader(stream))
POST = sorted([r for r in CAT if datetime.fromisoformat(r['origin_UTC']) >
               datetime.fromisoformat(core.EVENT['origin_UTC'])], key=lambda r: int(r['id']))
assert [int(r['id']) for r in POST] == list(range(29, 42))
CRS_LOCAL = CRS.from_proj4(f'+proj=aeqd +lat_0={core.LAT} +lon_0={core.LON} +datum=WGS84 +units=m')
TRANS = Transformer.from_crs('EPSG:4326', CRS_LOCAL, always_xy=True)


def coordinates(row):
    e, n = TRANS.transform(float(row['lon_WGS84']), float(row['lat_WGS84']))
    return e / 1000, n / 1000, float(row['depth_km_msl'])


def edge_distance(e, n, z, angles, source_depth=core.DEPTH):
    normal, _, along, up = core.vectors(angles)
    q = np.stack(np.broadcast_arrays(e, n, np.asarray(source_depth) - z), axis=-1)
    a, b, p = q @ along, q @ (-up), q @ normal
    hl, hw = DIM['length_km'] / 2, DIM['width_km'] / 2
    d1 = (np.abs(a) - hl)**2 + np.maximum(np.abs(b) - hw, 0)**2
    d2 = (np.abs(b) - hw)**2 + np.maximum(np.abs(a) - hl, 0)**2
    return np.sqrt(p * p + np.minimum(d1, d2))


def field(e, n, z, angles, source_depth=core.DEPTH):
    """Retain source interior; exclude only points within 10 m of an edge."""
    e, n = np.broadcast_arrays(np.asarray(e, dtype=float), np.asarray(n, dtype=float))
    normal = core.vectors(angles)[0]
    p = e * normal[0] + n * normal[1] + (source_depth - z) * normal[2]
    edge = edge_distance(e, n, z, angles, source_depth)
    bad = edge <= EDGE_KM
    ee, nn = e.copy(), n.copy()
    ee[bad], nn[bad] = 20, 20
    on = (np.abs(p) < 1e-7) & (~bad)
    if np.any(on):
        safee, safen = ee.copy(), nn.copy()
        safee[on], safen[on] = 20, 20
        tensor = core.evaluate(safee, safen, z, angles, DIM, source_depth)[0]
        eps = 1e-4  # 0.1 m symmetric limit on the extended source plane.
        plus = core.evaluate(ee[on] + eps * normal[0], nn[on] + eps * normal[1],
                             z - eps * normal[2], angles, DIM, source_depth)[0]
        minus = core.evaluate(ee[on] - eps * normal[0], nn[on] - eps * normal[1],
                              z + eps * normal[2], angles, DIM, source_depth)[0]
        tensor[on] = (plus + minus) / 2
    else:
        tensor = core.evaluate(ee, nn, z, angles, DIM, source_depth)[0]
    tensor[bad] = np.nan
    assert np.isfinite(tensor[~bad]).all()
    return tensor, bad, edge


def sign(value, excluded=False):
    return 'edge' if excluded else 'positive' if value > 0 else 'negative' if value < 0 else 'zero'


def save_csv(name, rows):
    with (DATA / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stencil(scale, count):
    theta = np.arange(count) * 2 * np.pi / count
    return np.vstack([np.zeros((1, 2)), scale * np.column_stack([np.sin(theta), np.cos(theta)])])


def calculate():
    checks = core.validate()
    nominal, friction_rows, locations, samples = [], [], [], []
    source_xy = stencil(float(core.EVENT['horizontal_se_conditional_km']), 4)
    source_z_scale = float(core.EVENT['depth_se_conditional_km'])
    for source, angles in core.PLANES.items():
        # Verify that the interior limit remains stable for this smaller source.
        for a, b in [(0, 0), (.1, .1), (-.15, .1)]:
            normal, _, along, up = core.vectors(angles)
            centre = a * along - b * up
            limits = []
            for eps in [.001, .0001]:
                pair = []
                for side in [-1, 1]:
                    q = centre + side * eps * normal
                    pair.append(core.evaluate(np.array([q[0]]), np.array([q[1]]),
                                               core.DEPTH - q[2], angles, DIM)[0][0])
                limits.append((pair[0] + pair[1]) / 2)
            error = float(np.max(np.abs(limits[0] - limits[1])) / np.max(np.abs(limits[1])))
            assert error < 1e-4
            checks[f'sided_limit_error_{source}_{a}_{b}'] = error
        for row in POST:
            event = int(row['id'])
            e, n, z = coordinates(row)
            tensor, bad, edge = field(np.array([e]), np.array([n]), z, angles)
            for evaluation, recv in core.PLANES.items():
                c, t, u = core.resolve(tensor[0], recv)
                nominal.append(dict(event_id=event, source_plane=source, evaluation_plane=evaluation,
                    origin_UTC=row['origin_UTC'], longitude_WGS84=float(row['lon_WGS84']),
                    latitude_WGS84=float(row['lat_WGS84']), depth_km_msl=z, east_km=e, north_km=n,
                    dCFS_MPa=float(c / 1e6), dCFS_bar=float(c / 1e5), dTau_bar=float(t / 1e5),
                    dSigma_n_tension_bar=float(u / 1e5), edge_distance_km=float(edge[0]),
                    distance_to_source_patch_km=float(core.distance_patch(e, n, z, angles, DIM)),
                    excluded_edge_10m=bool(bad[0]), sign=sign(c, bad[0])))
                for friction in [.2, .4, .6]:
                    value = core.resolve(tensor[0], recv, friction)[0] / 1e5
                    friction_rows.append(dict(event_id=event, source_plane=source,
                        evaluation_plane=evaluation, effective_friction=friction,
                        dCFS_bar=float(value), sign=sign(value, bad[0])))
            # Same location screening as the original delivery, at fixed new size.
            receiver_xy = stencil(float(row['horizontal_se_conditional_km']), 8)
            receiver_z_scale = float(row['depth_se_conditional_km'])
            relative = (receiver_xy[:, None, :] - source_xy[None, :, :]).reshape(-1, 2)
            values = {evaluation: [] for evaluation in core.PLANES}
            excluded = {evaluation: [] for evaluation in core.PLANES}
            for zr in [-1, 0, 1]:
                for zs in [-1, 0, 1]:
                    rd, sd = z + zr * receiver_z_scale, core.DEPTH + zs * source_z_scale
                    assert rd > 0 and sd > DIM['width_km'] / 2
                    tensor, bad, edge = field(e + relative[:, 0], n + relative[:, 1], rd, angles, sd)
                    for evaluation, recv in core.PLANES.items():
                        cfs = core.resolve(tensor, recv)[0] / 1e5
                        values[evaluation].extend(cfs.tolist())
                        excluded[evaluation].extend(bad.tolist())
                        for idx, value in enumerate(cfs):
                            samples.append(dict(event_id=event, source_plane=source,
                                evaluation_plane=evaluation, receiver_depth_step=zr,
                                source_depth_step=zs, receiver_horizontal_step=idx // 5,
                                source_horizontal_step=idx % 5, dCFS_bar=float(value),
                                excluded_edge_10m=bool(bad[idx])))
            for evaluation in core.PLANES:
                v = np.asarray(values[evaluation])
                finite = v[np.isfinite(v)]
                assert len(v) == 405
                stability = ('positive_all_sampled' if len(finite) == 405 and min(finite) > 0 else
                             'negative_all_sampled' if len(finite) == 405 and max(finite) < 0 else
                             'variable_or_edge')
                locations.append(dict(event_id=event, source_plane=source, evaluation_plane=evaluation,
                    event_horizontal_conditional_scale_km=float(row['horizontal_se_conditional_km']),
                    event_depth_conditional_scale_km=receiver_z_scale,
                    source_horizontal_conditional_scale_km=float(core.EVENT['horizontal_se_conditional_km']),
                    source_depth_conditional_scale_km=source_z_scale, sampled_configurations=len(v),
                    finite_configurations=len(finite), positive_configurations=int(np.sum(finite > 0)),
                    negative_configurations=int(np.sum(finite < 0)), excluded_configurations=int(np.sum(~np.isfinite(v))),
                    minimum_dCFS_bar=float(min(finite)), maximum_dCFS_bar=float(max(finite)),
                    sign_stability=stability))
            print(f'Evaluado plano fuente {source}, evento {event}', flush=True)
    summary = []
    for source in core.PLANES:
        for evaluation in core.PLANES:
            subset = [r for r in nominal if r['source_plane'] == source and r['evaluation_plane'] == evaluation]
            location = [r for r in locations if r['source_plane'] == source and r['evaluation_plane'] == evaluation]
            summary.append(dict(source=source, evaluation=evaluation,
                positive_ids=[r['event_id'] for r in subset if r['sign'] == 'positive'],
                negative_ids=[r['event_id'] for r in subset if r['sign'] == 'negative'],
                excluded_ids=[r['event_id'] for r in subset if r['sign'] == 'edge'],
                location_stable_positive_ids=[r['event_id'] for r in location if r['sign_stability'] == 'positive_all_sampled'],
                location_stable_negative_ids=[r['event_id'] for r in location if r['sign_stability'] == 'negative_all_sampled'],
                location_variable_ids=[r['event_id'] for r in location if r['sign_stability'] == 'variable_or_edge']))
    wide = []
    for row in POST:
        event = int(row['id'])
        out = dict(evento=event, profundidad_km=float(row['depth_km_msl']))
        for source in core.PLANES:
            for evaluation in core.PLANES:
                r = next(r for r in nominal if (r['event_id'], r['source_plane'], r['evaluation_plane']) ==
                         (event, source, evaluation))
                out[f'{source}_{evaluation}_dCFS_bar'] = r['dCFS_bar']
                out[f'{source}_{evaluation}_signo'] = r['sign']
        wide.append(out)
    save_csv('Coulomb_13_eventos_hipocentros.csv', nominal)
    save_csv('Tabla_13_eventos.csv', wide)
    save_csv('Sensibilidad_localizacion_resumen.csv', locations)
    save_csv('Sensibilidad_localizacion_21060_muestras.csv', samples)
    save_csv('Sensibilidad_friccion.csv', friction_rows)
    assert len(nominal) == 52 and len(locations) == 52 and len(samples) == 21060
    for row in nominal:
        assert np.isclose(row['dCFS_bar'], row['dTau_bar'] + .4 * row['dSigma_n_tension_bar'])
        assert not row['excluded_edge_10m']
    checks.update(nominal_evaluations=len(nominal), posterior_events=13,
        all_nominal_events_outside_edge_mask=True, location_samples=len(samples),
        moment_from_square_Nm=core.MU * DIM['area_km2'] * 1e6 * DIM['slip_m'],
        source_input_M0_Nm=core.M0)
    assert np.isclose(checks['moment_from_square_Nm'], core.M0, rtol=1e-14)
    metadata = dict(event_id=28, Mw_adopted=4.0, M0_input_Nm=core.M0,
        fc_adopted_Hz=core.FC_HZ, Vs_source_km_s=core.VS_KM_S, **DIM,
        spectral_radius_formula='r_m = 0.37 * Vs_m_s / fc_Hz',
        spectral_stress_drop_formula='Delta_sigma_MPa = 0.44 * M0_Nm / r_m^3 / 1e6',
        source_dimensions_method='square with same area as spectral equivalent circle; r directly from fc',
        stress_drop_scope='circular spectral estimate; not exact average shear stress drop of uniform-slip square',
        source_latitude_WGS84=core.LAT, source_longitude_WGS84=core.LON,
        source_depth_km_msl=core.DEPTH, source_origin_UTC=core.EVENT['origin_UTC'],
        nodal_planes_strike_dip_rake=core.PLANES, source_scenarios='A and B independently; not simultaneous',
        evaluation_orientations='hypothetical orientations A and B, no mapped receiver faults',
        shear_modulus_Pa=core.MU, poisson_ratio=core.NU, effective_friction=core.FRICTION,
        model='Okada 1992 isotropic homogeneous elastic half-space; uniform-slip finite square',
        source_centroid='placed at adjusted hypocenter', free_surface='horizontal mean sea level; no topography',
        equation='dCFS=dTau+mu_effective*dSigma_n, normal stress positive in tension/unclamping',
        classification='sign at individual 3D hypocenter; no classification from epicenter on a single depth map',
        edge_exclusion_km=EDGE_KM, interior='included, two-sided mean at +/-0.1m on source plane',
        location_screening='405 deterministic conditional-scale configurations per event and source/evaluation',
        screening_interpretation='ranges only, not confidence intervals or probabilities; covariance unavailable',
        limitations='fixed nodal angles, Vs, elastic properties and adopted fc; no dynamic stress or pore pressure model',
        input_status='hypothetical Mw4.0 scenario with supplied M0 and adopted fc; no new spectral fit',
        references=[dict(title='SEISAN manual, source spectral parameters',url='https://web.mst.edu/sgao/manuals/seisan.pdf'),
                    dict(title='Coulomb 3.3 user guide, Toda et al. 2011',url='https://pubs.usgs.gov/of/2011/1060/')])
    reference = DEST.parent / 'Entrega_Coulomb_evento28/datos/Coulomb_13_eventos_valores_nominales.csv'
    if reference.exists():
        old = list(csv.DictReader(reference.open(encoding='utf-8-sig')))
        comparison = []
        for r in nominal:
            prev = next(v for v in old if int(v['event_id']) == r['event_id'] and
                        v['source_plane'] == r['source_plane'] and v['evaluation_plane'] == r['evaluation_plane'])
            comparison.append(dict(event_id=r['event_id'], source_plane=r['source_plane'],
                evaluation_plane=r['evaluation_plane'], previous_3MPa_dCFS_bar=float(prev['dCFS_bar']),
                new_fc340_dCFS_bar=r['dCFS_bar'], previous_sign=prev['sign'], new_sign=r['sign']))
        save_csv('Comparacion_modelo_anterior_3MPa.csv', comparison)
        metadata['previous_model_comparison'] = '3 MPa circular dimensioning scale, same source location and mechanism'
    for filename, contents in [('Parametros_modelo.json', metadata), ('Resumen_resultados.json', summary),
                                ('Verificacion.json', checks)]:
        (DATA / filename).write_text(json.dumps(contents, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(dimensions=DIM, summary=summary), indent=2), flush=True)
    return nominal, locations, summary, checks


NAVY, MUTED, RED, BLUE = '#17354a', '#496475', '#b94034', '#22638f'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
NORM = SymLogNorm(linthresh=.01, linscale=.65, vmin=-10, vmax=10, base=10)
CMAP = plt.get_cmap('RdBu_r').copy()
CMAP.set_bad('#bbbbbb')
TICKS = [-10, -1, -.1, -.01, 0, .01, .1, 1, 10]
TICK_LABELS = ['−10', '−1', '−0.1', '−0.01', '0', '0.01', '0.1', '1', '10']


def export(fig, name):
    fig.savefig(FIG / f'{name}.png', dpi=240, facecolor='white')
    fig.savefig(FIG / f'{name}.svg', facecolor='white')
    plt.close(fig)


def polygon(angles):
    _, _, along, up = core.vectors(angles)
    half = DIM['length_km'] / 2
    return np.array([a * half * along + b * half * up for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])[:, :2]


def maps(nominal, summary):
    fig = plt.figure(figsize=(17.2, 11.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, .90], left=.06, right=.98,
                          bottom=.18, top=.80, wspace=.25, hspace=.30)
    grid = np.linspace(-1.6, 1.15, 551)
    E, N = np.meshgrid(grid, grid)
    arrays = dict(east_km=E, north_km=N, background_depth_km_msl=core.DEPTH)
    for i, (source, angles) in enumerate(core.PLANES.items()):
        stress, bad, _ = field(E, N, core.DEPTH, angles)
        arrays[f'tensor_ENU_Pa_source_{source}'] = stress
        for j, (evaluation, recv) in enumerate(core.PLANES.items()):
            ax = fig.add_subplot(gs[i, j])
            cfs = core.resolve(stress, recv)[0] / 1e5
            arrays[f'dCFS_bar_{source}_{evaluation}'] = cfs
            mesh = ax.pcolormesh(E, N, np.ma.masked_invalid(cfs), cmap=CMAP, norm=NORM,
                                 shading='auto', rasterized=True)
            ax.contour(E, N, np.ma.masked_invalid(cfs), levels=[0], colors='#525b62', linewidths=.5)
            ax.add_patch(Polygon(polygon(angles), fill=False, edgecolor=NAVY, lw=1.3, zorder=5))
            ax.scatter([0], [0], marker='*', s=140, facecolors=NAVY, edgecolors='white', lw=.5, zorder=9)
            ax.annotate('28', (0, 0), xytext=(8, 8), textcoords='offset points', fontweight='bold', fontsize=9)
            subset = [r for r in nominal if r['source_plane'] == source and r['evaluation_plane'] == evaluation]
            for r in subset:
                event, e, n = r['event_id'], r['east_km'], r['north_km']
                ax.scatter([e], [n], s=45, facecolors=RED if r['sign'] == 'positive' else BLUE,
                           edgecolors='white', lw=1.0, zorder=7)
                offset = {29: (6, -12), 34: (-20, -12), 38: (7, 6), 39: (-22, 6),
                          30: (7, -12), 36: (7, 5), 35: (5, -12)}.get(event, (6, 6))
                ax.annotate(str(event), (e, n), xytext=offset, textcoords='offset points', fontsize=8.5,
                    color=NAVY, fontweight='bold', zorder=8,
                    bbox=dict(boxstyle='round,pad=.12', facecolor='white', alpha=.93, edgecolor='none'),
                    arrowprops=dict(arrowstyle='-', lw=.55, color=NAVY))
            s = next(s for s in summary if (s['source'], s['evaluation']) == (source, evaluation))
            ax.text(.02, .98, f"En sus hipocentros: {len(s['positive_ids'])} positivos · {len(s['negative_ids'])} negativos",
                    transform=ax.transAxes, va='top', fontsize=8.6, color=NAVY,
                    bbox=dict(facecolor='white', alpha=.94, edgecolor='none', pad=3), zorder=10)
            ax.set(aspect='equal', xlabel='Este desde E28 (km)', ylabel='Norte desde E28 (km)',
                   xlim=(-1.6, 1.15), ylim=(-1.6, 1.15), xticks=[-1.5, -1, -.5, 0, .5, 1],
                   yticks=[-1.5, -1, -.5, 0, .5, 1])
            ax.set_title(f'Rompe el plano {source} · Se evalúa orientación {evaluation}', loc='left',
                         fontsize=11.8, fontweight='bold', color=NAVY, pad=9)
    ax = fig.add_subplot(gs[:, 2])
    ax.axis('off')
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.text(0, 1.02, 'Signo a la profundidad de cada evento', fontsize=12, fontweight='bold', color=NAVY)
    ax.text(0, .98, 'Columnas: plano que rompe / orientación evaluada.', fontsize=9.2, color=MUTED)
    xs = [.05, .22, .42, .58, .74, .90]
    for x, label in zip(xs, ['Evento', 'z (km)', 'A/A', 'A/B', 'B/A', 'B/B']):
        ax.text(x, .925, label, ha='center', va='center', fontsize=10, fontweight='bold', color=NAVY)
    for k, row in enumerate(POST):
        event, y = int(row['id']), .875 - k * .052
        ax.text(xs[0], y, str(event), ha='center', va='center', fontweight='bold', color=NAVY)
        ax.text(xs[1], y, f"{float(row['depth_km_msl']):.2f}", ha='center', va='center', fontsize=10, color=NAVY)
        for x, (source, evaluation) in zip(xs[2:], [('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')]):
            r = next(r for r in nominal if (r['event_id'], r['source_plane'], r['evaluation_plane']) ==
                     (event, source, evaluation))
            pos = r['sign'] == 'positive'
            ax.add_patch(Rectangle((x - .065, y - .020), .13, .040, facecolor=RED if pos else BLUE, edgecolor='none'))
            ax.text(x, y, '+' if pos else '−', ha='center', va='center', fontsize=13, fontweight='bold', color='white')
    ax.text(0, .16, 'Favorecidos / 13:', fontsize=10.5, fontweight='bold', color=NAVY)
    for x, s in zip(xs[2:], summary):
        ax.text(x, .11, str(len(s['positive_ids'])), ha='center', fontsize=16, fontweight='bold', color=RED)
    ax.text(0, .055, '+ ΔCFS > 0     − ΔCFS < 0', fontsize=10, color=NAVY)
    ax.text(0, .01, 'Clasificación nominal; incertidumbre en figura 3.', fontsize=9, color=MUTED)
    fig.suptitle('E28 · Cambio de esfuerzo de Coulomb para Mw 4.0 y f₀ = 3.40 Hz',
                 x=.06, y=.975, ha='left', fontsize=22, fontweight='bold', color=NAVY)
    fig.text(.06, .932, f"M₀ = 1.259 × 10¹⁵ N·m · β = 3.50 km/s · Δσ espectral = {DIM['spectral_circular_stress_drop_MPa']:.2f} MPa · Área = {DIM['area_km2']:.3f} km²",
             fontsize=12, color=MUTED)
    fig.text(.06, .895, f'Fondo: corte horizontal a {core.DEPTH:.3f} km. Color de cada círculo: signo en su profundidad individual.', fontsize=11.5, color=NAVY)
    fig.text(.06, .86, 'A: 46° / 66° / −100°     B: 249.44° / 25.89° / −68.69°  (rumbo / buzamiento / rake). Fuentes alternativas.', fontsize=10.5, color=MUTED)
    cax = fig.add_axes([.10, .107, .47, .018])
    cb = fig.colorbar(mesh, cax=cax, orientation='horizontal', extend='both', ticks=TICKS)
    cb.ax.set_xticklabels(TICK_LABELS)
    cb.set_label('ΔCFS (bar) · escala simétrica logarítmica; colores saturan en ±10 bar', fontsize=9.5)
    fig.text(.63, .119, 'Rojo: favorece el deslizamiento especificado.', fontsize=10.5, color=NAVY)
    fig.text(.63, .093, 'Azul: lo desfavorece. 1 bar = 0.1 MPa.', fontsize=10.5, color=NAVY)
    fig.text(.06, .038, 'Rectángulo: proyección de la ruptura. Orientaciones evaluadas hipotéticas; no se dispone de fallas receptoras cartografiadas.', fontsize=10, color=MUTED)
    fig.text(.06, .013, 'Interior incluido. Gris: puntos a ≤10 m de los bordes. Un punto puede diferir del fondo porque se evalúa a otra profundidad.', fontsize=10, color=MUTED)
    export(fig, 'Figura_1_Mapas_Coulomb_fc340')
    np.savez_compressed(DATA / 'Malla_y_tensores_corte_hipocentral.npz', **arrays)


def event_values_figure(nominal):
    fig, ax = plt.subplots(figsize=(13.8, 9.4))
    fig.subplots_adjust(left=.05, right=.97, bottom=.15, top=.82)
    ax.axis('off')
    labels = ['Evento', 'Profundidad\n(km)', 'Rompe A\nEvalúa A', 'Rompe A\nEvalúa B', 'Rompe B\nEvalúa A', 'Rompe B\nEvalúa B']
    rows, colors = [], []
    for event in range(29, 42):
        selected = [next(r for r in nominal if (r['event_id'], r['source_plane'], r['evaluation_plane']) ==
                         (event, s, ev)) for s, ev in [('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')]]
        rows.append([str(event), f"{selected[0]['depth_km_msl']:.3f}"] + [f"{r['dCFS_bar']:+.4f}" for r in selected])
        colors.append(['#f2f5f7', '#f2f5f7'] + ['#f9ddd7' if r['sign'] == 'positive' else '#dcebf7' for r in selected])
    table = ax.table(cellText=rows, colLabels=labels, cellColours=colors, colColours=[NAVY] * 6,
                     cellLoc='center', colLoc='center', loc='center', colWidths=[.10, .14, .19, .19, .19, .19])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.3)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        if r == 0:
            cell.set_text_props(color='white', weight='bold', fontsize=10.5)
            cell.set_height(.095)
        else:
            cell.set_text_props(color=NAVY, weight='bold' if c == 0 else 'normal')
    fig.suptitle('Valores de Coulomb en los 13 hipocentros posteriores', x=.05, y=.965,
                 ha='left', fontsize=23, fontweight='bold', color=NAVY)
    fig.text(.05, .905, 'ΔCFS en bar · Positivo: favorece el deslizamiento en la orientación evaluada; negativo: lo desfavorece.', fontsize=12, color=MUTED)
    fig.text(.05, .865, f"Mw 4.0 · M₀ 1.259 × 10¹⁵ N·m · f₀ 3.40 Hz · Δσ espectral {DIM['spectral_circular_stress_drop_MPa']:.2f} MPa · μ′ = 0.4", fontsize=12, color=MUTED)
    fig.text(.05, .09, 'Cada valor corresponde a la latitud, longitud y profundidad ajustadas de ese evento. 1 bar = 0.1 MPa.', fontsize=11, color=NAVY)
    fig.text(.05, .045, 'Los dos planos son alternativas de fuente; sus campos no se suman. Signo favorable expresa compatibilidad con el modelo.', fontsize=10.5, color=MUTED)
    export(fig, 'Figura_2_Valores_13_eventos')


def location_figure(nominal, locations):
    fig, axes = plt.subplots(1, 4, figsize=(15, 7.4), sharey=True)
    fig.subplots_adjust(left=.07, right=.98, bottom=.15, top=.78, wspace=.18)
    for ax, (source, evaluation) in zip(axes, [('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')]):
        for i, event in enumerate(range(29, 42)):
            row = next(r for r in locations if (r['event_id'], r['source_plane'], r['evaluation_plane']) == (event, source, evaluation))
            nom = next(r for r in nominal if (r['event_id'], r['source_plane'], r['evaluation_plane']) == (event, source, evaluation))
            stable = row['sign_stability'] == 'positive_all_sampled'
            ax.plot([row['minimum_dCFS_bar'], row['maximum_dCFS_bar']], [i, i],
                    c='#398061' if stable else '#a8aeb3', lw=2.2)
            ax.scatter([nom['dCFS_bar']], [i], c=RED if nom['sign'] == 'positive' else BLUE,
                       s=30, edgecolor='white', linewidth=.35, zorder=4)
        ax.axvline(0, color=NAVY, lw=.8)
        ax.set_xscale('symlog', linthresh=.1, linscale=.65)
        ax.set(xlim=(-1000, 1000), ylim=(12.7, -.7), xlabel='ΔCFS (bar)', yticks=range(13), yticklabels=range(29, 42))
        ax.set_xticks([-1000, -10, -.1, 0, .1, 10, 1000], labels=['−1000', '−10', '−0.1', '0', '0.1', '10', '1000'])
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(axis='x', alpha=.15)
        ax.set_title(f'Rompe {source} · Evalúa {evaluation}', loc='left', fontsize=11.5, fontweight='bold', color=NAVY)
    axes[0].set_ylabel('Evento posterior')
    fig.suptitle('Sensibilidad del signo a la localización', x=.07, y=.96, ha='left', fontsize=23, fontweight='bold', color=NAVY)
    fig.text(.07, .905, 'Puntos: valores nominales. Segmentos: mínimo y máximo de 405 configuraciones por evento y combinación.', fontsize=11, color=MUTED)
    fig.text(.07, .86, 'Se desplazan la fuente y cada hipocentro según las escalas condicionales del catálogo; tamaño y frecuencia fijos.', fontsize=11, color=MUTED)
    fig.text(.07, .068, 'Rangos de un ensayo determinista; no son intervalos de confianza ni probabilidades. Verde: positivo en todas las posiciones muestreadas.', fontsize=10, color=MUTED)
    fig.text(.07, .025, 'El signo positivo indica compatibilidad con la perturbación estática para la orientación elegida; no demuestra causalidad.', fontsize=10.5, color=MUTED)
    export(fig, 'Figura_3_Sensibilidad_localizacion')


def write_method(nominal, summary):
    text = ['# Modelo de Coulomb del evento 28: Mw 4.0 y frecuencia de esquina 3.40 Hz', '',
        '## Parámetros y dimensionamiento', '',
        'Escenario solicitado: Mw adoptada 4.0 y M₀ suministrado 1.259 × 10¹⁵ N·m. Se conserva exactamente ese momento redondeado; no se recalcula a partir de otra convención de Mw.',
        'La frecuencia de esquina adoptada es 3.40 Hz y la velocidad S del modelo espectral de E28 es 3.50 km/s. Se usa el mismo modelo de fuente circular de la estimación espectral: r = 0.37β/f₀ y Δσ = 0.44M₀/r³.', '',
        f"Resultados: radio {DIM['radius_equivalent_m']:.6f} m; Δσ espectral circular {DIM['spectral_circular_stress_drop_MPa']:.6f} MPa; área πr² = {DIM['area_km2']:.6f} km².",
        f"Para Okada se usa una falla cuadrada de igual área, L = W = {DIM['length_km']:.6f} km, con deslizamiento uniforme D = M₀/(μA) = {DIM['slip_m']:.6f} m y μ = 30 GPa. La caída espectral circular es una estimación equivalente y no la caída cortante media exacta del cuadrado con deslizamiento uniforme.", '',
        '## Geometría y cálculo', '',
        f'Centroide de fuente en el hipocentro ajustado: {core.LAT:.8f}° N, {core.LON:.8f}° E, {core.DEPTH:.8f} km bajo el nivel medio del mar.',
        'Plano A: 46.0° / 66.0° / −100.0°; plano B auxiliar: 249.437520° / 25.885716° / −68.693212° (rumbo/buzamiento/rake). Son dos fuentes alternativas e independientes; no se suman sus campos.',
        'Se emplea Okada (1992) en un semiespacio homogéneo, isotrópico y elástico; ν = 0.25, μ = 30 GPa. La superficie libre es horizontal a cota cero; no se incorpora topografía. El núcleo C procede de OkadaPy 0.0.1 (licencia GPLv3 incluida).',
        'ΔCFS = Δτ + μ′Δσn; μ′ = 0.4. Δτ es positivo en la dirección de deslizamiento elegida; Δσn es positivo en tensión/desconfinamiento. Los dos planos nodales se usan también como orientaciones hipotéticas de evaluación, lo que produce cuatro combinaciones. No se identifican como fallas receptoras cartografiadas.',
        'Se conserva el interior de la fuente. Únicamente se excluyen nodos a ≤10 m de un borde del rectángulo. Sobre el plano se utiliza la media de los límites a ±0.1 m. Ninguno de los 13 hipocentros nominales está dentro de esa franja excluida.', '',
        '## Clasificación y resultados', '',
        'Los 13 eventos son los posteriores al origen del evento 28 según la hora UTC del catálogo: eventos 29–41. El signo se calcula en la latitud, longitud y profundidad individual de cada evento. Un epicentro sobre un corte horizontal puede tener un color de fondo diferente al signo obtenido en su hipocentro.', '',
        '| Plano que rompe | Orientación evaluada | Positivos / 13 | Eventos positivos | Eventos negativos |',
        '|---|---|---:|---|---|']
    for row in summary:
        text.append(f"| {row['source']} | {row['evaluation']} | {len(row['positive_ids'])}/13 | {', '.join(map(str,row['positive_ids']))} | {', '.join(map(str,row['negative_ids']))} |")
    text += ['', 'ΔCFS positivo indica una perturbación que favorece el deslizamiento especificado en la orientación hipotética evaluada. Se interpreta como compatibilidad; no demuestra que el evento 28 causara cada evento posterior.', '',
        '## Sensibilidad de localización', '',
        'Se muestrean 9 posiciones horizontales por evento (centro y ocho direcciones), 5 por fuente (centro y cuatro direcciones) y tres profundidades para cada uno (−escala, centro, +escala): 9 × 5 × 3 × 3 = 405 configuraciones por evento/combinación, 21 060 evaluaciones en total. Las escalas son las columnas horizontal_se_conditional_km y depth_se_conditional_km del catálogo. Se conserva el tamaño nuevo de ruptura.',
        'Estos son rangos de un ensayo determinista; no son probabilidades ni intervalos de confianza. No hay covarianza disponible. No se perturban los ángulos nodales, β ni f₀. La sensibilidad de fricción 0.2/0.4/0.6 se calcula separadamente en los hipocentros nominales.', '']
    for row in summary:
        text.append(f"- Fuente {row['source']}, evaluación {row['evaluation']}: positivos en todas las posiciones muestreadas: {row['location_stable_positive_ids']}; negativos en todas: {row['location_stable_negative_ids']}; signo variable o muestra en borde: {row['location_variable_ids']}.")
    text += ['', '## Figuras y archivos', '',
        '- Figura 1: cuatro mapas, fuentes alternativas por fila y orientaciones evaluadas por columna. Fondo a la profundidad de E28; color de cada círculo según su propio hipocentro. Panel derecho con los signos de los 13 eventos.',
        '- Figura 2: los 52 valores de ΔCFS en bar, a las profundidades individuales.',
        '- Figura 3: sensibilidad a localización. Segmentos que cruzan cero representan cambios de signo entre las configuraciones muestreadas.',
        '- CSV: valores nominales, tabla de los 13 eventos, muestras y resumen de sensibilidad, ensayo de fricción y comparación con el modelo anterior de 3 MPa.',
        '- NPZ: malla y tensores ENU en Pa del corte hipocentral. El color es simétrico logarítmico, tramo lineal ±0.01 bar y saturación ±10 bar; los valores numéricos no se recortan.',
        '- Código y catálogo de entrada: permiten reproducir el cálculo. La entrega anterior se conserva sin cambios.', '',
        '## Referencias', '',
        '- SEISAN, parámetros espectrales y fórmulas de radio y caída: https://web.mst.edu/sgao/manuals/seisan.pdf',
        '- Toda, Stein, Sevilgen y Lin (2011). Coulomb 3.3 user guide: https://pubs.usgs.gov/of/2011/1060/', '']
    (DEST / 'Metodo_y_resultados.md').write_text('\n'.join(text), encoding='utf-8')


def package(checks):
    code = DEST / 'codigo'
    code.mkdir(exist_ok=True)
    def copy_if_changed(source, target):
        if source.resolve() == target.resolve():
            return
        if target.exists() and source.read_bytes() == target.read_bytes():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name in ['calcular_modelo.py', 'coulomb_core.py', 'source_data.csv', 'LICENSE_OkadaPy_GPLv3.txt']:
        copy_if_changed(HERE / name, code / name)
    if (HERE / 'pydeps').resolve() != (code / 'pydeps').resolve():
        for path in (HERE / 'pydeps').rglob('*'):
            if path.is_file():
                copy_if_changed(path, code / 'pydeps' / path.relative_to(HERE / 'pydeps'))
    (code / 'requirements.txt').write_text('numpy\nscipy\nmatplotlib\npyproj\n', encoding='utf-8')
    (code / 'README.txt').write_text('Python Windows x64; instalar requirements.txt. Ejecutar calcular_modelo.py desde esta carpeta.\nEl núcleo C compilado de OkadaPy se carga mediante ctypes; se incluyen fuentes C y licencia.\n', encoding='utf-8')
    manifest = {}
    for path in sorted(DEST.rglob('*')):
        if path.is_file() and path.name != 'SHA256.json':
            manifest[str(path.relative_to(DEST))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (DATA / 'SHA256.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    archive = DEST.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(DEST.rglob('*')):
            if path.is_file():
                zipped.write(path, path.relative_to(DEST.parent))
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
    print(f'Paquete completo: {archive}', flush=True)


if __name__ == '__main__':
    nominal, locations, summary, checks = calculate()
    maps(nominal, summary)
    print('Mapas completos', flush=True)
    event_values_figure(nominal)
    location_figure(nominal, locations)
    write_method(nominal, summary)
    package(checks)
