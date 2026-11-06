"""Event 28: Okada finite source. Cartesian tensors use East, North, Up.

The compiled displacement kernel is OkadaPy 0.0.1 (GPLv3).
Recover strain from the documented displacement derivatives, then apply
Hooke's law explicitly. This avoids the tensor component ordering ambiguity
in the high-level stress wrapper of that version.
"""
from pathlib import Path
import sys, ctypes, json, csv, hashlib
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'pydeps'))
import numpy as np

MU=30e9
NU=.25
FRICTION=.4
MW=4.0
M0=1.259e15
STRESS_DROP_MPA=5.3
VS_KM_S=3.50
FC_HZ=0.37*VS_KM_S*1000/(0.44*M0/(STRESS_DROP_MPA*1e6))**(1/3)
SOURCE_CSV=HERE/'source_data.csv'
if not SOURCE_CSV.exists():
    SOURCE_CSV=Path('C:/Users/Jason/Desktop/Tesis/Mapa de Hipocentros, resultados/03_Ajustado_reales_41_eventos_con_correcciones_estacion.csv')
with SOURCE_CSV.open(encoding='utf-8-sig',newline='') as f:
    rows=[r for r in csv.DictReader(f) if r['id']=='28']
assert len(rows)==1
EVENT=rows[0]
LAT=float(EVENT['lat_WGS84']);LON=float(EVENT['lon_WGS84']);DEPTH=float(EVENT['depth_km_msl'])

def vectors(angles):
    p,d,l=np.radians(angles)
    a=np.array([np.sin(p),np.cos(p),0.])
    updip=np.array([-np.cos(p)*np.cos(d),np.sin(p)*np.cos(d),np.sin(d)])
    n=np.array([np.cos(p)*np.sin(d),-np.sin(p)*np.sin(d),np.cos(d)])
    s=np.cos(l)*a+np.sin(l)*updip
    return n,s,a,updip

def auxiliary(angles):
    n,s,_,_=vectors(angles)
    nb,sb=s.copy(),n.copy()
    if nb[2]<0: nb,sb=-nb,-sb
    p=np.arctan2(-nb[1],nb[0])
    d=np.arccos(nb[2])
    a=np.array([np.sin(p),np.cos(p),0.])
    up=np.array([-np.cos(p)*np.cos(d),np.sin(p)*np.cos(d),np.sin(d)])
    l=np.arctan2(sb@up,sb@a)
    return tuple([float(np.degrees(p)%360),float(np.degrees(d)),float(np.degrees(l))])

PLANES={'A':(46.,66.,-100.)}
PLANES['B']=auxiliary(PLANES['A'])

_library=HERE/'pydeps/okada/core/src/libokada.cp312-win_amd64.pyd'
if not _library.exists():
    import importlib.util
    spec=importlib.util.find_spec('okada')
    if spec is None:raise RuntimeError('Install okada==0.0.1 with Python 3.12 on Windows.')
    _library=next((Path(spec.origin).parent/'core/src').glob('libokada*.pyd'))
DLL=ctypes.CDLL(str(_library))
_disp=DLL.compute_okada_displacement
_ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
_disp.argtypes=[_ptr,_ptr,ctypes.c_int32,_ptr,ctypes.c_int32,
               ctypes.c_double,ctypes.c_double,ctypes.c_double,_ptr,ctypes.c_int32]
_disp.restype=None

def dimensions():
    # Source radius inferred from the adopted S-wave corner frequency.
    radius_m=0.37*VS_KM_S*1000/FC_HZ
    area_m2=np.pi*radius_m**2
    side_km=np.sqrt(area_m2)/1000
    slip_m=M0/(MU*area_m2)
    # Spectral Brune approximation used by SEISAN. This is a circular
    # source estimate, not an exact average stress drop of the square.
    drop_mpa=0.44*M0/radius_m**3/1e6
    return dict(corner_frequency_Hz=FC_HZ,shear_velocity_km_s=VS_KM_S,
                spectral_circular_stress_drop_MPa=drop_mpa,
                radius_equivalent_m=radius_m,length_km=side_km,
                width_km=side_km,area_km2=area_m2/1e6,slip_m=slip_m)

def rotation(angles):
    p=np.radians(angles[0])
    return np.array([[np.sin(p),-np.cos(p),0.],
                     [np.cos(p),np.sin(p),0.],[0.,0.,1.]])

def raw_source(angles,dim,depth=DEPTH,slip_factor=1.):
    _,dip,rake=angles;d,l=np.radians([dip,rake])
    hl=dim['length_km']/2;hw=dim['width_km']/2
    D=dim['slip_m']*slip_factor
    return np.array([-hl,hw*np.cos(d),hl,hw*np.cos(d),100.,
                     -D*np.cos(l),D*np.sin(l),dip,
                     depth-hw*np.sin(d),depth+hw*np.sin(d)],dtype=np.float64)

def evaluate(east,north,eval_depth,angles,dim=None,source_depth=DEPTH,slip_factor=1.):
    if dim is None:dim=dimensions()
    e,n=np.broadcast_arrays(east,north);shape=e.shape
    R=rotation(angles)
    x=np.ascontiguousarray((R[0,0]*e+R[1,0]*n).ravel(),dtype=np.float64)
    y=np.ascontiguousarray((R[0,1]*e+R[1,1]*n).ravel(),dtype=np.float64)
    out=np.zeros((x.size,12))
    _disp(x,y,x.size,raw_source(angles,dim,source_depth,slip_factor),1,
          2*MU*(1+NU),NU,float(eval_depth),out.ravel(),2)
    local_grad=out[:,3:].reshape(-1,3,3).transpose(0,2,1)/1000
    grad=np.einsum('ij,njk,lk->nil',R,local_grad,R)
    strain=(grad+grad.transpose(0,2,1))/2
    lame=2*MU*NU/(1-2*NU)
    tensor=2*MU*strain+lame*np.trace(strain,axis1=1,axis2=2)[:,None,None]*np.eye(3)
    u=out[:,:3]@R.T
    return tensor.reshape(shape+(3,3)),u.reshape(shape+(3,)),grad.reshape(shape+(3,3))

def resolve(tensor,angles,friction=FRICTION):
    n,s,_,_=vectors(angles)
    traction=tensor@n
    normal=traction@n # positive tension / unclamping
    shear=traction@s # positive in prescribed receiver slip direction
    return shear+friction*normal,shear,normal

def distance_patch(east,north,eval_depth,angles,dim=None):
    if dim is None:dim=dimensions()
    normal,slip,a,up=vectors(angles)
    q=np.stack(np.broadcast_arrays(east,north,np.asarray(DEPTH-eval_depth)),axis=-1)
    along=q@a;downdip=q@(-up);perp=q@normal
    return np.sqrt(perp**2+np.maximum(np.abs(along)-dim['length_km']/2,0)**2+
                   np.maximum(np.abs(downdip)-dim['width_km']/2,0)**2)

def kelvin_stress(q,angles):
    # Independent full-space point moment solution; only for deep-source tests.
    normal,slip,*_=vectors(angles)
    moment=M0*(np.outer(normal,slip)+np.outer(slip,normal))
    q=np.asarray(q)*1000;r=np.linalg.norm(q,axis=-1)
    v=q@moment.T;h=np.sum(q*v,axis=-1)
    eye=np.eye(3);xx=q[:,:,None]*q[:,None,:]
    vx=v[:,:,None]*q[:,None,:];xv=q[:,:,None]*v[:,None,:]
    grad=(2*(1-2*NU)*(moment/r[:,None,None]**3-3*vx/r[:,None,None]**5)
          +3*eye*h[:,None,None]/r[:,None,None]**5
          +6*xv/r[:,None,None]**5-15*xx*h[:,None,None]/r[:,None,None]**7)/(16*np.pi*MU*(1-NU))
    eps=(grad+grad.transpose(0,2,1))/2
    return 2*MU*eps+(2*MU*NU/(1-2*NU))*np.trace(eps,axis1=1,axis2=2)[:,None,None]*eye

def validate():
    result={}
    ma=[]
    for key,angles in PLANES.items():
        n,s,*_=vectors(angles)
        assert abs(n@s)<1e-12 and np.isclose(n@n,1) and np.isclose(s@s,1)
        ma.append(np.outer(n,s)+np.outer(s,n))
    result['nodal_moment_tensor_error']=float(np.max(np.abs(ma[0]-ma[1])))
    assert result['nodal_moment_tensor_error']<1e-12
    e=np.array([1.3,-2.,3.2,-1.6]);n=np.array([2.1,1.4,-1.1,-2.4])
    for key,angles in PLANES.items():
        sig,u,g=evaluate(e,n,DEPTH+.3,angles)
        h=.0001
        fd=np.empty_like(g)
        fd[:,:,0]=(evaluate(e+h,n,DEPTH+.3,angles)[1]-evaluate(e-h,n,DEPTH+.3,angles)[1])/(2000*h)
        fd[:,:,1]=(evaluate(e,n+h,DEPTH+.3,angles)[1]-evaluate(e,n-h,DEPTH+.3,angles)[1])/(2000*h)
        fd[:,:,2]=(evaluate(e,n,DEPTH+.3-h,angles)[1]-evaluate(e,n,DEPTH+.3+h,angles)[1])/(2000*h)
        err=float(np.max(np.abs(g-fd))/np.max(np.abs(g)))
        result[f'gradient_fd_error_{key}']=err
        assert err<1e-5,(key,'gradient',err,g,fd)
        surface=evaluate(e,n,0.,angles)[0]
        err=float(np.max(np.abs(surface[:,:,2]))/np.max(np.abs(surface)))
        result[f'free_surface_traction_error_{key}']=err
        assert err<1e-7,(key,'surface',err)
        doubled=evaluate(e,n,DEPTH+.3,angles,slip_factor=2)[0]
        assert np.allclose(doubled,2*sig,rtol=1e-10,atol=1e-5)
        # Source-plane stress must oppose imposed slip at rupture centre.
        centre=evaluate(np.array([0.]),np.array([0.]),DEPTH+.0001,angles)[0]
        c,shear,normal=resolve(centre,angles)
        result[f'centre_shear_MPa_{key}']=float(shear[0]/1e6)
        assert shear[0]<0
        # Independent deep-source, far-field comparison to Kelvin solution.
        de=np.array([12.,-14.,18.]);dn=np.array([16.,11.,-13.])
        deep=evaluate(de,dn,1000.,angles,source_depth=1000.)[0]
        kelvin=kelvin_stress(np.stack([de,dn,np.zeros(3)],axis=1),angles)
        err=float(np.max(np.abs(deep-kelvin))/np.max(np.abs(kelvin)))
        result[f'kelvin_deep_farfield_error_{key}']=err
        assert err<.01,(key,'kelvin',err,deep,kelvin)
    result['moment_reconstruction_Nm']=MU*dimensions()['area_km2']*1e6*dimensions()['slip_m']
    result['source_hash_sha256']=hashlib.sha256(SOURCE_CSV.read_bytes()).hexdigest()
    return result

if __name__=='__main__':
    print(json.dumps({'planes':PLANES,'dimension_base':dimensions(),'checks':validate()},indent=2))
