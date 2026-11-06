"""Reproducible four-scenario relocation. Units: km, s, degrees, WGS84.
Uses exact planar layered first arrivals, including critically refracted waves.
No spatial clustering objective and no use of old residuals in the inverse problem.
"""
from pathlib import Path
from collections import Counter, defaultdict
import json, csv, hashlib, time, argparse
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from openpyxl import load_workbook  # input reading only; not workbook authoring
from datetime import datetime, timedelta, timezone

ROOT=Path(__file__).resolve().parent
INP=ROOT/'inputs'/'Ultimo intento'
OUT=ROOT/'analysis'
OUT.mkdir(exist_ok=True)
LAT0=13.58; LON0=-86.49
A=6378.137; F=1/298.257223563; B=A*(1-F)
E2=F*(2-F)
N0=A/np.sqrt(1-E2*np.sin(np.deg2rad(LAT0))**2)
M0=A*(1-E2)/(1-E2*np.sin(np.deg2rad(LAT0))**2)**1.5
SCALE_X=N0*np.cos(np.deg2rad(LAT0))*np.pi/180
SCALE_Y=M0*np.pi/180
TOPS=np.array([0.,20.,35.,77.5])
BASE=np.array([[5.8,6.5,8.04,8.05],[3.46,3.85,4.48,4.5]])

def ll_to_xy(lat,lon): return np.array([(lon-LON0)*SCALE_X,(lat-LAT0)*SCALE_Y])
def xy_to_ll(x,y): return LAT0+y/SCALE_Y, LON0+x/SCALE_X

def geodesic(lat,lon,slat,slon):
    """Vectorized Vincenty inverse on WGS84. Return km and forward azimuth."""
    u1=np.arctan((1-F)*np.tan(np.deg2rad(lat)))
    u2=np.arctan((1-F)*np.tan(np.deg2rad(slat)))
    su1,cu1=np.sin(u1),np.cos(u1);su2,cu2=np.sin(u2),np.cos(u2)
    L=np.deg2rad(np.asarray(slon)-lon);lam=L.copy()
    for _ in range(20):
        sl,cl=np.sin(lam),np.cos(lam)
        ss=np.sqrt((cu2*sl)**2+(cu1*su2-su1*cu2*cl)**2)
        ss=np.maximum(ss,1e-15)
        cs=su1*su2+cu1*cu2*cl; sig=np.arctan2(ss,cs)
        sa=cu1*cu2*sl/ss;ca2=1-sa**2
        c2=cs-2*su1*su2/np.maximum(ca2,1e-15)
        C=F/16*ca2*(4+F*(4-3*ca2))
        nxt=L+(1-C)*F*sa*(sig+C*ss*(c2+C*cs*(-1+2*c2**2)))
        if np.max(np.abs(nxt-lam))<1e-12:lam=nxt;break
        lam=nxt
    usq=ca2*(A*A-B*B)/(B*B)
    aa=1+usq/16384*(4096+usq*(-768+usq*(320-175*usq)))
    bb=usq/1024*(256+usq*(-128+usq*(74-47*usq)))
    ds=bb*ss*(c2+bb/4*(cs*(-1+2*c2**2)-bb/6*c2*(-3+4*ss**2)*(-3+4*c2**2)))
    az=np.mod(np.rad2deg(np.arctan2(cu2*np.sin(lam),cu1*su2-su1*cu2*np.cos(lam))),360)
    return B*aa*(sig-ds),az

def vertical_segments(zstart,zend):
    lo=np.minimum(zstart,zend);hi=np.maximum(zstart,zend)
    lo=np.asarray(lo);hi=np.asarray(hi)
    lower=np.array([-100.,20.,35.,77.5])
    upper=np.array([20.,35.,77.5,1e4])
    return np.maximum(0,np.minimum(hi[...,None],upper)-np.maximum(lo[...,None],lower))

def travel(r,z,elev,vel,derivatives=False):
    r=np.asarray(r,dtype=float);elev=np.asarray(elev,dtype=float)
    rz=-elev
    h=vertical_segments(rz,z)
    layer=int(np.searchsorted(TOPS,z,side='right')-1)
    vs=vel[layer]
    if layer==0:
        L=np.sqrt(r*r+(z+elev)**2)
        tt=L/vel[0];p=r/(vel[0]*np.maximum(L,1e-12));q=(z+elev)/(vel[0]*np.maximum(L,1e-12))
    else:
        vm=np.max(np.where(h>1e-12,vel,0),axis=-1)
        low=np.zeros(len(r));high=(1-1e-12)/vm
        for _ in range(38):
            pp=(low+high)/2
            pv=pp[:,None]*vel
            xx=np.sum(h*pv/np.sqrt(np.maximum(1-pv*pv,1e-20)),axis=-1)
            low=np.where(xx<r,pp,low);high=np.where(xx>=r,pp,high)
        p=(low+high)/2
        tt=np.sum(h/(vel*np.sqrt(np.maximum(1-(p[:,None]*vel)**2,1e-20))),axis=-1)
        q=np.sqrt(np.maximum(1/vs**2-p*p,0))
    branch=np.zeros(len(r),dtype=int)
    for k in range(1,len(vel)):
        D=TOPS[k]
        if z>D+1e-10 or vel[k]<=np.max(vel[:k]):continue
        hh=vertical_segments(rz,D)+vertical_segments(z,D)
        pv=vel[:k]/vel[k];sq=np.sqrt(1-pv*pv)
        crit=np.sum(hh[:,:k]*pv/sq,axis=-1)
        head=r/vel[k]+np.sum(hh[:,:k]*np.sqrt(1/vel[:k]**2-1/vel[k]**2),axis=-1)
        ok=(r>=crit)&(head<tt)
        tt=np.where(ok,head,tt);p=np.where(ok,1/vel[k],p)
        q=np.where(ok,-np.sqrt(max(0,1/vs**2-1/vel[k]**2)),q)
        branch=np.where(ok,k,branch)
    if derivatives:return tt,p,q,branch
    return tt

def load_inputs():
    model_book=load_workbook(INP/'IASPEI CRUST.xlsx',data_only=True)['Hoja1']
    supplied=np.array([[model_book.cell(k,3).value for k in range(3,7)],[model_book.cell(k,4).value for k in range(3,7)]],dtype=float)
    boundaries=np.array([model_book.cell(k,2).value for k in range(3,7)],dtype=float)
    assert np.allclose(supplied,BASE) and np.allclose(boundaries,[20,35,77.5,120])
    events=[];records=[]
    counts=defaultdict(int)
    for line in (INP/'phase1.dat').read_text().splitlines():
        if not line.strip():continue
        a=line.split()
        if a[0]=='#':
            yy,mo,da,hr,mi=map(int,a[1:6]); sec=float(a[6])
            e={'id':int(a[-1]),'origin':(datetime(yy,mo,da,hr,mi,tzinfo=timezone.utc)+timedelta(seconds=sec)).isoformat(),'lat':float(a[7]),'lon':float(a[8]),'depth':float(a[9]),'magnitude':float(a[10]),'original_rms':float(a[13])}
            e['q']=ll_to_xy(e['lat'],e['lon']).tolist()+[e['depth'],0.]
            events.append(e)
        else:
            st,t,w,ph=a
            key=(e['id'],st,ph);counts[key]+=1
            records.append({'id':e['id'],'station':st,'phase':ph,'time':float(t),'weight':float(w),'kind':'real','reading':counts[key]})
    ws=load_workbook(INP/'Estaciones1.xlsx',data_only=True)['Hoja1']
    stas={};dups=[]
    for row in list(ws.values)[1:]:
        net,st,ch,name,lat,lon,el=row
        if not st:continue
        item={'station':st,'network':net,'lat':float(lat),'lon':float(lon),'elevation_m':float(el),'channels':ch or ''}
        if st in stas:dups.append({'station':st,'first':stas[st],'second':item})
        else:stas[st]=item
    # phase1.dat flattened every pick to weight 1. Recover explicit upstream
    # exclusions and archive residuals from the supplied Nordic S-file.
    nordic=defaultdict(list);event_number=0
    for line in (INP/'select1.out').read_text().splitlines():
        if line.startswith(' 2026') and len(line)>79 and line[79]=='1':event_number+=1
        if len(line)<79 or line[10] not in 'PS' or line[9] not in 'IE ' or line[1:6].strip()=='STAT':continue
        try:
            e=events[event_number-1];origin=datetime.fromisoformat(e['origin'])
            absolute=origin.replace(hour=int(line[18:20]),minute=int(line[20:22]),second=0,microsecond=0)+timedelta(seconds=float(line[22:28]))
            dt=(absolute-origin).total_seconds()
            if dt < -43200:dt+=86400
            nordic[(e['id'],line[1:6].strip(),line[10])].append({'nordic_weight':line[14].strip() or '0','archived_residual_s':float(line[63:68]) if line[63:68].strip() else None,'absolute_travel_s':dt,'automatic':line[15]=='A','polarity':line[16].strip()})
        except ValueError:continue
    for rec in records:
        source=nordic.get((rec['id'],rec['station'],rec['phase']),[])
        if len(source)>=rec['reading']:
            rec.update(source[rec['reading']-1])
            assert abs(rec['absolute_travel_s']-rec['time'])<=.15,rec
        else:
            rec.update({'nordic_weight':'0','archived_residual_s':None,'absolute_travel_s':None,'automatic':False,'polarity':'','source_note':'User correction: OCON P, ID 2, 2.05 s.'})
    w=load_workbook(INP/'Tiempos de viaje Teoricos_y_observados.xlsx',data_only=True)['Tiempos_completados']
    realmap={(r['id'],r['station'],r['phase'],r['reading']):r for r in records}
    synth=[];issues=[];black=0;color_reclassified=[]
    for row in w.iter_rows(min_row=5):
        st,ph,rd=[c.value for c in row[:3]]
        if not st:continue
        for j,c in enumerate(row[3:]):
            if c.value is None:continue
            eid=j+1;key=(eid,st,ph,int(rd));color=c.font.color.rgb if c.font.color and c.font.color.type=='rgb' else ''
            if color=='FF176CB4':
                if key in realmap and abs(realmap[key]['time']-float(c.value))<1e-6:
                    color_reclassified.append({'key':key,'time':float(c.value),'reason':'Present in phase1.dat. Classified as real despite blue font.'})
                elif key in realmap:issues.append({'problem':'synthetic overwrites real','key':key})
                else:synth.append({'id':eid,'station':st,'phase':ph,'time':float(c.value),'weight':1.,'kind':'theoretical','reading':int(rd)})
            elif color=='FF111111':
                black+=1
                if key not in realmap or abs(realmap[key]['time']-float(c.value))>1e-6:issues.append({'problem':'black mismatch','key':key,'time':c.value})
            else:issues.append({'problem':'unknown color','key':key,'color':str(color)})
    missing=set(r['station'] for r in records+synth)-set(stas)
    assert not missing, missing
    audit={'events':len(events),'real_picks':len(records),'theoretical_picks':len(synth),'real_stations':len(set(r['station'] for r in records)),'original_cells_verified':black,'workbook_mismatches':issues,'color_reclassified':color_reclassified,'duplicated_station_metadata':dups,'repeated_station_phases':[{'id':k[0],'station':k[1],'phase':k[2],'n':n,'times':[r['time'] for r in records if (r['id'],r['station'],r['phase'])==k]} for k,n in counts.items() if n>1],'mag41_event':[e for e in events if e['magnitude']==4.1]}
    assert not issues, issues
    audit['nordic_weight4_count']=sum(r['nordic_weight']=='4' for r in records)
    audit['weight4_picks']=[r for r in records if r['nordic_weight']=='4']
    audit['mag41_P_polarity_count']=sum(r['id']==28 and r['phase']=='P' and bool(r['polarity']) for r in records)
    audit['model_supplied']=BASE.tolist()
    audit['layer_bottoms_km']=boundaries.tolist()
    audit['file_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in INP.iterdir() if p.is_file()}
    write_json(OUT/'audit.json',audit)
    write_json(OUT/'inputs.json',{'events':events,'stations':stas,'real':records,'theoretical':synth})
    return events,stas,records,synth,audit

def write_json(path,obj):
    def clean(x):
        if isinstance(x,np.ndarray):return clean(x.tolist())
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,(np.integer,)):return int(x)
        if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
        return x
    path.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2))

class Data:
    def __init__(self,events,stas,records):
        self.events=events;self.stas=stas;self.records=records
        idx={e['id']:j for j,e in enumerate(events)}
        self.ei=np.array([idx[r['id']] for r in records])
        self.ph=np.array([0 if r['phase']=='P' else 1 for r in records])
        self.lat=np.array([stas[r['station']]['lat'] for r in records]);self.lon=np.array([stas[r['station']]['lon'] for r in records])
        self.elev=np.array([stas[r['station']]['elevation_m']/1000 for r in records])
        self.t=np.array([r['time'] for r in records]);self.w=np.array([r['weight'] for r in records])
        self.real=np.array([r['kind']=='real' for r in records]);self.original=np.array([e['q'] for e in events])
        self.usable=np.array([r.get('nordic_weight','0')!='4' for r in records])
        self.groups=[np.flatnonzero(self.ei==j) for j in range(len(events))]

    def predict_event(self,j,q,model,sel=None,jac=False):
        ix=self.groups[j] if sel is None else sel
        lat,lon=xy_to_ll(q[0],q[1])
        r,az=geodesic(lat,lon,self.lat[ix],self.lon[ix])
        pred=np.zeros(len(ix));der=np.zeros((len(ix),4));branches=np.zeros(len(ix),int)
        for ph in (0,1):
            pick=self.ph[ix]==ph
            if not pick.any():continue
            t,p,dz,br=travel(r[pick],q[2],self.elev[ix][pick],model[ph],True)
            pred[pick]=t+q[3];branches[pick]=br
            rad=np.deg2rad(az[pick])
            # Vincenty distance partials, converted through the local parameter scales.
            n=A/np.sqrt(1-E2*np.sin(np.deg2rad(lat))**2)
            m=A*(1-E2)/(1-E2*np.sin(np.deg2rad(lat))**2)**1.5
            der[pick,0]=-p*np.sin(rad)*(n*np.cos(np.deg2rad(lat))*np.pi/180)/SCALE_X
            der[pick,1]=-p*np.cos(rad)*(m*np.pi/180)/SCALE_Y
            der[pick,2]=dz;der[pick,3]=1.
        if jac:return pred,der,r,az,branches
        return pred

    def predict(self,qs,model):
        pred=np.zeros(len(self.t))
        for j,ix in enumerate(self.groups):pred[ix]=self.predict_event(j,qs[j],model)
        return pred

def locate(d,j,model,mask=None,start=None,robust=True,multi=False):
    ix=d.groups[j];ix=ix if mask is None else ix[mask[ix]]
    if len(ix)<5:return np.full(4,np.nan),None
    wt=np.sqrt(d.w[ix]);s=d.original[j].copy() if start is None else start.copy()
    bounds=([-35.,-35.,0.,-15.],[35.,35.,40.,15.])
    s=np.maximum(bounds[0],np.minimum(bounds[1],s))
    starts=[s]
    if multi:starts.extend([np.array([s[0],s[1],z,s[3]]) for z in [1.,8.,18.,28.]])
    best=None
    for init in starts:
        f=lambda q:(d.predict_event(j,q,model,ix)-d.t[ix])*wt
        jac=lambda q:d.predict_event(j,q,model,ix,True)[1]*wt[:,None]
        result=least_squares(f,init,jac=jac,bounds=bounds,loss='soft_l1' if robust else 'linear',f_scale=.5,x_scale=[3.,3.,5.,.5],max_nfev=120,ftol=1e-9,xtol=1e-8,gtol=1e-8)
        if best is None or result.cost<best.cost:best=result
    return best.x,best

def locate_all(d,model,mask=None,starts=None,robust=True,multi=False):
    return np.array([locate(d,j,model,mask,None if starts is None else starts[j],robust,multi)[0] for j in range(len(d.events))])

def select_outliers(d,qs,model):
    res=d.t-d.predict(qs,model);keep=d.usable.copy();reasons={int(k):'original_Nordic_weight4' for k in np.flatnonzero(~d.usable)};stats=[]
    thresholds={}
    for ph,name,minimum in [(0,'P',1.0),(1,'S',1.5)]:
        r=res[d.real&d.usable&(d.ph==ph)]
        mad=1.4826*np.median(np.abs(r-np.median(r)))
        thresholds[ph]=max(minimum,3*mad)
        stats.append({'phase':name,'mad_sigma_s':mad,'cutoff_s':thresholds[ph]})
    # Statistical exclusions are phase-specific. A bad S does not remove a good P.
    for st in sorted(set(r['station'] for r in d.records)):
        for ph in (0,1):
            ix=np.array([k for k,r in enumerate(d.records) if r['station']==st and d.ph[k]==ph and d.real[k] and d.usable[k]])
            if len(ix)<6:continue
            bad=np.abs(res[ix])>thresholds[ph]
            same=max(np.mean(res[ix]>thresholds[ph]),np.mean(res[ix]<-thresholds[ph]))
            if np.mean(bad)>=.6 and same>=.6:
                for k,r in enumerate(d.records):
                    if r['station']==st and d.ph[k]==ph:
                        keep[k]=False;reasons[k]='station_phase_systematic_residual'
    for k in np.flatnonzero(np.abs(res)>np.array([thresholds[p] for p in d.ph])):
        keep[k]=False;reasons.setdefault(int(k),'individual_residual_above_cutoff')
    # Guard against destroying sparse event geometry: restore least discordant picks.
    safeguards=[]
    for j,ix in enumerate(d.groups):
        def valid():
            kk=ix[keep[ix]&d.real[ix]];eligible=d.real[ix]&d.usable[ix]
            return len(kk)>=min(8,sum(eligible)) and len(set(d.records[k]['station'] for k in kk))>=min(4,len(set(d.records[k]['station'] for k in ix[eligible]))) and sum(d.ph[kk]==0)>=3 and sum(d.ph[kk]==1)>=1 and sum(~keep[ix]&eligible)<=max(1,int(.3*sum(eligible)))
        for k in sorted(ix[~keep[ix]&d.real[ix]&d.usable[ix]],key=lambda k:abs(res[k])):
            if valid():break
            keep[k]=True;reasons.pop(k,None);safeguards.append({'id':d.events[j]['id'],'record':int(k),'residual_s':float(res[k])})
    return keep,res,reasons,{'thresholds':stats,'restored_to_preserve_geometry':safeguards}

def model_from_params(v,complexity):
    m=BASE.copy()
    if complexity==2:
        m[0,:3]=BASE[0,:3]*v[0];m[1,:3]=BASE[1,:3]*v[1]
        # Keep the unsampled deepest layer monotonic with the fitted third layer.
        m[:,3]=np.maximum(BASE[:,3],m[:,2]+.01)
    else:
        m[0,:3]=v[:3];m[1,:3]=v[3:6]
        m[:,3]=np.maximum(BASE[:,3],m[:,2]+.01)
    return m

def params_base(complexity):return np.ones(2) if complexity==2 else np.r_[BASE[0,:3],BASE[1,:3]]
def model_bounds(complexity):
    if complexity==2:return np.array([.82,.82]),np.array([1.15,1.15])
    return np.array([4.5,5.9,7.6,2.65,3.4,4.1]),np.array([6.4,7.5,8.6,3.9,4.35,5.0])

def joint(d,mask,complexity=2,start_model=None,start_q=None,damping=.3,max_nfev=150,robust=True):
    n=len(d.events);v0=params_base(complexity) if start_model is None else np.asarray(start_model)
    qq=locate_all(d,model_from_params(v0,complexity),mask,robust=True) if start_q is None else start_q
    lo,hi=model_bounds(complexity);var=np.r_[qq.ravel(),v0];ix=np.flatnonzero(mask)
    # Regularization is expressed as a small number of additional residuals.
    # The very narrow cluster cannot resolve arbitrary layer thicknesses.
    def reg(v):
        m=model_from_params(v,complexity)
        ratios=m[0,:3]/m[1,:3]
        return np.r_[damping*(v-params_base(complexity)), 5*np.maximum(0,1.55-ratios),5*np.maximum(0,ratios-2.0),5*np.maximum(0,m[:,0]-m[:,1]),5*np.maximum(0,m[:,1]-m[:,2])]
    nr=len(reg(v0))
    def fun(var):
        qs=var[:4*n].reshape(n,4);v=var[4*n:];m=model_from_params(v,complexity)
        pred=d.predict(qs,m)
        return np.r_[(pred[ix]-d.t[ix])*np.sqrt(d.w[ix]),reg(v)]
    sparsity=lil_matrix((len(ix)+nr,len(var)),dtype=int)
    for row,k in enumerate(ix):sparsity[row,4*d.ei[k]:4*d.ei[k]+4]=1;sparsity[row,4*n:]=1
    sparsity[len(ix):,4*n:]=1
    lower=np.r_[np.tile([-35,-35,0,-15],n),lo];upper=np.r_[np.tile([35,35,40,15],n),hi]
    var=np.clip(var,lower+1e-8,upper-1e-8)
    result=least_squares(fun,var,jac_sparsity=sparsity.tocsr(),bounds=(lower,upper),loss='soft_l1' if robust else 'linear',f_scale=.5,x_scale=np.r_[np.tile([3,3,5,.5],n),np.full(complexity,.2)],max_nfev=max_nfev,ftol=1e-7,gtol=1e-6,xtol=1e-7)
    return result.x[:4*n].reshape(n,4),model_from_params(result.x[4*n:],complexity),result.x[4*n:],{'nfev':result.nfev,'cost':result.cost,'optimality':result.optimality,'success':result.success,'message':result.message,'damping':damping,'complexity':complexity,'loss':'soft_l1' if robust else 'linear'}

def checks():
    # Analytic homogeneous and two-layer references plus derivative consistency.
    rr=np.array([0.,5.,30.,100.]);el=np.array([0.,.5,1.,0.])
    actual=travel(rr,3.,el,np.full(4,6.))
    assert np.max(abs(actual-np.sqrt(rr**2+(3+el)**2)/6))<1e-10
    homogeneous_error=float(np.max(abs(actual-np.sqrt(rr**2+(3+el)**2)/6)))
    vel=np.array([5.8,6.5,8.04,8.05]);r=np.array([10.,120.,300.]);el=np.ones(3)*.5
    tt,p,q,br=travel(r,4.,el,vel,True)
    fd=(travel(r+.0001,4.,el,vel)-travel(r-.0001,4.,el,vel))/.0002
    fz=(travel(r,4.0001,el,vel)-travel(r,3.9999,el,vel))/.0002
    assert np.max(abs(fd-p))<1e-7 and np.max(abs(fz-q))<1e-7
    equatorial,az=geodesic(0.,0.,np.array([0.]),np.array([1.]))
    assert abs(equatorial[0]-111.319490793)<1e-7
    return {'homogeneous_max_error_s':homogeneous_error,'ray_derivative_max_error':float(max(np.max(abs(fd-p)),np.max(abs(fz-q)))),'equator_1deg_km':float(equatorial[0])}

def audit_run():
    events,stas,real,synth,audit=load_inputs();d=Data(events,stas,real+synth)
    pred=d.predict(d.original,BASE);res=d.t-pred
    audit['ray_checks']=checks()
    audit['original_geometry_residuals']={kind:{'n':int(sum(d.real if kind=='real' else ~d.real)),'rms_s':float(np.sqrt(np.mean(res[d.real if kind=='real' else ~d.real]**2))),'median_abs_s':float(np.median(abs(res[d.real if kind=='real' else ~d.real]))),'max_abs_s':float(np.max(abs(res[d.real if kind=='real' else ~d.real])))} for kind in ['real','theoretical']}
    write_json(OUT/'audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)

def diagnostics():
    events,stas,real,synth,audit=load_inputs()
    cases={}
    for name,records in [('IASPEI_real',real),('IASPEI_mixed',real+synth)]:
        d=Data(events,stas,records)
        started=time.time();qs=locate_all(d,BASE,mask=d.usable,robust=True,multi=True)
        keep,res,reasons,filter_info=select_outliers(d,qs,BASE)
        # Duplicate P alternatives are not independent. Keep the one with the
        # smallest preliminary residual, explicitly retain both original values.
        grouped=defaultdict(list)
        for k,r in enumerate(d.records):grouped[(r['id'],r['station'],r['phase'])].append(k)
        for key,ix in grouped.items():
            if len(ix)>1:
                best=min(ix,key=lambda k:abs(res[k]))
                for k in ix:
                    if k!=best:keep[k]=False;reasons.setdefault(int(k),'alternative_pick_same_station_phase')
        final=locate_all(d,BASE,keep,starts=qs,robust=False,multi=True)
        rr=d.t-d.predict(final,BASE)
        info={'qs':final,'preliminary_qs':qs,'keep':keep.tolist(),'preliminary_residuals':res,'reasons':{str(k):v for k,v in reasons.items()},'filter_info':filter_info,'rms_global':np.sqrt(np.mean(rr[keep]**2)),'rms_real':np.sqrt(np.mean(rr[keep&d.real]**2)),'rms_theoretical':np.sqrt(np.mean(rr[keep&~d.real]**2)) if (~d.real).any() else None,'excluded_real':int(sum(~keep&d.real)),'excluded_theoretical':int(sum(~keep&~d.real)),'depth_min':float(final[:,2].min()),'depth_max':float(final[:,2].max()),'rms_event_max':float(max(np.sqrt(np.mean(rr[ix[keep[ix]]]**2)) for ix in d.groups)),'time_s':time.time()-started}
        cases[name]=info
        write_json(OUT/(name+'_fit.json'),info)
        print(name,json.dumps({k:v for k,v in info.items() if k not in ['qs','preliminary_qs','keep','preliminary_residuals','reasons','filter_info']},default=float),flush=True)
        print('filter',filter_info,flush=True)
        print('excluded',[(d.records[int(k)]['id'],d.records[int(k)]['station'],d.records[int(k)]['phase'],round(float(res[int(k)]),3),v) for k,v in reasons.items()],flush=True)
    write_json(OUT/'initial_cases.json',cases)

def exploratory_models():
    events,stas,real,synth,audit=load_inputs()
    for name,records in [('real',real),('mixed',real+synth)]:
        d=Data(events,stas,records)
        previous=json.loads((OUT/('IASPEI_'+name+'_fit.json')).read_text())
        mask=np.array(previous['keep']);qq=np.array(previous['qs'])
        candidates=[]
        for complexity,shift in [(2,0.),(6,0.),(6,-.25),(6,.25)]:
            started=time.time();v=params_base(complexity)
            if complexity==6:v[:3]+=shift;v[3:]+=shift/1.75
            q,m,v,inf=joint(d,mask,complexity,v,qq,damping=.3,max_nfev=200)
            residuals=d.t-d.predict(q,m)
            # Re-evaluate outliers against the adjusted velocity model rather
            # than permanently deleting data on the IASPEI residual alone.
            mask2,r0,reasons,finfo=select_outliers(d,q,m)
            for k,r in enumerate(records):
                if previous['reasons'].get(str(k))=='alternative_pick_same_station_phase':mask2[k]=False;reasons.setdefault(int(k),'alternative_pick_same_station_phase')
            q,m,v,inf2=joint(d,mask2,complexity,v,q,damping=.3,max_nfev=200)
            q=locate_all(d,m,mask2,starts=q,robust=False,multi=True)
            residuals=d.t-d.predict(q,m)
            item={'complexity':complexity,'initial_shift':shift,'model':m,'params':v,'qs':q,'keep':mask2.tolist(),'reasons':{str(k):v for k,v in reasons.items()},'preliminary_residuals':r0,'filter_info':finfo,'optimizer':[inf,inf2],'rms_global':np.sqrt(np.mean(residuals[mask2]**2)),'rms_real':np.sqrt(np.mean(residuals[mask2&d.real]**2)),'excluded_real':int(sum(~mask2&d.real)),'excluded_theoretical':int(sum(~mask2&~d.real)),'time_s':time.time()-started}
            candidates.append(item)
            write_json(OUT/('models_'+name+'.json'),candidates)
            print(name,'candidate',len(candidates),json.dumps({k:item[k] for k in ['complexity','initial_shift','model','params','optimizer','rms_global','rms_real','excluded_real','excluded_theoretical','time_s']},default=lambda a:a.tolist()),flush=True)

def cross_validation():
    events,stas,real,synth,audit=load_inputs();d=Data(events,stas,real)
    rng=np.random.default_rng(20261008);fold=np.full(len(real),-1,int)
    for j,ix in enumerate(d.groups):
        sta=sorted(set(real[k]['station'] for k in ix[d.usable[ix]]));rng.shuffle(sta)
        # Exclude very sparse geometries from validation rather than remove
        # their only locating station. Other events hold out station blocks.
        if len(sta)<5 or sum(d.usable[ix])<9:continue
        for f in range(5):
            test=ix[np.array([real[k]['station'] in sta[f::5] for k in ix])&d.usable[ix]]
            train=ix[d.usable[ix]&~np.isin(ix,test)]
            if len(train)<6 or len(set(real[k]['station'] for k in train))<4 or sum(d.ph[train]==0)<3 or sum(d.ph[train]==1)<1:continue
            fold[test]=f
    candidates={x:[] for x in ['IASPEI','scale2','layer6']}
    for f in range(5):
        mask=d.usable&(fold!=f);test=np.flatnonzero(fold==f)
        for name,complexity in [('IASPEI',0),('scale2',2),('layer6',6)]:
            started=time.time()
            if complexity:
                q,m,v,inf=joint(d,mask,complexity,max_nfev=180)
            else:
                m=BASE;q=locate_all(d,m,mask,robust=True,multi=True);inf={}
            r=d.t-d.predict(q,m)
            item={'fold':f,'test_indices':test.tolist(),'test_residuals_s':r[test],'n':len(test),'rms_s':np.sqrt(np.mean(r[test]**2)),'median_abs_s':np.median(abs(r[test])),'model':m,'optimizer':inf,'time_s':time.time()-started}
            candidates[name].append(item)
            write_json(OUT/'cross_validation.json',candidates)
            print('CV',name,f,'n',len(test),'rms',round(item['rms_s'],4),'model',m.tolist(),'elapsed',round(item['time_s'],1),flush=True)
    summary={}
    for name,items in candidates.items():
        residuals=np.concatenate([x['test_residuals_s'] for x in items]);indices=np.concatenate([x['test_indices'] for x in items]);per_event=[]
        for j in range(len(events)):
            rr=residuals[d.ei[indices]==j]
            if len(rr):per_event.append({'id':events[j]['id'],'n':len(rr),'rms_s':float(np.sqrt(np.mean(rr**2)))})
        summary[name]={'test_phases':len(residuals),'test_events':len(per_event),'pooled_rms_s':float(np.sqrt(np.mean(residuals**2))),'median_abs_s':float(np.median(abs(residuals))),'median_event_rms_s':float(np.median([x['rms_s'] for x in per_event])),'per_event':per_event}
    write_json(OUT/'cv_summary.json',summary);print('CV SUMMARY',json.dumps(summary),flush=True)

def final_models():
    events,stas,real,synth,audit=load_inputs()
    cv=json.loads((OUT/'cv_summary.json').read_text())
    complexity=6 if cv['layer6']['pooled_rms_s']<cv['scale2']['pooled_rms_s'] else 2
    # Real-based model family is chosen on withheld real picks. The mixed
    # experiment uses the same dimension for a transparent comparison.
    for name,records in [('real',real),('mixed',real+synth)]:
        d=Data(events,stas,records)
        candidates=json.loads((OUT/('models_'+name+'.json')).read_text())
        valid=[c for c in candidates if c['complexity']==complexity]
        # Multiple seeds must share their accepted observations for a fair fit.
        candidate=min(valid,key=lambda c:c['rms_global'])
        mask=np.array(candidate['keep']);q=np.array(candidate['qs']);v=np.array(candidate['params'])
        q,m,v,info=joint(d,mask,complexity,v,q,damping=.3,max_nfev=300,robust=False)
        q=locate_all(d,m,mask,starts=q,robust=False,multi=True)
        pred=d.predict(q,m);r=d.t-pred
        result={**candidate,'qs':q,'model':m,'params':v,'final_optimizer':info,'rms_global':np.sqrt(np.mean(r[mask]**2)),'rms_real':np.sqrt(np.mean(r[mask&d.real]**2)),'model_selection_cv':cv,'selected_complexity':complexity}
        write_json(OUT/('Adjusted_'+name+'_fit.json'),result)
        print('FINAL',name,'model',m.tolist(),'rms',result['rms_global'],'rms_real',result['rms_real'],'optimization',info,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['audit','diagnostics','models','cv','finalmodels']);args=parser.parse_args()
    if args.mode=='audit':audit_run()
    elif args.mode=='diagnostics':diagnostics()
    elif args.mode=='models':exploratory_models()
    elif args.mode=='cv':cross_validation()
    elif args.mode=='finalmodels':final_models()
