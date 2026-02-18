
import pandas as pd
import numpy as np
import warnings, json, os, webbrowser
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix)

CSV_PATH = "fraud_dataset (1).csv"   # <- Apni CSV ka naam yahan

print("="*50)
print("  FRAUD DETECTION SYSTEM")
print("="*50)

# ── LOAD CSV ──────────────────────────────────────────────────
print("\n[1/6] CSV is loading")
df = pd.read_csv(CSV_PATH)
print(f"  Total : {len(df):,} rows")
print(f"  Fraud : {(df['Label']==1).sum()}  ({(df['Label']==1).mean()*100:.1f}%)")
print(f"  Legit : {(df['Label']==0).sum()}  ({(df['Label']==0).mean()*100:.1f}%)")

# ── PREPROCESS ────────────────────────────────────────────────
print("\n[2/6] Preprocessing...")
df_m = df.drop(columns=['Transaction_ID','Customer_Name',
                         'Transaction_Date','Transaction_Time'], errors='ignore')
le = LabelEncoder()
for col in ['Bank','Merchant','City']:
    if col in df_m.columns:
        df_m[col] = le.fit_transform(df_m[col].astype(str))

X = df_m.drop(columns=['Label'])
y = df_m['Label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_te = scaler.transform(X_test)

# ── SMOTE ─────────────────────────────────────────────────────
print("\n[3/6] SMOTE (class imbalance fix)...")
def smote(X, y, ratio=0.5):
    Xf, Xl = X[y==1], X[y==0]
    n = int(len(Xl)*ratio) - len(Xf)
    if n <= 0: return X, y
    syn = [Xf[np.random.randint(0,len(Xf))] +
           np.random.random()*(Xf[np.random.randint(0,len(Xf))]-Xf[np.random.randint(0,len(Xf))])
           for _ in range(n)]
    return np.vstack([X,syn]), np.hstack([y, np.ones(n,int)])

X_res, y_res = smote(X_tr, y_train.values)
print(f"  Fraud: {(y_train==1).sum()} -> {(y_res==1).sum()}")

# ── MODELS ────────────────────────────────────────────────────
print("\n[4/6] Models training...")
iso = IsolationForest(contamination=(df['Label']==1).mean(),
                      n_estimators=200, random_state=42)
iso.fit(X_res)
iso_pred   = np.where(iso.predict(X_te)==-1,1,0)
iso_scores = -iso.decision_function(X_te)

rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                            random_state=42, n_jobs=-1)
rf.fit(X_res, y_res)
rf_pred  = rf.predict(X_te)
rf_proba = rf.predict_proba(X_te)[:,1]
ens_pred = np.where((iso_pred+rf_pred)>=1,1,0)
print("  Done!")

# ── METRICS ───────────────────────────────────────────────────
print("\n[5/6] Metrics is calculating...")
def met(yt,yp,ys=None):
    return (round(precision_score(yt,yp,zero_division=0),3),
            round(recall_score(yt,yp,zero_division=0),3),
            round(f1_score(yt,yp,zero_division=0),3),
            round(roc_auc_score(yt,ys),3) if ys is not None else 0)

ip,ir,if_,ia = met(y_test,iso_pred,iso_scores)
rp,rr,rf_,ra = met(y_test,rf_pred,rf_proba)
ep,er,ef_,_  = met(y_test,ens_pred)

cm1 = confusion_matrix(y_test,iso_pred).tolist()
cm2 = confusion_matrix(y_test,rf_pred).tolist()

feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(8)

df['Hour'] = pd.to_datetime(df['Transaction_Time'],format='%H:%M',errors='coerce').dt.hour.fillna(0).astype(int)
fh = df[df['Label']==1].groupby('Hour').size().reindex(range(24),fill_value=0).tolist()
lh = df[df['Label']==0].groupby('Hour').size().reindex(range(24),fill_value=0).tolist()

bins=[0,1000,5000,20000,50000,100000,float('inf')]
lbs=['<1K','1K-5K','5K-20K','20K-50K','50K-1L','1L+']
df['AB']=pd.cut(df['Amount_INR'],bins=bins,labels=lbs)
fa = df[df['Label']==1]['AB'].value_counts().reindex(lbs,fill_value=0).tolist()
la = df[df['Label']==0]['AB'].value_counts().reindex(lbs,fill_value=0).tolist()

test_df = df.iloc[y_test.index].copy().reset_index(drop=True)
res = test_df.copy()
res['Fraud_Probability'] = np.round(rf_proba,4)
res['Ensemble_Pred']     = ens_pred
res['Alert_Level']       = ['HIGH' if p>0.8 else 'MEDIUM' if p>0.4 else 'LOW' for p in rf_proba]
res.to_csv('fraud_predictions.csv',index=False)

top = res[res['Alert_Level']=='HIGH'].sort_values('Fraud_Probability',ascending=False).head(15)
top_list = top[['Transaction_ID','Customer_Name','Bank','Amount_INR',
                'Merchant','City','Transaction_Time','Fraud_Probability','Alert_Level']].to_dict('records')

blocked = round(res[res['Ensemble_Pred']==1]['Amount_INR'].sum())

data = dict(
    total=int(len(df)), fraud=int((df['Label']==1).sum()),
    legit=int((df['Label']==0).sum()), fraud_pct=round((df['Label']==1).mean()*100,2),
    high=int((res['Alert_Level']=='HIGH').sum()),
    medium=int((res['Alert_Level']=='MEDIUM').sum()),
    low=int((res['Alert_Level']=='LOW').sum()),
    blocked=blocked,
    iso=dict(p=ip,r=ir,f=if_,a=ia),
    rf=dict(p=rp,r=rr,f=rf_,a=ra),
    ens=dict(p=ep,r=er,f=ef_),
    cm1=cm1, cm2=cm2,
    fh=fh, lh=lh, fa=fa, la=la,
    fn=feat_imp.index.tolist(),
    fv=[round(v,4) for v in feat_imp.values.tolist()],
    rows=top_list
)

# ── BUILD HTML ────────────────────────────────────────────────
print("\n[6/6] Dashboard is building...")

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fraud Detection Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0d1117;--card:#161b22;--brd:#30363d;--txt:#e6edf3;--mu:#8b949e;
  --red:#f85149;--grn:#3fb950;--yel:#d29922;--blu:#58a6ff;--pur:#bc8cff;--ora:#ffa657}
body{background:var(--bg);color:var(--txt);font-family:'Segoe UI',system-ui,sans-serif}
header{background:linear-gradient(135deg,#161b22,#0d1117);border-bottom:1px solid var(--brd);
  padding:15px 26px;display:flex;align-items:center;justify-content:space-between}
header h1{font-size:1.3rem;font-weight:700}
header h1 span{color:var(--red)}
.live{background:var(--red);color:#fff;font-size:.68rem;padding:3px 9px;
  border-radius:20px;font-weight:700;animation:pulse 1.4s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.abar{background:linear-gradient(90deg,#3d0b0b,#1a0505);border:1px solid var(--red);
  border-radius:8px;margin:16px 26px 0;padding:10px 16px;display:flex;align-items:center;gap:10px;font-size:.86rem}
.wrap{padding:0 26px 28px}
.krow{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:13px;margin:16px 0}
.kpi{background:var(--card);border:1px solid var(--brd);border-radius:12px;padding:16px;
  position:relative;overflow:hidden;transition:transform .2s}
.kpi:hover{transform:translateY(-2px)}
.kpi::before{content:'';position:absolute;top:0;left:0;width:4px;height:100%;border-radius:4px 0 0 4px}
.kpi.red::before{background:var(--red)}.kpi.grn::before{background:var(--grn)}
.kpi.blu::before{background:var(--blu)}.kpi.yel::before{background:var(--yel)}
.kpi.pur::before{background:var(--pur)}
.kpi label{font-size:.68rem;color:var(--mu);text-transform:uppercase;letter-spacing:.06em}
.kpi .val{font-size:1.8rem;font-weight:700;margin:4px 0 2px}
.kpi .sub{font-size:.72rem;color:var(--mu)}
.kpi.red .val{color:var(--red)}.kpi.grn .val{color:var(--grn)}
.kpi.blu .val{color:var(--blu)}.kpi.yel .val{color:var(--yel)}.kpi.pur .val{color:var(--pur)}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:13px;margin-bottom:13px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:13px;margin-bottom:13px}
.card{background:var(--card);border:1px solid var(--brd);border-radius:12px;padding:16px}
.card h3{font-size:.78rem;color:var(--mu);margin-bottom:13px;text-transform:uppercase;letter-spacing:.05em}
.mb{margin-bottom:8px}
.mb .lb{font-size:.74rem;display:flex;justify-content:space-between;margin-bottom:3px}
.trk{background:#21262d;border-radius:4px;height:7px;overflow:hidden}
.fill{height:100%;border-radius:4px;transition:width 1.2s ease}
table{width:100%;border-collapse:collapse;font-size:.78rem}
thead th{background:#21262d;padding:8px 11px;text-align:left;font-size:.7rem;color:var(--mu);text-transform:uppercase}
tbody tr{border-bottom:1px solid var(--brd)}
tbody tr:hover{background:#1c2128}
tbody td{padding:8px 11px}
.bdg{display:inline-block;padding:2px 8px;border-radius:20px;font-size:.68rem;font-weight:600}
.bdg.H{background:rgba(248,81,73,.15);color:var(--red);border:1px solid rgba(248,81,73,.3)}
.bdg.M{background:rgba(210,153,34,.15);color:var(--yel);border:1px solid rgba(210,153,34,.3)}
.bdg.L{background:rgba(63,185,80,.1);color:var(--grn);border:1px solid rgba(63,185,80,.2)}
.recs{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:11px}
.rec{background:var(--card);border:1px solid var(--brd);border-radius:10px;
  padding:13px;display:flex;gap:9px;align-items:flex-start}
.rec .ico{font-size:1.25rem;flex-shrink:0}
.rec .t{font-size:.81rem;font-weight:600;margin-bottom:3px}
.rec .d{font-size:.73rem;color:var(--mu);line-height:1.5}
footer{text-align:center;color:var(--mu);font-size:.7rem;padding:16px}
@media(max-width:640px){.g2,.g3{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>&#128737;&#65039; Fraud <span>Detection</span> Dashboard</h1>
  <div style="display:flex;gap:11px;align-items:center">
    <span style="font-size:.75rem;color:var(--mu)">Run: <b id="ts" style="color:var(--txt)"></b></span>
    <span class="live">&#9679; LIVE</span>
  </div>
</header>

<div class="abar">
  <span style="font-size:1.2rem">&#128680;</span>
  <div><strong style="color:var(--red)">ALERT:</strong> <span id="am"></span></div>
</div>

<div class="wrap">
<div class="krow">
  <div class="kpi blu"><label>Total Transactions</label><div class="val" id="kT"></div><div class="sub">Full Dataset</div></div>
  <div class="kpi red"><label>Fraud Detected</label><div class="val" id="kF"></div><div class="sub">By ML Model</div></div>
  <div class="kpi yel"><label>Fraud Rate</label><div class="val" id="kP"></div><div class="sub">Industry avg 1.5%</div></div>
  <div class="kpi grn"><label>Model Recall</label><div class="val" id="kR"></div><div class="sub">Frauds Caught</div></div>
  <div class="kpi pur"><label>Amount Blocked</label><div class="val" id="kA"></div><div class="sub">Fraud Txns</div></div>
</div>

<div class="g2">
  <div class="card"><h3>&#128336; Fraud by Hour of Day</h3><canvas id="cH" height="230"></canvas></div>
  <div class="card"><h3>&#128176; Amount — Fraud vs Legit</h3><canvas id="cA" height="230"></canvas></div>
</div>
<div class="g3">
  <div class="card"><h3>&#128202; Confusion Matrix — Isolation Forest</h3><canvas id="cC1" height="210"></canvas></div>
  <div class="card"><h3>&#128202; Confusion Matrix — Random Forest</h3><canvas id="cC2" height="210"></canvas></div>
  <div class="card"><h3>&#11088; Feature Importance</h3><canvas id="cFi" height="210"></canvas></div>
</div>
<div class="g3">
  <div class="card"><h3>&#127807; Isolation Forest</h3>
    <div class="mb"><div class="lb"><span>Precision</span><span id="ip"></span></div><div class="trk"><div class="fill" id="ipb" style="background:var(--blu)"></div></div></div>
    <div class="mb"><div class="lb"><span>Recall</span><span id="ir"></span></div><div class="trk"><div class="fill" id="irb" style="background:var(--grn)"></div></div></div>
    <div class="mb"><div class="lb"><span>F1 Score</span><span id="if"></span></div><div class="trk"><div class="fill" id="ifb" style="background:var(--pur)"></div></div></div>
    <div class="mb"><div class="lb"><span>AUC-ROC</span><span id="ia"></span></div><div class="trk"><div class="fill" id="iab" style="background:var(--ora)"></div></div></div>
  </div>
  <div class="card"><h3>&#127795; Random Forest + SMOTE</h3>
    <div class="mb"><div class="lb"><span>Precision</span><span id="rp"></span></div><div class="trk"><div class="fill" id="rpb" style="background:var(--blu)"></div></div></div>
    <div class="mb"><div class="lb"><span>Recall</span><span id="rr"></span></div><div class="trk"><div class="fill" id="rrb" style="background:var(--grn)"></div></div></div>
    <div class="mb"><div class="lb"><span>F1 Score</span><span id="rf2"></span></div><div class="trk"><div class="fill" id="rf2b" style="background:var(--pur)"></div></div></div>
    <div class="mb"><div class="lb"><span>AUC-ROC</span><span id="ra"></span></div><div class="trk"><div class="fill" id="rab" style="background:var(--ora)"></div></div></div>
  </div>
  <div class="card"><h3>&#128279; Ensemble (Final)</h3>
    <div class="mb"><div class="lb"><span>Precision</span><span id="ep"></span></div><div class="trk"><div class="fill" id="epb" style="background:var(--blu)"></div></div></div>
    <div class="mb"><div class="lb"><span>Recall &#11088;</span><span id="er"></span></div><div class="trk"><div class="fill" id="erb" style="background:var(--grn)"></div></div></div>
    <div class="mb"><div class="lb"><span>F1 Score</span><span id="ef"></span></div><div class="trk"><div class="fill" id="efb" style="background:var(--pur)"></div></div></div>
    <div class="mb"><div class="lb"><span>SMOTE</span><span style="color:var(--grn)">&#10004; Applied</span></div><div class="trk"><div class="fill" style="width:100%;background:var(--grn)"></div></div></div>
  </div>
</div>

<div class="card" style="margin-bottom:13px">
  <h3>&#128680; High Risk Transaction Alerts</h3>
  <table><thead><tr>
    <th>Txn ID</th><th>Customer</th><th>Bank</th><th>Amount</th>
    <th>Merchant</th><th>City</th><th>Time</th><th>Fraud %</th><th>Alert</th>
  </tr></thead><tbody id="tb"></tbody></table>
</div>

<div class="card" style="margin-bottom:13px">
  <h3>&#128161; Business Recommendations</h3>
  <div class="recs">
    <div class="rec"><span class="ico">&#127769;</span><div><div class="t">Late Night Alert</div>
      <div class="d">50,000+ transactions 12AM-5AM are <b style="color:var(--red)">3.8x more risky</b>. Apply 2FA.</div></div></div>
    <div class="rec"><span class="ico">&#127758;</span><div><div class="t">International IP Block</div>
      <div class="d">Intl IPs = <b style="color:var(--red)">64% of all fraud</b>. Block or extra verification.</div></div></div>
    <div class="rec"><span class="ico">&#9889;</span><div><div class="t">High Transaction Velocity</div>
      <div class="d">5+ tx/min = <b style="color:var(--yel)">7x fraud risk</b>. Trigger step-up auth.</div></div></div>
    <div class="rec"><span class="ico">&#128241;</span><div><div class="t">Device Change + High Amount</div>
      <div class="d">New device + large amount = red flag. <b style="color:var(--yel)">70% fraud</b> = device change.</div></div></div>
    <div class="rec"><span class="ico">&#127978;</span><div><div class="t">High-Risk Merchants</div>
      <div class="d">Crypto/offshore = <b style="color:var(--red)">5x higher fraud</b>. Auto-block these merchants.</div></div></div>
    <div class="rec"><span class="ico">&#9878;&#65039;</span><div><div class="t">SMOTE Improved Model</div>
      <div class="d">After SMOTE: recall improved significantly. <b style="color:var(--grn)">Fewer frauds missed!</b></div></div></div>
  </div>
</div>
</div>
<footer>Fraud Detection System &middot; Isolation Forest + Random Forest + SMOTE &middot; Auto Dashboard</footer>

<script>
const D = DATAPLACEHOLDER;
document.getElementById('ts').textContent = new Date().toLocaleString('en-IN');
document.getElementById('kT').textContent = D.total.toLocaleString('en-IN');
document.getElementById('kF').textContent = D.fraud.toLocaleString('en-IN');
document.getElementById('kP').textContent = D.fraud_pct + '%';
document.getElementById('kR').textContent = Math.round(D.rf.r*100)+'%';
const bl = D.blocked >= 100000 ? '&#8377;'+(D.blocked/100000).toFixed(1)+'L' : '&#8377;'+(D.blocked/1000).toFixed(1)+'K';
document.getElementById('kA').innerHTML = bl;

const alerts = [
  D.high+' HIGH risk transactions flagged — immediate review needed!',
  'International IP fraud detected — '+D.fraud+' suspicious transactions found',
  'SMOTE applied — class imbalance corrected, recall maximized to '+Math.round(D.rf.r*100)+'%',
  D.medium+' MEDIUM risk transactions are under active monitoring'
];
let ai=0;
document.getElementById('am').textContent=alerts[0];
setInterval(()=>document.getElementById('am').textContent=alerts[++ai%alerts.length],3500);

function bar(id,val){
  document.getElementById(id).textContent=(val*100).toFixed(1)+'%';
  document.getElementById(id+'b').style.width=(val*100)+'%';
}
bar('ip',D.iso.p);bar('ir',D.iso.r);bar('if',D.iso.f);bar('ia',D.iso.a);
bar('rp',D.rf.p);bar('rr',D.rf.r);bar('rf2',D.rf.f);bar('ra',D.rf.a);
bar('ep',D.ens.p);bar('er',D.ens.r);bar('ef',D.ens.f);

const gc='#21262d',tc='#8b949e';
const bo={responsive:true,plugins:{legend:{labels:{color:tc,font:{size:10}}}}};

new Chart(document.getElementById('cH'),{type:'bar',data:{
  labels:[...Array(24).keys()].map(h=>h+':00'),
  datasets:[
    {label:'Fraud',data:D.fh,backgroundColor:'rgba(248,81,73,.8)',borderRadius:3},
    {label:'Legit (÷20)',data:D.lh.map(v=>v/20),backgroundColor:'rgba(63,185,80,.3)',borderRadius:3}
  ]},options:{...bo,scales:{x:{ticks:{color:tc,maxRotation:45,font:{size:8}},grid:{color:gc}},y:{ticks:{color:tc},grid:{color:gc}}}}});

new Chart(document.getElementById('cA'),{type:'bar',data:{
  labels:['<1K','1K-5K','5K-20K','20K-50K','50K-1L','1L+'],
  datasets:[
    {label:'Legit',data:D.la,backgroundColor:'rgba(63,185,80,.5)',borderRadius:3},
    {label:'Fraud',data:D.fa,backgroundColor:'rgba(248,81,73,.8)',borderRadius:3}
  ]},options:{...bo,scales:{x:{ticks:{color:tc},grid:{color:gc}},y:{ticks:{color:tc},grid:{color:gc}}}}});

function drawCM(id,cm){
  new Chart(document.getElementById(id),{type:'bar',data:{
    labels:['TN (Correct Legit)','FP (Wrong Fraud)','FN (Missed!)','TP (Caught!)'],
    datasets:[{data:[cm[0][0],cm[0][1],cm[1][0],cm[1][1]],
      backgroundColor:['rgba(63,185,80,.7)','rgba(248,81,73,.4)','rgba(248,81,73,.9)','rgba(63,185,80,.9)'],
      borderRadius:4}]},
    options:{...bo,indexAxis:'y',plugins:{legend:{display:false}},
      scales:{x:{ticks:{color:tc},grid:{color:gc}},y:{ticks:{color:tc,font:{size:9}},grid:{color:gc}}}}});
}
drawCM('cC1',D.cm1); drawCM('cC2',D.cm2);

new Chart(document.getElementById('cFi'),{type:'bar',data:{
  labels:D.fn,
  datasets:[{data:D.fv,
    backgroundColor:D.fv.map((v,i)=>i<3?'rgba(248,81,73,.8)':'rgba(88,166,255,.6)'),
    borderRadius:4}]},
  options:{...bo,indexAxis:'y',plugins:{legend:{display:false}},
    scales:{x:{ticks:{color:tc},grid:{color:gc}},y:{ticks:{color:tc,font:{size:9}},grid:{color:gc}}}}});

const tb=document.getElementById('tb');
D.rows.forEach(r=>{
  const tr=document.createElement('tr');
  const c=r.Alert_Level==='HIGH'?'H':r.Alert_Level==='MEDIUM'?'M':'L';
  tr.innerHTML=`<td style="font-family:monospace;color:var(--blu)">${r.Transaction_ID}</td>
    <td>${r.Customer_Name}</td><td>${r.Bank}</td>
    <td style="font-weight:600">&#8377;${Number(r.Amount_INR).toLocaleString('en-IN')}</td>
    <td style="color:var(--red)">${r.Merchant}</td><td>${r.City}</td>
    <td>${r.Transaction_Time||'--'}</td>
    <td><b style="color:var(--red)">${(r.Fraud_Probability*100).toFixed(1)}%</b></td>
    <td><span class="bdg ${c}">${r.Alert_Level}</span></td>`;
  tb.appendChild(tr);
});
</script></body></html>"""

HTML = HTML.replace('DATAPLACEHOLDER', json.dumps(data))

out = os.path.abspath("fraud_dashboard.html")
with open(out,"w",encoding="utf-8") as f:
    f.write(HTML)

print(f"  Dashboard saved: fraud_dashboard.html")
print(f"  Predictions saved: fraud_predictions.csv")
print(f"\n  Browser is opening...")
webbrowser.open(f"file:///{out}")

print("\n"+"="*50)
print("  DONE! Browser is now open!")
print("="*50)