# SIH26153 LSTM baseline script
from pathlib import Path
import json, random
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, confusion_matrix

SEED=42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED); tf.get_logger().setLevel('ERROR')
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'/'processed'/'lstm_sequences'
OUT=ROOT/'results'/'lstm_baseline'; OUT.mkdir(parents=True,exist_ok=True)
EPOCHS=100; BATCH_SIZE=128; LR=1e-3; LSTM_UNITS=64; DENSE_UNITS=32; DROPOUT=.20
PATIENCE=10; LR_PATIENCE=5
HORIZONS=['+30s','+60s','+90s','+120s','+150s']; N_H=len(HORIZONS)
THRESHOLDS=np.arange(.05,.951,.01)

def load(name):
 p=DATA/name
 if not p.exists(): raise FileNotFoundError(f'Missing file: {p}')
 a=np.load(p); print(f'Loaded {name}: shape={a.shape}')
 if not np.isfinite(a).all(): raise ValueError(f'{name} contains NaN/infinite values')
 return a

def validate(X,y,name):
 if X.ndim!=3 or y.ndim!=2 or len(X)!=len(y) or X.shape[1:]!=(10,24) or y.shape[1]!=N_H: raise ValueError(f'{name} shape error: X={X.shape}, y={y.shape}')
 if not np.all(np.isin(np.unique(y),[0,1])): raise ValueError(f'{name} labels are not binary')
 print(f'{name}: X={X.shape}, y={y.shape}')

def metrics(y,p,t):
 pred=(p>=t).astype(int)
 tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 try: auc=roc_auc_score(y,p)
 except ValueError: auc=np.nan
 return {'threshold':float(t),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'pr_auc':float(average_precision_score(y,p)),'roc_auc':float(auc) if not np.isnan(auc) else np.nan,'fpr':float(fp/(fp+tn)) if tn+fp else 0.,'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

print('='*70); print('SIH26153 — LSTM BASELINE'); print('='*70)
X_train,y_train=load('X_train.npy'),load('y_train.npy')
X_val,y_val=load('X_validation.npy'),load('y_validation.npy')
X_test,y_test=load('X_test.npy'),load('y_test.npy')
validate(X_train,y_train,'TRAIN'); validate(X_val,y_val,'VALIDATION'); validate(X_test,y_test,'TEST')

for name,y in [('TRAIN',y_train),('VALIDATION',y_val),('TEST',y_test)]:
 print('\n'+name)
 for i,h in enumerate(HORIZONS): print(f'  {h}: {int(y[:,i].sum())}/{len(y)} ({y[:,i].mean():.4%})')

N,T,F=X_train.shape
scaler=StandardScaler().fit(X_train.reshape(-1,F))
X_train=scaler.transform(X_train.reshape(-1,F)).reshape(X_train.shape)
X_val=scaler.transform(X_val.reshape(-1,F)).reshape(X_val.shape)
X_test=scaler.transform(X_test.reshape(-1,F)).reshape(X_test.shape)
joblib.dump(scaler,OUT/'scaler.joblib')

pos=y_train.sum(0); neg=len(y_train)-pos; weights=neg/np.maximum(pos,1); W=tf.constant(weights.astype('float32'))
print('\nTrain positive weights:',dict(zip(HORIZONS,weights.round(4))))

def weighted_bce(y,p):
 e=tf.keras.backend.epsilon(); p=tf.clip_by_value(p,e,1-e)
 return tf.reduce_mean(-(W*y*tf.math.log(p)+(1-y)*tf.math.log(1-p)),axis=-1)

model=tf.keras.Sequential([
 tf.keras.layers.Input((T,F)),
 tf.keras.layers.LSTM(LSTM_UNITS),
 tf.keras.layers.Dropout(DROPOUT),
 tf.keras.layers.Dense(DENSE_UNITS,activation='relu'),
 tf.keras.layers.Dense(N_H,activation='sigmoid')],name='SIH26153_LSTM_Forecaster')
model.compile(optimizer=tf.keras.optimizers.Adam(LR,clipnorm=1.),loss=weighted_bce)
model.summary()
callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=PATIENCE,restore_best_weights=True,verbose=1),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=LR_PATIENCE,min_lr=1e-6,verbose=1),tf.keras.callbacks.ModelCheckpoint(OUT/'model.keras',monitor='val_loss',save_best_only=True,verbose=1)]
history=model.fit(X_train,y_train.astype('float32'),validation_data=(X_val,y_val.astype('float32')),epochs=EPOCHS,batch_size=BATCH_SIZE,shuffle=False,callbacks=callbacks,verbose=1)
pd.DataFrame(history.history).to_csv(OUT/'history.csv',index=False)

vp=model.predict(X_val,batch_size=BATCH_SIZE,verbose=1); tp=model.predict(X_test,batch_size=BATCH_SIZE,verbose=1)
best=[]; vrows=[]
for i,h in enumerate(HORIZONS):
 r=max((metrics(y_val[:,i],vp[:,i],t) for t in THRESHOLDS),key=lambda x:x['f1'])
 best.append(r['threshold']); r.update(horizon=h,horizon_index=i); vrows.append(r)
pd.DataFrame({'horizon':HORIZONS,'threshold':best}).to_csv(OUT/'thresholds.csv',index=False)
pd.DataFrame(vrows).to_csv(OUT/'validation_metrics.csv',index=False)

rows=[]
for i,h in enumerate(HORIZONS):
 r=metrics(y_test[:,i],tp[:,i],best[i]); r.update(horizon=h,horizon_index=i); rows.append(r)
 print(f"\n{h}: threshold={r['threshold']:.2f} precision={r['precision']:.4f} recall={r['recall']:.4f} F1={r['f1']:.4f} PR-AUC={r['pr_auc']:.4f} ROC-AUC={r['roc_auc']:.4f} FPR={r['fpr']:.4f} TN={r['tn']} FP={r['fp']} FN={r['fn']} TP={r['tp']}")
pd.DataFrame(rows).to_csv(OUT/'test_metrics.csv',index=False)

pred=pd.DataFrame()
for i,h in enumerate(HORIZONS): pred[f'y_true_{h}']=y_test[:,i]; pred[f'prob_{h}']=tp[:,i]; pred[f'pred_{h}']=(tp[:,i]>=best[i]).astype(int)
pred.to_csv(OUT/'test_predictions.csv',index=False)

config={'seed':SEED,'epochs':EPOCHS,'batch_size':BATCH_SIZE,'learning_rate':LR,'lstm_units':LSTM_UNITS,'dense_units':DENSE_UNITS,'dropout':DROPOUT,'sequence_length':T,'features':F,'horizons':HORIZONS,'pos_weights':weights.tolist(),'thresholds':best,'shuffle':False}
(OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')

with open(OUT/'report.txt','w',encoding='utf-8') as f:
 f.write('SIH26153 — LSTM BASELINE REPORT\n'+'='*60+'\n\n')
 f.write(f'Train: {len(X_train)}\nValidation: {len(X_val)}\nTest: {len(X_test)}\nSequence length: {T}\nFeatures: {F}\n\n')
 f.write('Scaler fitted on train only. Thresholds tuned on validation only. Test used only for final evaluation.\n\n')
 f.write('VALIDATION\n'+pd.DataFrame(vrows).to_string(index=False)+'\n\nTEST\n'+pd.DataFrame(rows).to_string(index=False)+'\n')
print('\n'+'='*70); print('LSTM TRAINING COMPLETE'); print('Results:',OUT); print('Next: compare LSTM directly against persistence.'); print('='*70)