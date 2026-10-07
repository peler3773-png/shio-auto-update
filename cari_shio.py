import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
SEED_TETAP = 20261004
import random
random.seed(SEED_TETAP)
import numpy as np
np.random.seed(SEED_TETAP)
import tensorflow as tf
tf.random.set_seed(SEED_TETAP)
tf.get_logger().setLevel('ERROR')
try:
    import optuna
except ImportError:
    print("❌ Pasang: pip install optuna")
    sys.exit(1)
import urllib.request
import json
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

DATA_UNDIAN_URL = "https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/data_undian.txt"
LOOKBACK = 12
LIMIT_PER_PASARAN = 2000
FAKTOR_OVERDUE = 0.40
OPTUNA_EPOCH_MIN = 49
OPTUNA_EPOCH_MAX = 80
OPTUNA_BATCH_CHOICES = [32, 64, 128]
OPTUNA_CUPIKAN = 10
VALIDASI_MIN = 7
PATIENCE_ES = 12
PATIENCE_OPTUNA_SEARCH = 8
PATIENCE_OPTUNA_FINAL = 10

SHIO_MAP = {
    '0': 1, '1': 2, '2': 3, '3': 4, '4': 5,
    '5': 6, '6': 7, '7': 8, '8': 9, '9': 12
}

def hitung_bobot_overdue(data_pasaran, posisi_idx):
    terakhir_muncul = {str(d): None for d in range(10)}
    for urutan, baris in enumerate(reversed(data_pasaran)):
        angka = str(baris['angka'][posisi_idx])
        if terakhir_muncul[angka] is None:
            terakhir_muncul[angka] = urutan
    jarak_tercatat = [v for v in terakhir_muncul.values() if v is not None]
    if not jarak_tercatat:
        max_jarak = 1
        for d in terakhir_muncul:
            terakhir_muncul[d] = 0
    else:
        max_jarak = max(jarak_tercatat) + 1
        rata_jarak = sum(jarak_tercatat) / len(jarak_tercatat)
        for d in terakhir_muncul:
            if terakhir_muncul[d] is None:
                terakhir_muncul[d] = rata_jarak
    bobot = {}
    for d in range(10):
        j = terakhir_muncul[str(d)]
        bobot[str(d)] = 1.0 + (j / max_jarak) * FAKTOR_OVERDUE
    return bobot

def format_hasil(prob):
    urut = np.argsort(prob)[::-1].tolist()
    sembilan = urut[:9]
    tujuh = urut[:7] + [urut[9]]
    return {
        "tujuh": [str(a) for a in tujuh],
        "sembilan": [str(a) for a in sembilan]
    }

def hitung_2d_shio(pred_as, pred_kop, pred_kep, pred_eko, pakai_overdue=True):
    a = 'overdue' if pakai_overdue else 'murni'
    daftar = {}
    dd_pasangan, dd_shio = [], set()
    for x in pred_as[a]['tujuh']:
        for y in pred_kop[a]['tujuh']:
            dd_pasangan.append(x+y)
            dd_shio.add(SHIO_MAP[x]); dd_shio.add(SHIO_MAP[y])
    daftar['2DD'] = {'pasangan': dd_pasangan, 'shio': sorted(dd_shio)}
    dt_pasangan, dt_shio = [], set()
    for x in pred_kop[a]['tujuh']:
        for y in pred_kep[a]['tujuh']:
            dt_pasangan.append(x+y)
            dt_shio.add(SHIO_MAP[x]); dt_shio.add(SHIO_MAP[y])
    daftar['2DT'] = {'pasangan': dt_pasangan, 'shio': sorted(dt_shio)}
    db_pasangan, db_shio = [], set()
    for x in pred_kep[a]['tujuh']:
        for y in pred_eko[a]['tujuh']:
            db_pasangan.append(x+y)
            db_shio.add(SHIO_MAP[x]); db_shio.add(SHIO_MAP[y])
    daftar['2DB'] = {'pasangan': db_pasangan, 'shio': sorted(db_shio)}
    return daftar

def bangun_model(ukuran_urutan=LOOKBACK):
    inp = Input(shape=(ukuran_urutan, 4))
    x = LSTM(64, activation='relu')(inp)
    x = Dense(32, activation='relu')(x)
    mdl = Model(inputs=inp, outputs=[
        Dense(10, activation='softmax', name='as')(x),
        Dense(10, activation='softmax', name='kop')(x),
        Dense(10, activation='softmax', name='kep')(x),
        Dense(10, activation='softmax', name='eko')(x)
    ])
    mdl.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return mdl

def siapkan_data(dp):
    total = len(dp)
    sampel = total - LOOKBACK
    if sampel < 20 + VALIDASI_MIN:
        return None, None, None, None, 0
    X = np.zeros((sampel, LOOKBACK, 4), dtype=np.float32)
    Y = np.zeros((sampel, 4), dtype=np.int32)
    for i in range(sampel):
        X[i] = [dp[j]['angka'] for j in range(i, i+LOOKBACK)]
        Y[i] = dp[i+LOOKBACK]['angka']
    batas = min(max(5, int(0.85*sampel)), sampel-VALIDASI_MIN)
    return X[:batas], {'as':Y[:batas,0],'kop':Y[:batas,1],'kep':Y[:batas,2],'eko':Y[:batas,3]}, \
           X[batas:], {'as':Y[batas:,0],'kop':Y[batas:,1],'kep':Y[batas:,2],'eko':Y[batas:,3]}, batas

def latih_earlystop(X, Y, Xv, Yv):
    if X is None: return None, None
    m = bangun_model()
    es = EarlyStopping(monitor='val_loss', patience=PATIENCE_ES, restore_best_weights=True, verbose=0)
    h = m.fit(X,Y,epochs=100,batch_size=32,validation_data=(Xv,Yv),callbacks=[es],verbose=0)
    return m, {"epoch":len(h.history['loss']),"batch_size":32,"berhenti_di":len(h.history['loss'])}

def cari_optuna(X, Y, Xv, Yv):
    if X is None: return None, None
    def tujuan(t):
        m = bangun_model()
        e = t.suggest_int('epoch',OPTUNA_EPOCH_MIN,OPTUNA_EPOCH_MAX)
        b = t.suggest_categorical('batch_size',OPTUNA_BATCH_CHOICES)
        es = EarlyStopping(monitor='val_loss',patience=PATIENCE_OPTUNA_SEARCH,restore_best_weights=True,verbose=0)
        return min(m.fit(X,Y,epochs=e,batch_size=b,validation_data=(Xv,Yv),callbacks=[es],verbose=0).history['val_loss'])
    st = optuna.create_study(direction='minimize')
    st.optimize(tujuan, n_trials=OPTUNA_CUPIKAN, show_progress_bar=False)
    bp = st.best_params
    m = bangun_model()
    es = EarlyStopping(monitor='val_loss',patience=PATIENCE_OPTUNA_FINAL,restore_best_weights=True,verbose=0)
    m.fit(X,Y,epochs=bp['epoch'],batch_size=bp['batch_size'],validation_data=(Xv,Yv),callbacks=[es],verbose=0)
    return m, {"epoch":bp['epoch'],"batch_size":bp['batch_size'],"skor_terbaik":round(st.best_value,8)}

def proses_semua():
    print("📥 Membaca data...")
    req = urllib.request.Request(DATA_UNDIAN_URL, headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=120) as r:
        isi = r.read().decode('utf-8')
    mentah = []
    for b in isi.strip().splitlines():
        p = b.split('|')
        if len(p)<4: continue
        an = p[2].strip()
        if len(an)==4 and an.isdigit():
            mentah.append({"pasaran":p[0].strip().upper(),"tanggal":p[1].strip(),
                           "angka":[int(d) for d in an],"nomor":an,"waktu":p[3].strip()})
    dilihat, bersih = set(), []
    for e in mentah:
        k = (e['pasaran'],e['tanggal'],e['nomor'])
        if k not in dilihat: dilihat.add(k); bersih.append(e)
    mentah = bersih
    mentah.sort(key=lambda x:(x['tanggal'],x['waktu']))
    data_pasaran = {}
    for b in mentah:
        data_pasaran.setdefault(b['pasaran'],[]).append(b)
    for p in data_pasaran:
        if len(data_pasaran[p])>LIMIT_PER_PASARAN:
            data_pasaran[p] = data_pasaran[p][-LIMIT_PER_PASARAN:]
    list_pasaran = sorted(data_pasaran.keys())
    hasil_akhir = {
        "diperbarui":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"zona_waktu":"WIB / UTC+7",
        "pengaturan":{"LOOKBACK":LOOKBACK,"LIMIT_PER_PASARAN":LIMIT_PER_PASARAN,"peta_shio":SHIO_MAP},
        "daftar_pasaran":list_pasaran,"hasil":{}
    }
    nama_posisi, nama_output = ["AS","KOP","KEPALA","EKOR"],["as","kop","kep","eko"]
    for p in list_pasaran:
        dp = data_pasaran[p]
        if len(dp) < LOOKBACK+20+VALIDASI_MIN: continue
        X,Y,Xv,Yv,_ = siapkan_data(dp)
        if X is None: continue
        inp = np.expand_dims(np.array([dp[j]['angka'] for j in range(-LOOKBACK,0)],dtype=np.float32),0)
        def jalankan(m,info):
            pr = m.predict(inp, verbose=0)
            pos = {}
            res = {"pengaturan":info}
            for idx,(nm,onm) in enumerate(zip(nama_posisi,nama_output)):
                pb = pr[idx][0].copy(); pb /= pb.sum()
                bobot = hitung_bobot_overdue(dp,idx)
                pbov = pb.copy()
                for d in range(10): pbov[d] *= bobot[str(d)]
                pbov /= pbov.sum()
                res[nm] = {"murni":format_hasil(pb),"overdue":format_hasil(pbov)}
                pos[nm] = res[nm]
            res["2D"] = hitung_2d_shio(pos["AS"],pos["KOP"],pos["KEPALA"],pos["EKOR"])
            return res
        me, ie = latih_earlystop(X,Y,Xv,Yv)
        mo, io = cari_optuna(X,Y,Xv,Yv)
        hasil_akhir["hasil"][p] = {
            "early_stopping": jalankan(me,ie),
            "optuna": jalankan(mo,io)
        }
    with open("hasil_prediksi.json","w",encoding="utf-8") as f:
        json.dump(hasil_akhir,f,ensure_ascii=False,indent=2)
    print("✅ Selesai → hasil_prediksi.json")

if __name__ == "__main__":
    proses_semua()
