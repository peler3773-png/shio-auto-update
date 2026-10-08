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
    print("❌ ERROR: Pasang dulu → pip install optuna")
    sys.exit(1)
import urllib.request
import json
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# === PENGATURAN ===
DAFTAR_PASARAN_TETAP = ["BE", "CLF", "SD", "HK", "SGP", "PS", "MCSE", "NCE", "TM", "TXM"]
LIBUR_MINGGU = ["PS", "TM", "TXM"]
LIBUR_SGP_HARI = ["Selasa", "Jumat"]
DATA_UNDIAN_URL = "https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/data_undian.txt"
LOOKBACK = 12
LIMIT_PER_PASARAN = 600
FAKTOR_OVERDUE = 0.40
OPTUNA_EPOCH_MIN, OPTUNA_EPOCH_MAX = 30, 55
OPTUNA_BATCH = [32, 64]
OPTUNA_CUPIKAN = 5
VALIDASI_MIN = 7
PATIENCE_ES = 8
PATIENCE_OPTUNA_SEARCH = 5
PATIENCE_OPTUNA_FINAL = 7

# === SHIO DINAMIS ===
DAFTAR_SHIO_URUTAN = [
    {"urutan": 1, "nama": "KUDA"},
    {"urutan": 2, "nama": "ULAR"},
    {"urutan": 3, "nama": "NAGA"},
    {"urutan": 4, "nama": "KELINCI"},
    {"urutan": 5, "nama": "HARIMAU"},
    {"urutan": 6, "nama": "KERBAU"},
    {"urutan": 7, "nama": "TIKUS"},
    {"urutan": 8, "nama": "BABI"},
    {"urutan": 9, "nama": "ANJING"},
    {"urutan": 10, "nama": "AYAM"},
    {"urutan": 11, "nama": "MONYET"},
    {"urutan": 12, "nama": "KAMBING"}
]
POLA_DASAR = {
    "KUDA":    ["01","13","25","37","49","61","73","85","97"],
    "ULAR":    ["02","14","26","38","50","62","74","86","98"],
    "NAGA":    ["03","15","27","39","51","63","75","87","99"],
    "KELINCI": ["04","16","28","40","52","64","76","88","00"],
    "HARIMAU": ["05","17","29","41","53","65","77","89"],
    "KERBAU":  ["06","18","30","42","54","66","78","90"],
    "TIKUS":   ["07","19","31","43","55","67","79","91"],
    "BABI":    ["08","20","32","44","56","68","80","92"],
    "ANJING":  ["09","21","33","45","57","69","81","93"],
    "AYAM":    ["10","22","34","46","58","70","82","94"],
    "MONYET":  ["11","23","35","47","59","71","83","95"],
    "KAMBING": ["12","24","36","48","60","72","84","96"]
}

def hitung_geseran(tahun=None):
    if tahun is None: tahun = datetime.now().year
    return (tahun - 2026) % 12

def bangun_peta_shio(tahun=None):
    if tahun is None: tahun = datetime.now().year
    geser = hitung_geseran(tahun)
    urut_digeser = DAFTAR_SHIO_URUTAN[geser:] + DAFTAR_SHIO_URUTAN[:geser]
    nama_dasar = list(POLA_DASAR.keys())
    peta_shio, peta_angka, nomor_nama = {}, {}, {}
    for idx, s in enumerate(urut_digeser):
        i_dasar = (idx - geser) % len(nama_dasar)
        angka = POLA_DASAR[nama_dasar[i_dasar]]
        peta_shio[s["nama"]] = {"urutan": s["urutan"], "angka": angka, "tahun": tahun}
        nomor_nama[s["urutan"]] = s["nama"]
        for a in angka:
            peta_angka[a] = {"nomor": s["urutan"], "nama": s["nama"]}
    return peta_shio, peta_angka, nomor_nama, geser

TAHUN_SEKARANG = datetime.now().year
DATA_SHIO, ANGKA_KE_SHIO, NOMOR_KE_NAMA, GESERAN = bangun_peta_shio(TAHUN_SEKARANG)

def nama_hari(tgl_str):
    hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    try:
        return hari[datetime.strptime(tgl_str, "%Y-%m-%d").weekday()]
    except: return ""

def boleh_tampil(pasaran, tgl=None):
    hari_ini = nama_hari(tgl or datetime.now().strftime("%Y-%m-%d"))
    if pasaran in LIBUR_MINGGU and hari_ini == "Minggu": return False
    if pasaran == "SGP" and hari_ini in LIBUR_SGP_HARI: return False
    return True

# === ✅ HITUNG SHIO UNIK PER BAGIAN — SEMUA 12 URUT TERKUAT ===
def hitung_shio_dari_daftar(daftar_angka):
    bobot = {}
    for urut, ang in enumerate(daftar_angka):
        berat = (10 - urut) / 10  # urutan depan = bobot lebih besar
        if ang in ANGKA_KE_SHIO:
            sn = ANGKA_KE_SHIO[ang]["nomor"]
            bobot[sn] = bobot.get(sn, 0) + berat
        if len(ang) == 1:
            akh = ang
        else:
            akh = ang[-1]
        for info in DATA_SHIO.values():
            if any(a.endswith(akh) for a in info["angka"]):
                sn = info["urutan"]
                bobot[sn] = bobot.get(sn, 0) + berat * 0.5
    # Pastikan SEMUA 12 Shio ada, isi 0 jika tidak muncul
    for sn in range(1, 13):
        if sn not in bobot:
            bobot[sn] = 0
    # Urut terkuat ke terlemah
    urut = sorted(bobot.items(), key=lambda x: (-round(x[1],4), x[0]))
    return [f"{k} {NOMOR_KE_NAMA[k]}" for k, _ in urut]

def hitung_2d_shio(as_list, kop_list, kep_list, eko_list):
    mode = "overdue"
    hasil = {}
    for kd, xd, yd in [
        ("2DD", as_list[mode], kop_list[mode]),
        ("2DT", kop_list[mode], kep_list[mode]),
        ("2DB", kep_list[mode], eko_list[mode]),
    ]:
        psg = []
        gabungan = []
        for x in xd[:10]:
            for y in yd[:10]:
                gabungan.append(f"{x}{y}")
                psg.append(f"{x}{y}")
        # Hitung Shio unik dari gabungan angka pasangan
        shio_urut = hitung_shio_dari_daftar(gabungan)
        hasil[kd] = {
            "pasangan": psg[:10],
            "shio_urut": shio_urut,
            "shio_teks": ", ".join(shio_urut)
        }
    return hasil

def bobot_overdue(data, pos):
    terakhir = {str(d): None for d in range(10)}
    for urut, b in enumerate(reversed(data)):
        a = str(b["angka"][pos])
        if terakhir[a] is None: terakhir[a] = urut
    jarak = [v for v in terakhir.values() if v is not None]
    if not jarak:
        for d in terakhir: terakhir[d] = 0
    else:
        max_j = max(jarak) + 1
        rata = sum(jarak)/len(jarak)
        for d in terakhir:
            if terakhir[d] is None: terakhir[d] = rata
    return {d: 1 + (j/max_j)*FAKTOR_OVERDUE for d,j in terakhir.items()}

def format_10(prob):
    urut = np.argsort(prob)[::-1].tolist()
    return {
        "p10": [str(a) for a in urut[:10]],
        "p7":  [str(a) for a in urut[:7]],
        "p9":  [str(a) for a in urut[:9]]
    }

def bangun_model():
    inp = Input(shape=(LOOKBACK, 4))
    x = LSTM(64, activation='relu')(inp)
    x = Dense(32, activation='relu')(x)
    out_as = Dense(10, activation='softmax', name='as')(x)
    out_kop = Dense(10, activation='softmax', name='kop')(x)
    out_kep = Dense(10, activation='softmax', name='kep')(x)
    out_eko = Dense(10, activation='softmax', name='eko')(x)
    
    model = Model(inputs=inp, outputs=[out_as, out_kop, out_kep, out_eko])
    model.compile(
        optimizer='adam',
        loss={
            'as': 'sparse_categorical_crossentropy',
            'kop': 'sparse_categorical_crossentropy',
            'kep': 'sparse_categorical_crossentropy',
            'eko': 'sparse_categorical_crossentropy'
        },
        metrics={
            'as': 'accuracy',
            'kop': 'accuracy',
            'kep': 'accuracy',
            'eko': 'accuracy'
        }
    )
    return model

def siapkan_data(dp):
    n = len(dp) - LOOKBACK
    if n < 20 + VALIDASI_MIN: return None, None, None, None, 0
    X = np.zeros((n, LOOKBACK, 4), dtype=np.float32)
    Y_as = np.zeros((n,), dtype=np.int32)
    Y_kop = np.zeros((n,), dtype=np.int32)
    Y_kep = np.zeros((n,), dtype=np.int32)
    Y_eko = np.zeros((n,), dtype=np.int32)
    for i in range(n):
        X[i] = [dp[j]["angka"] for j in range(i, i+LOOKBACK)]
        Y_as[i] = dp[i+LOOKBACK]["angka"][0]
        Y_kop[i] = dp[i+LOOKBACK]["angka"][1]
        Y_kep[i] = dp[i+LOOKBACK]["angka"][2]
        Y_eko[i] = dp[i+LOOKBACK]["angka"][3]
    batas = min(max(5, int(0.85*n)), n-VALIDASI_MIN)
    X_tr, X_val = X[:batas], X[batas:]
    Y_tr = {'as':Y_as[:batas], 'kop':Y_kop[:batas], 'kep':Y_kep[:batas], 'eko':Y_eko[:batas]}
    Y_val = {'as':Y_as[batas:], 'kop':Y_kop[batas:], 'kep':Y_kep[batas:], 'eko':Y_eko[batas:]}
    return X_tr, Y_tr, X_val, Y_val, batas

def latih_es(X,Y,Xv,Yv):
    if X is None: return None, None
    m = bangun_model()
    es = EarlyStopping(monitor='val_loss', patience=PATIENCE_ES, restore_best_weights=True, verbose=0)
    h = m.fit(X, Y, epochs=100, batch_size=32, validation_data=(Xv, Yv), callbacks=[es], verbose=0)
    return m, {"epoch": len(h.history['loss']), "batch_size": 32, "jenis": "Early Stopping"}

def cari_optuna(X,Y,Xv,Yv):
    if X is None: return None, None
    def tujuan(t):
        m = bangun_model()
        e = t.suggest_int("epoch", OPTUNA_EPOCH_MIN, OPTUNA_EPOCH_MAX)
        b = t.suggest_categorical("batch_size", OPTUNA_BATCH)
        es = EarlyStopping(monitor='val_loss', patience=PATIENCE_OPTUNA_SEARCH, restore_best_weights=True, verbose=0)
        riwayat = m.fit(X, Y, epochs=e, batch_size=b, validation_data=(Xv, Yv), callbacks=[es], verbose=0)
        return min(riwayat.history['val_loss'])
    st = optuna.create_study(direction="minimize")
    st.optimize(tujuan, n_trials=OPTUNA_CUPIKAN, show_progress_bar=False)
    bp = st.best_params
    m = bangun_model()
    es = EarlyStopping(monitor='val_loss', patience=PATIENCE_OPTUNA_FINAL, restore_best_weights=True, verbose=0)
    m.fit(X, Y, epochs=bp["epoch"], batch_size=bp["batch_size"], validation_data=(Xv, Yv), callbacks=[es], verbose=0)
    return m, {"epoch": bp["epoch"], "batch_size": bp["batch_size"], "jenis": "Optuna"}

def jalankan_prediksi(model, info, dp):
    inp = np.expand_dims(np.array([dp[j]["angka"] for j in range(-LOOKBACK, 0)], dtype=np.float32), 0)
    pred = model.predict(inp, verbose=0)
    
    hasil_pos = {}
    nama_posisi = ["AS", "KOP", "KEPALA", "EKOR"]
    nama_output = ["as", "kop", "kep", "eko"]
    
    for idx, (nm, onm) in enumerate(zip(nama_posisi, nama_output)):
        p = pred[idx][0].copy()
        p /= p.sum()
        bobot = bobot_overdue(dp, idx)
        po = p.copy()
        for d in range(10): po[d] *= bobot[str(d)]
        po /= po.sum()
        hasil_pos[nm] = {
            "murni": format_10(p),
            "overdue": format_10(po)
        }
    
    # === 2D + SHIO — Shio DIHITUNG TERPISAH per bagian, SEMUA 12 urut terkuat ===
    hasil_pos["2D"] = hitung_2d_shio(
        hasil_pos["AS"], hasil_pos["KOP"], hasil_pos["KEPALA"], hasil_pos["EKOR"]
    )
    hasil_pos["pengaturan"] = info
    return hasil_pos

def utama():
    print("📥 Mengambil data...")
    req = urllib.request.Request(DATA_UNDIAN_URL, headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        isi = r.read().decode("utf-8")
    mentah = []
    for b in isi.strip().splitlines():
        p = b.split("|")
        if len(p)>=4 and len(p[2])==4 and p[2].isdigit():
            mentah.append({"pasaran":p[0].strip().upper(),"tanggal":p[1].strip(),
                           "angka":[int(d) for d in p[2]],"nomor":p[2],"waktu":p[3].strip()})
    dilihat, bersih = set(), []
    for e in mentah:
        k = (e["pasaran"], e["tanggal"], e["nomor"])
        if k not in dilihat: dilihat.add(k); bersih.append(e)
    mentah = bersih
    mentah.sort(key=lambda x:(x["tanggal"], x["waktu"]))
    data_pasaran = {}
    for b in mentah: data_pasaran.setdefault(b["pasaran"], []).append(b)
    for p in data_pasaran:
        if len(data_pasaran[p])>LIMIT_PER_PASARAN:
            data_pasaran[p] = data_pasaran[p][-LIMIT_PER_PASARAN:]

    hari_ini = datetime.now().strftime("%Y-%m-%d")
    daftar_aktif = []
    for p in DAFTAR_PASARAN_TETAP:
        if p not in data_pasaran:
            print(f"ℹ️ {p:8} — tidak ada data"); continue
        if not boleh_tampil(p, hari_ini):
            print(f"🚫 {p:8} — libur ({nama_hari(hari_ini)})"); continue
        daftar_aktif.append(p)

    hasil_akhir = {
        "diperbarui": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "zona_waktu": "WIB / UTC+7",
        "shio_tahun": TAHUN_SEKARANG,
        "shio_geser": GESERAN,
        "pengaturan": {
            "LOOKBACK": LOOKBACK, "faktor_overdue": FAKTOR_OVERDUE,
            "daftar_shio": DATA_SHIO, "hari_ini": nama_hari(hari_ini),
            "daftar_pasaran_aktif": daftar_aktif
        },
        "hasil": {}
    }

    for p in daftar_aktif:
        dp = data_pasaran[p]
        if len(dp) < LOOKBACK + 20 + VALIDASI_MIN:
            print(f"\n⚠️ {p:8} — kurang data ({len(dp)} baris)"); continue
        print(f"\n{'─'*70}\n📊 {p} | {len(dp)} baris\n{'─'*70}")
        X,Y,Xv,Yv,_ = siapkan_data(dp)
        if X is None: continue

        me, ie = latih_es(X,Y,Xv,Yv)
        mo, io = cari_optuna(X,Y,Xv,Yv)
        if not me or not mo: continue

        hasil_e = jalankan_prediksi(me, ie, dp)
        hasil_o = jalankan_prediksi(mo, io, dp)

        hasil_akhir["hasil"][p] = {
            "early_stopping": hasil_e,
            "optuna": hasil_o
        }

        for nama, d in [("Early Stopping", hasil_e), ("Optuna", hasil_o)]:
            print(f"\n🏆 {nama} — Epoch: {d['pengaturan']['epoch']} | Batch: {d['pengaturan']['batch_size']}")
            for pos in ["AS","KOP","KEPALA","EKOR"]:
                p7 = " ".join(d[pos]["overdue"]["p7"])
                p9 = " ".join(d[pos]["overdue"]["p9"])
                print(f"{pos}: 7+1D: {p7}  9D: {p9}")
            print("\nGabungan 2D + Shio")
            for kd in ["2DD","2DT","2DB"]:
                ps = " ".join(d["2D"][kd]["pasangan"])
                sh = d["2D"][kd]["shio_teks"]
                print(f"\n{kd}: {ps}")
                print(f"Shio: {sh}")

    with open("hasil_prediksi.json","w",encoding="utf-8") as f:
        json.dump(hasil_akhir, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Selesai → hasil_prediksi.json")

if __name__ == "__main__":
    utama()
