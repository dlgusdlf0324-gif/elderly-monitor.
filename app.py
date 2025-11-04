"""
Elderly Monitor – 단일 파일 웹사이트(Flask)
-------------------------------------------------
기능 요약:
- 사용자 등록(이름, 연락처, 집 좌표, 지오펜스 반경)
- 센서 데이터 업로드(위치/수도/가스/활동/문상태)
- 이상징후 자동 판정(수도/가스 z-score, 활동 패턴, 지오펜스 이탈, 복합규칙)
- 대시보드(사용자/최근 데이터/알림 내역 시각화 + 수동 업로드 폼)
- 주기적 검사 스케줄러(15분)

실행 방법:
1) Python 3.10+ 권장
2) pip install -U flask flask_sqlalchemy pandas scikit-learn apscheduler
3) python app.py 실행 후 http://127.0.0.1:5000 접속

운영 참고:
- 실제 SMS/이메일 연동은 send_sms()/send_email()에 API 키 세팅 필요
- SQLite 파일: elderly_monitor.db (동일 디렉토리)
- 데모/프로토타입 용도로 설계(보안/권한, 암호화, 감사로그 등은 운영 시 강화 필요)
"""

from flask import Flask, request, jsonify, render_template_string
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.ensemble import IsolationForest  # 선택적 사용(아래 주석 예시)
import pandas as pd
import numpy as np
import json, math

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///elderly_monitor.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JSON_AS_ASCII'] = False

db = SQLAlchemy(app)

# ----------------------------- DB 모델 -----------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False)
    phone = db.Column(db.String, nullable=True)
    email = db.Column(db.String, nullable=True)
    home_lat = db.Column(db.Float, nullable=True)
    home_lon = db.Column(db.Float, nullable=True)
    geofence_m = db.Column(db.Integer, default=500)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SensorReading(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    lat = db.Column(db.Float, nullable=True)
    lon = db.Column(db.Float, nullable=True)
    water_l = db.Column(db.Float, default=0.0)
    gas_m3 = db.Column(db.Float, default=0.0)
    motion = db.Column(db.Integer, default=0)
    door_open = db.Column(db.Integer, default=0)
    meta = db.Column(db.String, default='{}')

class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    alert_type = db.Column(db.String)
    details = db.Column(db.String)
    sent_to = db.Column(db.String)

with app.app_context():
    db.create_all()

# ----------------------------- 유틸 -----------------------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# ----------------------------- 알림(플레이스홀더) -----------------------------

def send_email(to_email, subject, body):
    print(f"[EMAIL] to={to_email} subj={subject} body={body}")


def send_sms(to_phone, message):
    print(f"[SMS] to={to_phone} msg={message}")


def record_and_send_alert(user_id, alert_type, details):
    user = User.query.get(user_id)
    sent_to = []
    if user and user.email:
        send_email(user.email, f"[알림] {alert_type}", details)
        sent_to.append(user.email)
    if user and user.phone:
        send_sms(user.phone, details)
        sent_to.append(user.phone)
    alert = Alert(user_id=user_id, alert_type=alert_type, details=details, sent_to=",".join(sent_to))
    db.session.add(alert)
    db.session.commit()

# ----------------------------- 통계/베이스라인 -----------------------------

def compute_baselines(user_id, lookback_days=14):
    since = datetime.utcnow() - timedelta(days=lookback_days)
    rows = SensorReading.query.filter(
        SensorReading.user_id == user_id,
        SensorReading.timestamp >= since
    ).all()
    if not rows:
        return {}
    df = pd.DataFrame([{
        'timestamp': r.timestamp,
        'water_l': r.water_l,
        'gas_m3': r.gas_m3,
        'motion': r.motion
    } for r in rows])
    baselines = {}
    for m in ['water_l', 'gas_m3', 'motion']:
        mean = float(df[m].mean())
        std = float(df[m].std(ddof=0) if len(df[m]) > 1 else 0.0)
        baselines[m] = {'mean': mean, 'std': std}
    return baselines


def check_latest_for_user(user_id):
    user = User.query.get(user_id)
    if not user:
        return
    last = SensorReading.query.filter_by(user_id=user_id).order_by(SensorReading.timestamp.desc()).first()
    if not last:
        return

    baselines = compute_baselines(user_id)
    details = []
    high_risk = False

    # 수도 이상 탐지
    if 'water_l' in baselines and baselines['water_l']['std'] > 0:
        z = (last.water_l - baselines['water_l']['mean']) / baselines['water_l']['std']
        if abs(z) >= 3:
            details.append(f"수도 사용량 이상: 최근={last.water_l}L, 평균={baselines['water_l']['mean']:.1f}L, z={z:.2f}")
            if z > 3:
                high_risk = True
    else:
        if baselines.get('water_l', {}).get('mean', 0) < 0.1 and last.water_l > 5:
            details.append(f"수도 사용 급증: {last.water_l}L (평소 거의 사용 없음)")
            high_risk = True

    # 가스 이상 탐지
    if 'gas_m3' in baselines and baselines['gas_m3']['std'] > 0:
        z = (last.gas_m3 - baselines['gas_m3']['mean']) / baselines['gas_m3']['std']
        if abs(z) >= 3:
            details.append(f"가스 사용량 이상: 최근={last.gas_m3}m³, 평균={baselines['gas_m3']['mean']:.2f}m³, z={z:.2f}")
            if z > 3:
                high_risk = True
    else:
        if baselines.get('gas_m3', {}).get('mean', 0) < 0.01 and last.gas_m3 > 0.1:
            details.append(f"가스 사용 급증: {last.gas_m3}m³ (평소 거의 사용 없음)")
            high_risk = True

    # 활동 이상
    if 'motion' in baselines:
        mean = baselines['motion']['mean']
        if mean >= 0.5 and last.motion == 0:
            details.append("활동 이상(평소 활동적이나 현재 무활동)")
        if mean < 0.2 and last.motion == 1:
            details.append("활동 이상(평소 거의 없음에도 현재 활동 감지)")

    # 지오펜스
    if last.lat and last.lon and user.home_lat and user.home_lon:
        d = haversine(last.lat, last.lon, user.home_lat, user.home_lon)
        if d > user.geofence_m:
            details.append(f"지오펜스 이탈: 거리 {int(d)}m (설정 {user.geofence_m}m)")
            if last.timestamp.hour >= 23 or last.timestamp.hour < 6:
                high_risk = True

    # 복합 규칙: 집 안 + 자원 급증
    if last.lat and last.lon and user.home_lat and user.home_lon:
        d = haversine(last.lat, last.lon, user.home_lat, user.home_lon)
        at_home = d <= user.geofence_m
        if at_home and any(("급증" in s) or ("이상" in s) for s in details):
            high_risk = True

    if details:
        level = "HIGH" if high_risk else "LOW"
        rec = "; ".join(details)
        record_and_send_alert(user_id, f"{level} 이상징후", rec)

# ----------------------------- 스케줄러 -----------------------------

sched = BackgroundScheduler()

@sched.scheduled_job('interval', minutes=15)
def periodic_check_all():
    users = User.query.all()
    for u in users:
        check_latest_for_user(u.id)

sched.start()

# ----------------------------- API -----------------------------

@app.route('/api/users', methods=['GET'])
def api_list_users():
    users = User.query.order_by(User.id.asc()).all()
    return jsonify([{
        'id': u.id,
        'name': u.name,
        'phone': u.phone,
        'email': u.email,
        'home_lat': u.home_lat,
        'home_lon': u.home_lon,
        'geofence_m': u.geofence_m,
        'created_at': u.created_at.isoformat()
    } for u in users])


@app.route('/api/users', methods=['POST'])
def api_create_user():
    data = request.json
    u = User(
        name=data.get('name'),
        phone=data.get('phone'),
        email=data.get('email'),
        home_lat=data.get('home_lat'),
        home_lon=data.get('home_lon'),
        geofence_m=data.get('geofence_m', 500)
    )
    db.session.add(u)
    db.session.commit()
    return jsonify({'status': 'ok', 'user_id': u.id})


@app.route('/api/sensor', methods=['POST'])
def api_upload_sensor():
    data = request.json
    r = SensorReading(
        user_id=int(data['user_id']),
        timestamp=datetime.fromisoformat(data.get('timestamp')) if data.get('timestamp') else datetime.utcnow(),
        lat=data.get('lat'),
        lon=data.get('lon'),
        water_l=float(data.get('water_l', 0)),
        gas_m3=float(data.get('gas_m3', 0)),
        motion=int(data.get('motion', 0)),
        door_open=int(data.get('door_open', 0)),
        meta=json.dumps(data.get('meta', {}))
    )
    db.session.add(r)
    db.session.commit()
    # 업로드 즉시 체크
    check_latest_for_user(r.user_id)
    return jsonify({'status': 'ok'})


@app.route('/api/latest', methods=['GET'])
def api_latest_reading():
    user_id = int(request.args.get('user_id'))
    r = SensorReading.query.filter_by(user_id=user_id).order_by(SensorReading.timestamp.desc()).first()
    if not r:
        return jsonify({'status': 'no data'})
    return jsonify({
        'timestamp': r.timestamp.isoformat(),
        'lat': r.lat, 'lon': r.lon,
        'water_l': r.water_l, 'gas_m3': r.gas_m3, 'motion': r.motion, 'door_open': r.door_open
    })


@app.route('/api/alerts', methods=['GET'])
def api_alerts():
    user_id = request.args.get('user_id')
    q = Alert.query
    if user_id:
        q = q.filter_by(user_id=int(user_id))
    alerts = q.order_by(Alert.timestamp.desc()).limit(200).all()
    return jsonify([{
        'id': a.id,
        'user_id': a.user_id,
        'timestamp': a.timestamp.isoformat(),
        'alert_type': a.alert_type,
        'details': a.details,
        'sent_to': a.sent_to
    } for a in alerts])

# ----------------------------- 웹 UI -----------------------------
INDEX_HTML = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Elderly Monitor Dashboard</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" />
  <style>
    body { background:#f8fafc; }
    .card { border:none; border-radius:1rem; box-shadow:0 8px 24px rgba(0,0,0,0.06); }
    .badge-soft { background:#eef2ff; color:#3730a3; }
    .table thead { background:#f1f5f9; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
  </style>
</head>
<body>
<div class="container py-4">
  <div class="d-flex justify-content-between align-items-center mb-4">
    <h2 class="fw-bold">독거노인 모니터링 대시보드</h2>
    <span class="badge badge-soft px-3 py-2">프로토타입</span>
  </div>

  <div class="row g-4">
    <!-- 사용자 목록 -->
    <div class="col-lg-4">
      <div class="card h-100">
        <div class="card-body">
          <div class="d-flex justify-content-between align-items-center mb-3">
            <h5 class="mb-0">사용자</h5>
            <button class="btn btn-sm btn-primary" data-bs-toggle="modal" data-bs-target="#userModal">+ 등록</button>
          </div>
          <ul id="userList" class="list-group list-group-flush"></ul>
        </div>
      </div>
    </div>

    <!-- 최근 데이터 & 업로드 -->
    <div class="col-lg-5">
      <div class="card h-100">
        <div class="card-body">
          <div class="d-flex justify-content-between align-items-center mb-3">
            <h5 class="mb-0">최근 데이터</h5>
            <button class="btn btn-sm btn-outline-secondary" id="btnRefresh">새로고침</button>
          </div>
          <div id="latestBox" class="small mono text-muted">사용자를 선택하세요.</div>
          <hr/>
          <h6 class="mb-2">수동 업로드</h6>
          <form id="uploadForm" class="row g-2">
            <div class="col-6">
              <label class="form-label">사용자</label>
              <select class="form-select" id="uploadUser"></select>
            </div>
            <div class="col-6">
              <label class="form-label">시간(ISO)</label>
              <input class="form-control" id="ts" placeholder="빈칸=지금"/>
            </div>
            <div class="col-6"><label class="form-label">위도</label><input class="form-control" id="lat" type="number" step="any"></div>
            <div class="col-6"><label class="form-label">경도</label><input class="form-control" id="lon" type="number" step="any"></div>
            <div class="col-6"><label class="form-label">수도(L)</label><input class="form-control" id="water" type="number" step="any" value="0"></div>
            <div class="col-6"><label class="form-label">가스(m³)</label><input class="form-control" id="gas" type="number" step="any" value="0"></div>
            <div class="col-6"><label class="form-label">활동(0/1)</label><input class="form-control" id="motion" type="number" value="0"></div>
            <div class="col-6"><label class="form-label">문열림(0/1)</label><input class="form-control" id="door" type="number" value="0"></div>
            <div class="col-12 d-grid"><button class="btn btn-success" type="submit">업로드</button></div>
          </form>
        </div>
      </div>
    </div>

    <!-- 알림 -->
    <div class="col-lg-3">
      <div class="card h-100">
        <div class="card-body">
          <div class="d-flex justify-content-between align-items-center mb-3">
            <h5 class="mb-0">알림</h5>
            <button class="btn btn-sm btn-outline-secondary" id="btnAlertRefresh">새로고침</button>
          </div>
          <div class="table-responsive" style="max-height:420px; overflow:auto;">
            <table class="table table-sm align-middle">
              <thead><tr><th>시간</th><th>유형</th><th>내용</th></tr></thead>
              <tbody id="alertTable"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- 사용자 등록 모달 -->
<div class="modal fade" id="userModal" tabindex="-1">
  <div class="modal-dialog">
    <div class="modal-content">
      <div class="modal-header"><h5 class="modal-title">사용자 등록</h5><button class="btn-close" data-bs-dismiss="modal"></button></div>
      <div class="modal-body">
        <form id="userForm" class="row g-2">
          <div class="col-6"><label class="form-label">이름</label><input class="form-control" id="name" required></div>
          <div class="col-6"><label class="form-label">전화</label><input class="form-control" id="phone"></div>
          <div class="col-12"><label class="form-label">이메일</label><input class="form-control" id="email"></div>
          <div class="col-6"><label class="form-label">집 위도</label><input class="form-control" id="homeLat" type="number" step="any"></div>
          <div class="col-6"><label class="form-label">집 경도</label><input class="form-control" id="homeLon" type="number" step="any"></div>
          <div class="col-12"><label class="form-label">지오펜스 반경(m)</label><input class="form-control" id="geofence" type="number" value="300"></div>
          <div class="col-12 d-grid"><button class="btn btn-primary" type="submit">저장</button></div>
        </form>
      </div>
    </div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
let currentUser = null;

async function loadUsers(){
  const res = await fetch('/api/users');
  const users = await res.json();
  const list = document.getElementById('userList');
  list.innerHTML = '';
  const sel = document.getElementById('uploadUser');
  sel.innerHTML='';
  users.forEach(u=>{
    const li = document.createElement('li');
    li.className='list-group-item d-flex justify-content-between align-items-center';
    li.innerHTML = `<div><div class="fw-semibold">${u.name}</div><div class="small text-muted">#${u.id} · ${u.phone||''} ${u.email||''}</div></div>`+
                   `<button class="btn btn-sm btn-outline-primary">선택</button>`;
    li.querySelector('button').onclick = ()=>{ currentUser = u; showLatest(); loadAlerts(); };
    list.appendChild(li);

    const opt = document.createElement('option');
    opt.value = u.id; opt.textContent = `${u.id} - ${u.name}`;
    sel.appendChild(opt);
  });
}

async function showLatest(){
  const box = document.getElementById('latestBox');
  if(!currentUser){ box.textContent='사용자를 선택하세요.'; return; }
  const res = await fetch(`/api/latest?user_id=${currentUser.id}`);
  const data = await res.json();
  if(data.status==='no data'){ box.textContent='데이터 없음'; return; }
  box.innerHTML = `
    <div><b>사용자:</b> ${currentUser.name} (#${currentUser.id})</div>
    <div><b>시간:</b> ${data.timestamp}</div>
    <div><b>좌표:</b> ${data.lat??'-'}, ${data.lon??'-'}</div>
    <div><b>수도(L):</b> ${data.water_l} · <b>가스(m³):</b> ${data.gas_m3}</div>
    <div><b>활동:</b> ${data.motion} · <b>문열림:</b> ${data.door_open}</div>
  `;
}

async function loadAlerts(){
  const tbody = document.getElementById('alertTable');
  const uidParam = currentUser? `?user_id=${currentUser.id}` : '';
  const res = await fetch('/api/alerts'+uidParam);
  const rows = await res.json();
  tbody.innerHTML = '';
  rows.forEach(a=>{
    const tr = document.createElement('tr');
    tr.innerHTML = `<td class='small mono'>${a.timestamp}</td><td><span class='badge text-bg-warning'>${a.alert_type}</span></td><td class='small'>${a.details}</td>`;
    tbody.appendChild(tr);
  })
}

// 폼 이벤트

document.getElementById('userForm').addEventListener('submit', async (e)=>{
  e.preventDefault();
  const payload = {
    name: document.getElementById('name').value,
    phone: document.getElementById('phone').value,
    email: document.getElementById('email').value,
    home_lat: parseFloat(document.getElementById('homeLat').value||null),
    home_lon: parseFloat(document.getElementById('homeLon').value||null),
    geofence_m: parseInt(document.getElementById('geofence').value||300)
  };
  await fetch('/api/users',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  document.querySelector('#userModal .btn-close').click();
  await loadUsers();
});

document.getElementById('uploadForm').addEventListener('submit', async (e)=>{
  e.preventDefault();
  const payload = {
    user_id: parseInt(document.getElementById('uploadUser').value),
    timestamp: document.getElementById('ts').value || null,
    lat: document.getElementById('lat').value ? parseFloat(document.getElementById('lat').value) : null,
    lon: document.getElementById('lon').value ? parseFloat(document.getElementById('lon').value) : null,
    water_l: parseFloat(document.getElementById('water').value||0),
    gas_m3: parseFloat(document.getElementById('gas').value||0),
    motion: parseInt(document.getElementById('motion').value||0),
    door_open: parseInt(document.getElementById('door').value||0)
  };
  await fetch('/api/sensor',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  if(currentUser && currentUser.id===payload.user_id){
    showLatest(); loadAlerts();
  }
});

// 버튼

document.getElementById('btnRefresh').onclick = showLatest;
document.getElementById('btnAlertRefresh').onclick = loadAlerts;

// 주기 갱신(30초)
setInterval(()=>{ if(currentUser){ showLatest(); loadAlerts(); } }, 30000);

// 초기 로드
loadUsers();
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)


if __name__ == '__main__':
    # 개발용 서버 실행
    app.run(host='0.0.0.0', port=5000, debug=True)


# 배포(무료) — Render로 5분 컷 가이드
다음 3개 파일을 프로젝트 루트에 추가하시고, 깃허브에 푸시한 뒤 Render에 연결하시면 곧바로 공용 URL이 발급됩니다.

## 1) requirements.txt
```
flask==3.0.3
flask_sqlalchemy==3.1.1
pandas==2.2.2
scikit-learn==1.5.2
apscheduler==3.10.4
gunicorn==22.0.0
```

## 2) Procfile  (대소문자 주의)
```
web: gunicorn app:app --workers=2 --threads=4 --timeout=120
```
> 파일명이 `app.py`이고 Flask 인스턴스가 `app`이므로 `app:app`을 사용합니다.

## 3) render.yaml  (선택·권장: 인프라 자동 생성)
```
services:
  - type: web
    name: elderly-monitor
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --workers=2 --threads=4 --timeout=120
    autoDeploy: true
    region: singapore
    envVars: []
```

---
