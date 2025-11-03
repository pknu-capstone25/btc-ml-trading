# Bitcoin Price Prediction ML

비트코인(BTCUSDT) 1분봉 데이터를 활용한 머신러닝 가격 예측 프로젝트

## 📌 프로젝트 개요

Binance API에서 수집한 과거 데이터를 기반으로 다음 1분의 가격 방향(상승/하락)을 예측하는 머신러닝 모델 개발

## 🚀 주요 기능

- **데이터 수집**: Binance Public Data API를 통한 1분봉 OHLCV 데이터 다운로드
- **데이터 전처리**: 대용량 CSV 분할 처리 (chunk 단위)
- **특성 공학**: 수익률, 변동성, 거래량 변화 등 7개 기본 특성 생성
- **베이스라인 모델**: Logistic Regression 기반 분류 모델
- **백테스팅**: 간단한 PnL 계산 및 성능 평가

## 📦 설치 및 실행

### 1. 데이터 다운로드
```bash
python download.py full     # 전체 다운로드 (최초)
python download.py update   # 업데이트 (기본값)
```

### 2. 데이터 분할 (선택사항)
```bash
python division.py
```

### 3. 모델 학습 및 평가
```bash
python ml_step1_baseline.py
```

## 📂 프로젝트 구조

```
.
├── download.py              # Binance 데이터 다운로드 스크립트
├── division.py              # 대용량 CSV 분할 도구
├── ml_step1_baseline.py     # 베이스라인 ML 모델
└── source/                  # 데이터 저장 폴더
    ├── 1m_history.csv       # 전체 데이터
    └── chunk_*.csv          # 분할된 데이터
```

## 🛠 기술 스택

- **Python 3.x**
- **Pandas**: 데이터 처리
- **NumPy**: 수치 계산
- **Scikit-learn**: 머신러닝 모델
- **Binance API**: 데이터 수집

## 📊 현재 모델 성능

- **모델**: Logistic Regression (베이스라인)
- **데이터**: 약 300만행
- **Train/Test 분할**: 80:20 (시계열 고려)
- **평가 지표**: Accuracy, Balanced Accuracy, Classification Report

## 🔮 향후 계획

- [ ] 기술적 지표 추가 (RSI, MACD, Bollinger Bands)
- [ ] 시간 특성 One-Hot Encoding
- [ ] Train/Validation/Test 3분할
- [ ] 하이퍼파라미터 튜닝
- [ ] 고급 모델 실험 (XGBoost, LSTM)
- [ ] Cross Validation 도입

## 📝 라이선스

MIT License

## 👤 작성자

개인 학습 프로젝트

