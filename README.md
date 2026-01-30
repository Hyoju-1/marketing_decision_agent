# Marketing Decision Agent

## 개요
이 프로젝트는 구매 가능성·예상 행동 시점과 같은 정량적 예측 결과를 입력으로 받아, **규칙 기반 Agent가 1차 의사결정을 수행하고, Solar가 해당 결정을 해석·설명하고 마케팅 의사결정으로 확장**하는 흐름으로 보여줍니다.

## 목적
- 예측 모델의 출력 결과를 Agent 구조로 연결하는 방법 제시
- LLM(Solar)을 단순 텍스트 생성이 아닌 의사결정 해석 및 설명주체로 활용
- 실제 서비스 환경을 가정한 경량 AI Agent 구조 데모
- 모델 학습은 포함하지 않고 추론 결과를 연결

---

## 간략 파이프라인
```
[고객 행동 로그]
      ↓
(사전 추론 결과)
- 구매 가능성(prob)
- 예상 행동 시점(eta_days)
      ↓
[Marketing Agent Layer]
- 사용자 세분화 (의도 × 시급성)
- 우선순위 점수 계산
- 액션 결정
      ↓
[Upstage Solar]
- 의사결정 근거 설명
- 마케팅 카피 생성
- 실행 요약 리포트
```

---

## 구성 파일
- `behavior_to_agent_pipeline.py`  
  - 고객 행동 로그를 Agent 입력 형식(CSV)으로 변환하는 데모 파이프라인
- `agent_layer.py`  
  - 규칙 기반 의사결정 + Solar 연동 Agent Layer
- `run_agent.py`  
  - Agent + Solar 실행 엔트리포인트

---

## 실행 전 필수 사항

### Upstage Solar API Key 설정

```bash
export UPSTAGE_API_KEY="up_your_api_key"
```

---

## 실행 방법

### 1) 입력 파일 준비
다음 두 CSV가 필요합니다.

**inference_targets.csv**
```csv
user_id,category_id,prob
```

**service_eta_predictions.csv**
```csv
user_id,category_id,eta_days
```

---

### 2) Agent + Solar 실행
```bash
python run_agent.py
```

실행 결과:
- `marketing_segments.csv`
- `marketing_actions.json`
- `marketing_report.txt`
- Solar 기반 설명 및 마케팅 문구 출력

---
