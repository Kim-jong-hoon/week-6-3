# Runpad에서 running 해보기

## Nvidia api 키 받기

<img width="1915" height="651" alt="화면 캡처 2025-07-30 111056" src="https://github.com/user-attachments/assets/ba2ee2de-ca0f-4843-b4e4-f963b022dc44" />

## Runpad 실행 후 web termminal 들어가기
<img width="1467" height="618" alt="KakaoTalk_20250730_094808749" src="https://github.com/user-attachments/assets/bd6e783f-558d-411a-941d-f2346f6733a6" />

## 교수님이 주신 코드 입력(! 빼기)
<img width="773" height="642" alt="KakaoTalk_20250730_094854371" src="https://github.com/user-attachments/assets/ab3d7d63-d94c-4558-88f8-0e7e8399e845" />

## Jupyter notebook에서 실행
<img width="1298" height="812" alt="Image20250730112354" src="https://github.com/user-attachments/assets/61e96dcc-dc40-491b-9891-a6af166472a3" />

## detect 영상 결과
<img width="1188" height="796" alt="Image20250730112514" src="https://github.com/user-attachments/assets/9676e9c5-8308-413c-88ca-94fa20cd0d65" />

# 앙상블 방식을 통한 Peoplenet + trafficnet 합치기(Yolo v11을 이용)
# 📌 PeopleNet, TrafficNet, YOLO v11 앙상블 통합 계획

## 1. 프로젝트 개요

- **목표:** 서로 다른 객체 탐지 모델(PeopleNet, TrafficNet, YOLO v11)의 장점을 융합하여, 탐지 정확도와 신뢰도 향상.
- **핵심 아이디어:** 다양한 객체 유형에 특화된 모델의 출력을 통합하여 더 나은 전체 성능 확보.

---

## 2. 모델별 특징 요약

| 모델 | 주요 특징 | 강점 |
|------|-----------|------|
| **PeopleNet** | 사람 중심 객체 탐지 | 사람 탐지 정확도 우수 |
| **TrafficNet** | 교통 객체 탐지 (차량, 신호등 등) | 교통 인식 특화 |
| **YOLO v11** | 범용 객체 탐지 | 속도 빠르고 다양한 객체 대응 가능 |

## 3 . 앙상블 방식 설계
### 📌통합 설계 구조
<br>[PeopleNet Output]
<br>[TrafficNet Output]
<br>[YOLO v11 Output]
<br>        ↓
<br>[결과 정규화 & 클래스 매핑]
<br>        ↓
<br>[중복 제거 (IoU 기반 NMS)]
<br>        ↓
<br>[스코어 재조정 및 앙상블 통합]
<br>        ↓
<br>[최종 Detection 결과]

### 📌단계별 구현 방법
<br>🔹 1) 클래스 통일 (Class Harmonization)
<br>세 모델은 각기 다른 클래스 체계를 가질 수 있음

<br>예: YOLO는 person, car, bus, traffic light 등 다양한 클래스를 갖지만, PeopleNet/TrafficNet은 일부 객체에 집중

<br>→ 공통 클래스 이름 매핑 테이블 생성
```
# 예시: 클래스 통합 매핑
UNIFIED_CLASSES = {
    'person': ['person', 'pedestrian'],
    'car': ['car', 'vehicle'],
    'bus': ['bus'],
    'traffic_light': ['traffic light', 'signal']
}
```

<br>🔹 2) 결과 정규화 (Bounding Box + Score Alignment)
<br>각 모델에서 얻은 바운딩 박스는 (x, y, w, h) 형식이 다를 수 있음 → 통일 필요

<br>Confidence score 역시 범위가 다르므로 min-max normalization 또는 softmax-like scaling

<br>🔹 3) 중복 제거 (NMS 또는 WBF)
<br>📌 옵션 1: Soft-NMS
<br>IoU가 높을 경우 score를 줄이는 방식

<br>단순 NMS보다 정보 손실이 적음

<br>📌 옵션 2: Weighted Box Fusion (WBF)
<br>겹치는 객체에 대해 가중 평균으로 바운딩 박스 좌표 재계산

<br>score가 높은 모델에 더 많은 가중치를 줌

<br>🔹 4) 신뢰도 기반 선택 또는 Voting
<br>모델 간 객체 예측이 일치할 경우 스코어 평균

<br>세 모델 중 2개 이상이 탐지한 객체만 채택하는 Hard Voting도 가능


