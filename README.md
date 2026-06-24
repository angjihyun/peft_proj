# PEFT Layer Placement Analysis

한국어 인코더 모델(klue/roberta-base)에서 **LoRA·Adapter의 레이어 위치가 태스크 성능에 미치는 영향**을 분석하는 연구 프로젝트입니다. 기존 PEFT 위치 관련 연구가 영어 기반 대형 디코더 모델에 집중된 것과 달리, 한국어 인코더 모델을 대상으로 LoRA와 Adapter의 최적 레이어/모듈 위치를 동일 파라미터 예산(Same Budget) 조건에서 체계적으로 비교했습니다.


---

## Motivation

실무에서 LoRA/Adapter는 Transformer 전체 레이어에 일괄 적용하는 것이 관행입니다. 이 관행에는 "모든 레이어가 태스크 적응에 동등하게 기여한다"는 가정이 암묵적으로 깔려 있습니다.

하지만 선행 연구(Jawahar et al., ACL 2019 / Tenney et al., ACL 2019)에 따르면 레이어마다 처리하는 언어 정보가 다릅니다.

| 구간 | 처리 정보 |
|---|---|
| 하위 (L1–L4) | 어휘 · 형태 · 표층 특징 |
| 중간 (L5–L8) | 통사 구조 · 품사 · 의존 관계 |
| 상위 (L9–L12) | 의미 관계 · 추론 · 공지시 |

→ 태스크 성격(SA: 어휘 중심 / NLI: 논리 추론)에 따라 최적 PEFT 위치도 달라질 것이라는 가설에서 출발했습니다.

기존 연구(Act-LoRA, FLoE, Layer Card, MoLA, Fomenko et al.)는 대부분 영어·디코더 LLM을 대상으로 하며, 한국어 인코더 모델에서 파라미터 예산을 엄격히 통제한 위치 효과 비교는 다루지 않습니다. 본 연구는 이 공백을 채우는 것을 목표로 합니다.

---

## Research Questions

- **RQ1.** klue/roberta-base에서 LoRA·Adapter의 최적 레이어 위치는 어디인가?
- **RQ2.** 최적 위치는 태스크 유형(SA vs NLI)에 따라 달라지는가?
- **RQ3.** 선별적 레이어 적용이 전체 적용 대비 파라미터 효율성에서 이점이 있는가?
- **RQ4.** Q, K, V 모듈 중 어느 것에 적용하는 것이 최적이며, 태스크에 따라 달라지는가?
- **RQ5.** LoRA와 Adapter는 레이어 위치 민감도에서 차이를 보이는가?

---

## Experimental Design

### Same Budget 원칙

레이어/모듈 수를 줄이면 파라미터 수도 함께 줄어, 성능 차이가 '위치 효과'인지 '파라미터 부족'인지 구별할 수 없습니다. 이를 제거하기 위해 레이어·모듈 수가 줄면 rank를 높여 모든 조건의 trainable parameter를 동일하게(±5% 이내) 통제했습니다.

| 조건 | 모듈 수 | rank | LoRA params |
|---|---|---|---|
| All-12 (QV) | 2 | 4 | 147,456 |
| 4-layer (QV) | 2 | 12 | 147,456 |
| Q-only | 1 | 24 | 147,456 |
| K-only | 1 | 24 | 147,456 |
| V-only | 1 | 24 | 147,456 |
| QKV | 3 | 8 | 147,456 |

Adapter는 구조적 특성상 파라미터 수가 LoRA 대비 약 2.6배로, 완전한 1:1 통제에는 한계가 있어 결과 해석 시 이를 고려했습니다.

### Phase 1 — 레이어 선택 전략 (모듈: QV 고정)

| 전략 | 내용 |
|---|---|
| All-12 | 전체 12개 레이어 (기존 관행 기준선) |
| First-Last-4 | L0, L1, L10, L11 (probing 이론 기반) |
| Random-4 | 무작위 4개 레이어 (위치 효과 통제) |
| Select-4 | ΔW Norm 상위 Top-4 (데이터 기반 선택) |

### Phase 2 — 모듈 선택 전략 (레이어: First-Last-4 고정)

| 전략 | rank | 내용 |
|---|---|---|
| QV | 12 | LoRA 원논문 권장 / 기준선 |
| Q-only | 24 | Query만 단독 적용 |
| K-only | 24 | Key만 단독 적용 |
| V-only | 24 | Value만 단독 적용 |
| QKV | 8 | 전체 모듈 적용 |

**공통 설정**: klue/roberta-base · SA(NSMC 150K) / NLI(KLUE-NLI 25K) · Epoch 5 · Batch 32 · MaxLen 128 · AdamW · Seed 42/123/456 (3회 반복)

### Select-4 방법론 (ΔW Norm 기반 자동 레이어 선택)

이론적 가정이 아니라 실제 학습 결과에 기반해 중요 레이어를 데이터 기반으로 선정하는 방식입니다.

1. All-12 LoRA를 3 seed로 독립 학습
2. 학습 완료 후 레이어별 ΔW norm(‖BA‖_F) 측정
3. 3-seed 평균 기준 상위 4개 레이어 선정
4. 선택된 레이어만 QV·rank=12로 재학습

```python
def get_top_layers_by_grad_norm(model, top_k=4):
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    sorted_layers = sorted(grad_norms, key=grad_norms.get, reverse=True)
    return sorted_layers[:top_k]
```

---

## Results

### Layer 실험 — SA (NSMC)

- LoRA-All-12(0.66% 파라미터)가 FFT(100%)를 **+0.18%p** 초과 — low-rank 제약이 암묵적 정규화로 작용했을 가능성
- First-Last-4: −0.41%p 손실로 28.7% 시간 절감 가능 / Random-4: −0.27%p — SA에서는 레이어 위치 자체의 영향이 작음
- Select-4가 First-Last-4보다 오히려 성능이 낮음 — ΔW norm은 "많이 변한 레이어"를 식별할 뿐 "가장 중요한 레이어"와 항상 일치하지는 않음

### Layer 실험 — NLI (KLUE-NLI)

- SA vs NLI 격차: Select-4 조건에서 최대 **−6.60%p** (약 11.6배 차이)
- All-12도 FFT 대비 −3.10%p 격차 — LoRA 자체의 표현력 한계도 존재
- Random-4 표준편차: NLI 2.86% vs SA 0.07% — NLI는 레이어 조합에 따라 수렴 자체가 불안정

### Module 실험 — SA vs NLI (레이어: First-Last-4 고정)

**SA (파라미터 739K)**

| 순위 | 모듈 | Mean Acc | Std |
|---|---|---|---|
| 1 | Q-only (r=24) | 90.04% | 0.07% |
| 2 | QV (r=12) | 89.97% | 0.04% |
| 3 | K-only (r=24) | 89.20% | 0.06% |
| 4 | V-only (r=24) | 89.18% | 0.07% |
| 5 | QKV (r=8) | 88.89% | 0.13% |

**NLI (파라미터 147K)**

| 순위 | 모듈 | Mean Acc | Std |
|---|---|---|---|
| 1 | QKV (r=8) | 78.46% | 0.35% |
| 2 | QV (r=12) | 77.93% | 0.80% |
| 3 | V-only (r=24) | 76.38% | 0.23% |
| 4 | K-only (r=24) | 54.54% | 2.20% |
| 5 | Q-only (r=24) | 48.61% | 0.80% |

→ SA에서 1위였던 Q-only가 NLI에서는 거의 랜덤 베이스라인(33%) 수준으로 붕괴. **복잡한 추론 태스크일수록 단일 모듈 집중이 아니라 Q/K/V를 고르게 적응시켜야 학습이 안정적으로 수렴**함. SA·NLI는 파라미터 기준(739K vs 147K)이 달라 직접 비교에는 한계가 있음.

### Adapter vs LoRA

| 태스크 | 결과 |
|---|---|
| SA | Adapter-All-12 (91.14%) > LoRA-All-12 (90.44%) > FFT (90.26%) |
| NLI | Adapter-All-12 (85.20%) ≈ FFT (85.81%) ≫ LoRA-All-12 (82.71%) |

Adapter의 비선형(ReLU) 표현력이 복잡 태스크에서 더 큰 이점을 제공하나, 파라미터 2.6배·추론 지연이라는 트레이드오프가 존재. NLI에서 Adapter Select-4(L7,8,10,11)가 First-Last-4 대비 +2.06%p 우수 — 중간층(L7, L8) 포함 여부가 결정적.

### ΔW Norm / CKA / Gradient Norm 분석

- ΔW Norm: SA는 L11이 3 seed 모두 Top-1 (순수 상위 레이어 집중), NLI는 L11이 압도적 Top-1이면서 L7·L8도 일관되게 상위권 → 통사+의미 동시 필요
- CKA: 모든 조건에서 L12(분류 헤드 직전)가 가장 많이 변형. NLI K-only는 변형량(CKA=0.2689)이 가장 크지만 성능은 최저 — **변형량이 곧 성능 기여도를 의미하지는 않음**
- Gradient Norm: 학습 초반 100 step 기준으로도 상위 레이어(L10/L11) 중심 패턴이 ΔW norm 분석과 일관 — Select-4 선정의 타당성을 뒷받침

---

## 가설 검증 종합

| 가설 | 내용 | 결과 |
|---|---|---|
| H1 | SA: Early-Middle 레이어 최적 | ✗ 기각 — 상위 레이어(L8–11) 집중이 핵심 |
| H2 | NLI: Late 레이어 최적 | ▲ 부분 지지 — Late 단독이 아닌 중간+상위 혼합 |
| H3 | 선별 적용 > 전체 적용 | △ 조건부 — SA는 근사 가능, NLI는 불가 |
| H4 | 태스크별 최적 모듈이 다르다 | ✓ 강하게 지지 — SA·NLI에서 1위↔5위 완전 반전 |
| H5 | LoRA vs Adapter 레이어 민감도 차이 | ✓ 지지 — NLI에서 중간층 포함 여부가 결정적 |

## 핵심 발견 4가지

1. **태스크 복잡도가 필요 레이어 수를 결정한다.** SA의 4-layer 선별은 −0.41%p 손실로 28.7% 시간 절감 가능하지만, 동일 조건에서 NLI는 최대 −6.60%p 손실. 11.6배 격차는 '레이어를 줄이면 무조건 효율적'이라는 관행에 반례를 제공.
2. **LoRA는 단순 분류에서 FFT를 능가할 수 있다.** SA에서 LoRA-All-12(0.66% 파라미터)가 FFT를 +0.18%p 초과. NLI에서는 FFT에 +3.10%p 뒤져 복잡 추론에서의 표현력 한계가 드러남.
3. **최적 어텐션 모듈이 태스크에 따라 완전히 반전된다.** SA: QKV≈QV>V>K>Q / NLI: Q>QV>K>V>QKV. LoRA 원논문이 권장하는 QV 조합이 모든 태스크에 최적은 아님.
4. **Adapter가 성능에서 LoRA를 능가하며, 레이어 선택 전략도 다르게 반응한다.** ReLU 비선형 표현력이 복잡 태스크에 이점을 주지만, 파라미터 2.6배·추론 지연 트레이드오프 존재.

---

## Limitations & Future Work

**한계**
- LoRA-LoRA 간 비교는 Same Budget이 철저히 적용되었으나, Adapter(~1.96M)는 구조적으로 LoRA(~0.74M) 대비 2.6배 파라미터가 많아 Adapter 우위가 '비선형성' 덕분인지 '절대적 파라미터 수' 덕분인지 완전히 분리하지 못함
- NLI 최적 레이어(L7,8,10,11) 환경에서의 모듈 실험 등 교차 최적화 탐색은 진행하지 않음

**향후 연구**
- Adapter 병목 차원(d_b)을 극소화해 LoRA와 파라미터 예산을 1:1로 맞춘 재실험 → 비선형성의 순수 효과 측정
- Act-LoRA처럼 학습 중 Activation을 추적하는 Dynamic Routing 적용 → 레이어 중요도를 학습 중 실시간 반영

---

## Tech Stack

`klue/roberta-base` · PyTorch · HuggingFace Transformers/PEFT · Weights & Biases (실험 관리)
