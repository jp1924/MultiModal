# mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration

## Abstract

### 기존 방식의 문제점

NOTE: Multi-Modal에서 각 모달을 합치기 위해 사용하는 모듈의 이름은 논문마다 다를 수 있다. 여기선 각 모달을 정렬하기 위해 사용하는 모듈의 이름을 Cross-Modal Alignment Module라 명칭 함.

- InstructBLIP, LLaVA 등의 기존의 Multi-Modal들은 Multi-Modality를 구현하기 위해 Q-Former, Linear Projection와 같은 Cross-Modal Alignment Module를 사용해 왔음
    하지만 Cross-Modal Alignment Module사용해 Vision, language 모달간의 신호를 완벽하게 정렬하는(혹은 매핑) 것을 불가능 하며, 각 모달간의 간극(Modality Gap)이 존재하게 됨

- 기존의 Multi-Modal들은 Cross-Modal Alignment Module하나의 모듈에 의지해 Multi-Modality를 구현해 왔음.
    Text, Image 데이터 간에는 정보량 차이로 인해 Text & Image 데이터를 처리할 때 Image의 정보량의 간섭이(Modality Interference) 발생해 Text 생성능력이 저하될 수 있음.
    더불어 한= 모달이 과도하게 최적화 되어서 다른 모달들의 성능이 낮아질 수 있음.

### 논문이 제안하는 방식

- 단일 Cross-Modal Alignment Module로 인한 간섭을 완화하고 모달간 협력을 높이기 위해 mPlug-Owl2를 제안
    mPlug-Owl2에 Modality-Adaptive Module(MAM)를 이용해 모달간 협력성을 높임.

## Introduction

- InstructBLIP, Minigpt-4 같은 Multi-Modal들은 모달간 간섭을 최소화 하기 위해 각 모듈을 얼린 상태로 학습힘.
    하지만 각 모듈을 얼려서 학습을 진행하다 보니 각 모달간의 협업성이 떨어짐.(협업성이 떨어지다 -> 모달리티 성능이 낮다)

- mPlug-Owl, LLava와 같은 모달의 협업성을 개선하기 위해 Multi-Modal SFT를 진행하는 하는 방법을 제안.
    하지만 Multi-Modal SFT를 진행하다 보니 각 모달간의 간섭이 발생해 Text 생성 능력이 저하되는 문제가 발생 함.

- 모달리티 간의 간섭은 최소화 하고 협력을 늘리기 위한 방법으로 Modality-Adaptive Module(MAM)이 추가된 mPlug-Owl2를 제안
    각 모달마다 MAM를 추가해 모달리티간 간섭을 최소화 하고 각 모달간 공유할 수 있는 정보를 추출해 모달리티 협력을 유도

- mPlug-Owl2을 위한 2 stage 기반의 학습 방법을 제안 함.
    1-stage로 vision-language pretrain
    2-stage로 join vision-language instruction tunining

## Methodology

### Overview

![Fidure-2](./assets/fig2_page-0001.jpg)

- mPlug-Owl2는 Vision 모델, Vision abstractor, Language 모델 3가지의 주요 파트로 구성되어 있다.
    Language 모델로 GPT, LLaMA2-7B 모델을 사용함. Vision 모델로 ViT-L/14를 사용

### Model Architecture

- mPlug-Owl2는 Vision 모델, Vision abstractor, Language 모델 3가지의 주요 파트로 구성
    Language 모델로 GPT, LLaMA2-7B 모델을 사용함. Vision 모델로 ViT-L/14를 사용

- Vision 모델은 입력된 H * W의 해상도의 이미지를 $\frac{H}{patch} \times \frac{W}{patch}$ 크기의 토큰 시퀀스로 변환한 뒤 Text 임베딩과 결합해 Decoder에 입력
    하지만 Image 해상도에 비례해 토큰 시퀀스의 길이도 비례해서 커진다는 문제가 있음.

- Image에 존재하는 `배경` 같은 중복된 정보로 인해 자원의 낭비와 노이즈가 발생
    이 문재를 해결하기 위해 Vision Abstarctor를 사용해 Vision 신호 내에서 학습에 필요한 정보 만을 추출해 긴 시퀀스를 요약

- Vision Abstarctor는 BLIP-2와 같이 일정 개수의 학습 가능한 Query가 존재.
    이 Query를 이용해 전달받은 Vision 신호로 부터 필요한 정보만을 골라내서 추출 함.
    NOTE: (BLIP-2 처럼 ITC, ITM, ITG 학습을 진행하는 지는 의문, 이건 좀더 봐야할 듯)

- Vision Abstarctor는 다음과 같은 과정을 거쳐서 정보를 압축하게 됨.
    1. Vision 모델이 추출한 $I = [I_1, I_2, \ldots, I_P ] \in \mathbb{R}^{P \times d}$와 $K$개의 학습가능한 쿼리 $Q \in \mathbb{R}^{K \times d}$를 같이 입력

    2. Image 신호간의 관계성을 고려하기 위해 Relative Potisional Encoding을 $I$에 추가함.
        다음 논문에서 이미지 위치 인코딩의 필요성이 입증됨 [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

    3. $V^{i+1}$ 레이어를 가진 Vision Abstractor에서 $\mathcal{C}^{i} = Attn(\mathcal{V}^{i}, [\mathcal{I}; \mathcal{V}^{i}], [\mathcal{I}; \mathcal{V}^{i}])$연산을 수행
        $V^0$번쨰 레이어에는 $K$개의 $Q \in \mathbb{R}^{K \times d}$가 입력됨
        ----- 각 수식의 의미 -----
        $\mathcal{V}^{i}$는 $i$번쨰 레이어에서 압축된 Image 신호
        $[\mathcal{I}; \mathcal{V}^{i}]$는 입력된 Image 신호 $\mathcal{I}$와 $\mathcal{V}^{i}$가 서로 결합된 걸 표현.
        $\mathcal{C}^{i}$는 Self-Attention 연산을 통해 추출해서 얻은 표현

    4. Attention 연산을 통해 추출해서 얻은 표현 $\mathcal{C}^{i}$를 $SwiGLU(\mathcal{C}^{i}W_1)W_2.$에 입력해 $\mathcal{V}^{i+1}$를 얻음
        이때 $W_1$와 $W_2$는 학습가능한 파라메터
        NOTE: (왜 $SwiGLU(\cdot)$을 사용했는지는 의문, 이건 논문을 읽어봐야 할 듯)

    5. 위 과정을 통해서 $I$로부터 시각적 특징을 추출하고 $K$개의 $Q$로 압축을 하게됨.

    6. 결과 $O((P + L)^2)$의 연산량에서 $O((K + L)^2)$로 시간이 줄어듬

위 과정을 거치면서 시각적인 특징이 압축된 Image 토큰 임베딩과 Text 토큰 임베딩을 concat한 뒤 Text Decoder에서 처리함

### Modality-Adaptive Module

- 기존에 Multi-Modality을 구현을 위해 Cross-Modal Alignment Module(Q-Former, Linear Projection)를 사용해 옴
    하지만 Cross-Modal Alignment Module의 부족한 성능 때문에 Vision, language 모델 신호간의 간극(불일치)이 발생할 수 있다.

- 각 모달간 신호 불일치를 피하고자 Modality-Adaptive Module(MAM)를 제안함.
    MAM는 Vision Abstractor로부터 전달받은 Image + Text 신호가 합쳐진 시퀀스 토큰을 입력받게 됨.

- MAM는 다음과 같은 과정을 거쳐서 모달간 신호 간극을 축소함.
    1. Vision 신호와 Language 신호가 서로 합쳐진 $X \in \mathbb{R}^{(L_V + L_T) \times d}$를 입력받음.
        입력할 때 $m \in \{0,1\}$도 같이 입력해 각 시퀀스 별로 어떤 시퀀스가 Vision(0), Language(1)인지를 구분
        NOTE: (token_type_id라 생각하면 됨.)
    2. 각 모달간의 간섭 최소화를 위해 $\phi(X, M, m) = X \odot \mathbb{1}_{\{M = m\}}$ 통해 각 신호를 분리함.
        AND 마스크를 적용시켜서 Vision, Langage 신호를 따로 분리함.
    3. $LN_V(\phi(H_{l-1}, M, 0)) + LN_T(\phi(H_{l-1}, M, 1))$ Vision, Language 별로 분리한 각 신호들에 Layer Normalization을 적용함.
    4. 정규화 까지 거친 Vision, Language 신호를 Attention 연산을 취한다.
        각각 학습 가능한 Projection Layer를 통과시켜 Vision에 대한 Key, Value, Language에 대한 Key, Value 값으로 분리한 뒤 각각 $H^K_{l}$와 $H^V_{l}$를 얻어냄
        다만 Query는 Vision, Language 신호를 분리하지 않은 상태로 Projection Layer를 통과시켜 얻음.
        Key: $H^K_{l} = \phi(\tilde{H}_{l-1}, M, 0) W^{K_0}_l + \phi(\tilde{H}_{l-1}, M, 1) W^{K_1}_l$
        Value: $H^V_{l} = \phi(\tilde{H}_{l-1}, M, 0) W^{V_0}_l + \phi(\tilde{H}_{l-1}, M, 1) W^{V_1}_l$
        Query: $H^Q_{l} = \tilde{H}_{l-1} W^Q_l$

        이후 $H^K_{l}$, $H^V_{l}$, $H^Q_{l}$ 으로 Attention연산을 수행.

        $C_{l} = Softmax\left(\frac{H^Q_{l} {H^K_{l}}^\top}{\sqrt{d}}\right)H^V_{l}$
        NOTE: (근데 이런 과정을 통해서 어떻게 모달간 간극과 간섭을 해결 한다는지 잘 와닿지 않음. 이건 직접 해보면서 볼 것)
    5. Attention이 되어 나온 결과물을 FFN을 거치게 함.

- 각 모달의 신호를 분리해서 처리하기 때문에 모달간의 간섭은 최소화 하면서 Attention연산을 통해 모달별 협력을 향상시킬 수 있다.
NOTE: (신호를 분리를 통해서 모달간 불일치(GAP) 문제도 해결된된다고 하는데 납득은 안됨)

### Training Paradigm

- mPlug-Owl2는 mPlug-Owl1과 유시하게 2 stage의 학습을 통해 Multi-Modality를 학습 함.
    1. 1-stage: Vision 모델, Vision Abstractor, MAM모듈을 훈련시키고 FFN는 고정시킨 상태로 사전학습 하는 단계
    NOTE: (이렇게 학습해도 괜찮은게 맞나? 이러면 당연히 이전에 학습 했던 지식이 영향을 받을탠데?)
    2. 2-stage: multi-modal SFT + Text SFT을 동시에 학습하는 단계
        [mPLUG-2](https://arxiv.org/abs/2302.00402)에서 multi-modal SFT + Text SFT를 동시에 시키면 성능이 좋아진다고 입증 되었다고는 함.
        하지만 아직 읽어보진 않았음.

## Experiments
