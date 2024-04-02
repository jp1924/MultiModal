# mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration

## Abstract

### 기존 방식의 문제점

NOTE: Multi-Modal에서 각 모달을 합치기 위해 사용하는 모듈의 이름은 논문마다 다를 수 있다. 여기선 각 모달을 정렬하기 위해 사용하는 모듈의 이름을 Cross-Modal Alignment Module라 명칭 함.

- InstructBLIP, LLaVA 등의 기존의 Multi-Modal들은 Multi-Modality를 구현하기 위해 Q-Former, Linear Projection와 같은 Cross-Modal Alignment Module를 사용해 왔음
    하지만 Cross-Modal Alignment Module사용해 Vision, language 모달간의 신호를 완벽하게 정렬하는(혹은 매핑) 것을 불가능 하며, 각 모달간의 간극(Modality Gap)이 존재하게 됨

- 기존의 Multi-Modal들은 Cross-Modal Alignment Module하나의 모듈에 의지해 Multi-Modality를 구현해 왔음.
    Text, Image 데이터 간에는 정보량 차이로 인해 Text & Image 데이터를 처리할 때 Image의 정보량의 간섭이(Modality Interference) 발생해 Text 생성능력이 저하될 수 있음.
    더불어 한 모달이 과도하게 최적화 되어서 다른 모달들의 성능이 낮아질 수 있음.

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

![hustlin_erd](./assets/fig2.pdf)

- mPlug-Owl2는 Vision 모델, Vision abstractor, Language 모델 3가지의 주요 파트로 구성되어 있다.
    Language 모델로 GPT, LLaMA2-7B 모델을 사용함. Vision 모델로 ViT-L/14를 사용

### Model Architecture

- mPlug-Owl2는 Vision 모델, Vision abstractor, Language 모델 3가지의 주요 파트로 구성
    Language 모델로 GPT, LLaMA2-7B 모델을 사용함. Vision 모델로 ViT-L/14를 사용

- Vision 모델은 입력된 H * W의 해상도의 이미지를 $\frac{H}{patch} \times \frac{W}{patch}$ 크기의 토큰 시퀀스로 변환한 뒤 Text 임베딩과 결합해 Decoder에 입력
    하지만 Image 해상도에 비례해 토큰 시퀀스의 길이도 비례해서 커진다는 문제가 있음.

- Image에 존재하는 `배경` 같은 중복된 정보로 인해 자원의 낭비와 노이즈가 발생<br>
    이 문재를 해결하기 위해 Vision Abstarctor를 사용해 Vision 신호 내에서 학습에 필요한 정보 만을 추출해 긴 시퀀스를 요약

- Vision Abstarctor는 BLIP-2와 같이 일정 개수의 학습 가능한 Query가 존재.<br>
    이 Query를 이용해 전달받은 Vision 신호로 부터 필요한 정보만을 골라내서 추출 함.<br>
    NOTE: (BLIP-2 처럼 ITC, ITM, ITG 학습을 진행하는 지는 의문, 이건 좀더 봐야할 듯)

- Vision Abstarctor는 다음과 같은 과정을 거쳐서 정보를 압축하게 됨.<br>
    1. Vision 모델이 추출한 $I = [I_1, I_2, \ldots, I_P ] \in \mathbb{R}^{P \times d}$와 $K$개의 학습가능한 쿼리 $Q \in \mathbb{R}^{K \times d}$를 같이 입력<br>
    <br>
    2. Image 신호간의 관계성을 고려하기 위해 Relative Potisional Encoding을 $I$에 추가함.<br>
        다음 논문에서 이미지 위치 인코딩의 필요성이 입증됨 [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) <br>
    <br>
    3. $V^{i+1}$ 레이어를 가진 Vision Abstractor에서 $\mathcal{C}^{i} = Attn(\mathcal{V}^{i}, [\mathcal{I}; \mathcal{V}^{i}], [\mathcal{I}; \mathcal{V}^{i}])$연산을 수행<br>
        $V^0$번쨰 레이어에는 $K$개의 $Q \in \mathbb{R}^{K \times d}$가 입력됨<br>
        ----- 각 수식의 의미 -----<br>
        $\mathcal{V}^{i}$는 $i$번쨰 레이어에서 압축된 Image 신호<br>
        $[\mathcal{I}; \mathcal{V}^{i}]$는 입력된 Image 신호 $\mathcal{I}$와 $\mathcal{V}^{i}$가 서로 결합된 걸 표현.<br>
        $\mathcal{C}^{i}$는 Self-Attention 연산을 통해 추출해서 얻은 표현<br>
    <br>
    4. Attention 연산을 통해 추출해서 얻은 표현 $\mathcal{C}^{i}$를 $SwiGLU(\mathcal{C}^{i}W_1)W_2.$에 입력해 $\mathcal{V}^{i+1}$를 얻음<br>
        이때 $W_1$와 $W_2$는 학습가능한 파라메터<br>
        NOTE: (왜 $SwiGLU(\cdot)$을 사용했는지는 의문, 이건 논문을 읽어봐야 할 듯)<br>
    <br>
    5. 위 과정을 통해서 $I$로부터 시각적 특징을 추출하고 $K$개의 $Q$로 압축을 하게됨.<br>
    <br>
    6. 결과 $O((P + L)^2)$의 연산량에서 $O((K + L)^2)$로 시간이 줄어듬<br>

    위 과정을 거치면서 시각적인 특징이 압축된 Image 토큰 임베딩과 Text 토큰 임베딩을 concat한 뒤 Text Decoder에서 처리함.

### Modality-Adaptive Module

### Training Paradigm

## Experiments
