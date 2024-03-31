# BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

## Abstract

### 기존 방식의 문제점

### 논문이 제안하는 방식

## Introduction

- OFA, M3AE, BEIT-3, ALBEF 등 CLIP과 같은 Vision 신호와 Text 신호를 학습에 사용하는 멀티 모달이 많이 등장하고 있다.
하지만 end-to-end 사전학습을 위해 많은 양의 자원을 필요로 한다.

- 적은 자원으로 모달을 학습시키기 위해서 우린 사전학습이 되어 있는 Vision, Langauge 모델을 사용 함.
다만 자원 절약과 기존에 학습한 것을 망각하는 것을 방지 하기 위해 각 모달은 얼린 상태로 학습 함.

// TODO: 정리, 기존엔 image-to-text generation loss에 의존해 각 모달간의 신호 차를 줄여 왔지만 이 방법엔 한계가 있음.

- 하지만 각 모달을 얼린 상태로 학습하면 각 모달별로 신호를 정렬하는 것이 어려움.  
    대안으로 Vision, Langauge 신호를 서로 정렬하기 위한 2-stage 학습 기법과 Q-Former를 제안 함.
    1. 1 step으로 입력된 Vision 신호를 Langauge 모델이 이해할 수 있게 Q-Former가 모달간 신호를 정렬하는 과정
    2. 2 step으로

// Q-Former내의 학습 가능한 QKV 백터를 이용해 얼려진 Vision 모델의 신호를 추출 함.

## Related Work
