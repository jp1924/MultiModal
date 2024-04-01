# mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration

## Abstract

### 기존 방식의 문제점

- Vision Encoder의 정보를 LLM에 맵핑하기 위해서 Q-former, Linear Projection 등을 사용해 Vision Encoder와 LLM간의 임베딩을 정렬 함.
하지만 제한된 수의 파라메터로 인해 맵핑이 불완전하게 일어나는 경우가 많아 성능이 떨어짐.

- MLLM의 SFT를 위해 Multimodal SFT 데이터와 일반 Text SFT 데이터를 혼합시켜 학습하는 경우가 있음.
하지만 multi modal을 SFT 하는 도중 일반 SFT를 시키면 당장에 Multi-modal의 성능은 개선되어서 일반 LLM의 성능이 떨어짐.
당연히 SFT 시키는 도중에 multi modal SFT를 시키면 기존 LLM이 영향을 받아서 기존 성능이 떨어지지 않나?

### 논문이 제안하는 방식

- Transformer Decoder 모델을 사용해 각 모달리티 간의 협업성을 늘리는 방법을 연구함.

## Introduction

- BLIP-2, InstructBLIP, Minigpt-4와 같이 Q-Former를 사용하거나 Shikra, LLaVA와 같이 Projection레이어를 통해 multi modal을 구현하는 방식이 주를 이루었음.
    하지만 이 방식을 사용하면 모달의 성능이 제한되는 문제가 있었음

- mPlug-Owl, LLava와 같은 모델들이 Multi-Modal SFT를 진행하는 하는 방법을 제안 했었음
    하지만 이 방식은 Multi Modality는 충족시킬 수 있더라도 일반 LLM에 영향을 끼쳐 성능에 영향을 준다는 것이 문제

- MLLM에서 각 모달별로 협력해 성능 적으로 이득을 볼 수 있지만 반대로 각 모달별로 간섭해 성능 적으로 손해를 볼 수 있다.
    때문에 mPlug-Owl2를 소개 함.

- mPlug-Owl2는 각 모달별 신호를 관리하기 위한 모듈로 Text Decoder를 추가해 각 모달별로 협력을 도모하게 만듭니다.
