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
