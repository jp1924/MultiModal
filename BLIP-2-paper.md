# BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

## Abstract

The cost of vision-and-language pre-training has become increasingly prohibitive due to end-toend training of large-scale models. This paper proposes BLIP-2, a generic and efficient pretraining strategy that bootstraps vision-language pre-training from off-the-shelf frozen pre-trained image encoders and frozen large language models. BLIP-2 bridges the modality gap with a lightweight Querying Transformer, which is pretrained in two stages. The first stage bootstraps vision-language representation learning from a frozen image encoder. The second stage bootstraps vision-to-language generative learning from a frozen language model. BLIP-2 achieves state-of-the-art performance on various visionlanguage tasks, despite having significantly fewer trainable parameters than existing methods. For example, our model outperforms Flamingo80B by 8.7% on zero-shot VQAv2 with 54x fewer trainable parameters. We also demonstrate the model’s emerging capabilities of zero-shot image-to-text generation that can follow natural language instructions.

대규모 모델의 엔드투엔드 훈련으로 인해 비전 및 언어 사전 훈련 비용이 점점 더 비싸지고 있습니다. 이 백서에서는 기성품의 고정된 사전 훈련 이미지 인코더와 고정된 대규모 언어 모델에서 비전-언어 사전 훈련을 부트스트랩하는 일반적이고 효율적인 사전 훈련 전략인 BLIP-2를 제안합니다. BLIP-2는 두 단계로 사전 학습된 경량 쿼리 트랜스포머로 양식 간 격차를 해소합니다. 첫 번째 단계에서는 고정 이미지 인코더에서 비전-언어 표현 학습을 부트스트랩합니다. 두 번째 단계는 고정 언어 모델에서 비전-언어 생성 학습을 부트스트랩합니다. BLIP-2는 기존 방법보다 훨씬 적은 수의 훈련 가능한 매개변수에도 불구하고 다양한 비전 언어 작업에서 최첨단 성능을 달성합니다. 예를 들어, 이 모델은 훈련 가능한 파라미터가 54배 적은 제로 샷 VQAv2에서 Flamingo80B보다 8.7% 더 뛰어난 성능을 보였습니다. 또한 자연어 지시를 따를 수 있는 제로 샷 이미지-텍스트 생성이라는 모델의 새로운 기능도 시연합니다.

## Introduction

Vision-language pre-training (VLP) research has witnessed a rapid advancement in the past few years, where pre-trained models with increasingly larger scale have been developed to continuously push the state-of-the-art on various downstream tasks (Radford et al., 2021; Li et al., 2021; 2022; Wang et al., 2022a; Alayrac et al., 2022; Wang et al., 2022b). However, most state-of-the-art vision-language models incur a high computation cost during pre-training, due to end-to-end training using large-scale models and datasets

Vision-language research sits at the intersection between vision and language, therefore it is naturally expected that vision-language models can harvest from the readilyavailable unimodal models from the vision and natural language communities. In this paper, we propose a generic and compute-efficient VLP method by bootstrapping from offthe-shelf pre-trained vision models and language models.
Pre-trained vision models offer high-quality visual representation. Pre-trained language models, in particular large language models (LLMs), offer strong language generation and
zero-shot transfer abilities. To reduce computation cost and counteract the issue of catastrophic forgetting, the unimodal pre-trained models remain frozen during the pre-training.

In order to leverage pre-trained unimodal models for VLP, it is key to facilitate cross-modal alignment. However, since LLMs have not seen images during their unimodal pretraining, freezing them makes vision-language alignment in particular challenging. In this regard, existing methods (e.g. Frozen (Tsimpoukelli et al., 2021), Flamingo (Alayrac et al., 2022)) resort to an image-to-text generation loss, which we show is insufficient to bridge the modality gap.

To achieve effective vision-language alignment with frozen unimodal models, we propose a Querying Transformer (QFormer) pre-trained with a new two-stage pre-training strategy. As shown in Figure 1, Q-Former is a lightweight transformer which employs a set of learnable query vectors to extract visual features from the frozen image encoder. It acts as an information bottleneck between the frozen image encoder and the frozen LLM, where it feeds the most useful

visual feature for the LLM to output the desired text. In the first pre-training stage, we perform vision-language representation learning which enforces the Q-Former to learn visual representation most relevant to the text. In the second pre-training stage, we perform vision-to-language generative learning by connecting the output of the Q-Former to a frozen LLM, and trains the Q-Former such that its output visual representation can be interpreted by the LLM.

We name our VLP framework as BLIP-2: Bootstrapping Language-Image Pre-training with frozen unimodal models. The key advantages of BLIP-2 include:

• BLIP-2 effectively leverages both frozen pre-trained image models and language models. We bridge the modality gap using a Q-Former pre-trained in two-stages: representation learning stage and generative learning stage. BLIP-2 achieves state-of-the-art performance on various vision-language tasks including visual question answering, image captioning, and image-text retrieval

• Powered by LLMs (e.g. OPT (Zhang et al., 2022), FlanT5 (Chung et al., 2022)), BLIP-2 can be prompted to perform zero-shot image-to-text generation that follows natural language instructions, which enables emerging capabilities such as visual knowledge reasoning, visual conversation, etc. (see Figure 4 for examples).

• Due to the use of frozen unimodal models and a lightweight Q-Former, BLIP-2 is more compute-efficient than exisiting state-of-the-arts. For example, BLIP-2 outperforms Flamingo (Alayrac et al., 2022) by 8.7% on zero-shot VQAv2, while using 54× fewer trainable parameters. Furthermore, our results show that BLIP-2 is a generic method that can harvest more advanced unimodal models for better VLP performance.

지난 몇 년 동안 비전 언어 사전 훈련(VLP) 연구는 급속도로 발전해 왔으며, 점점 더 큰 규모의 사전 훈련 모델이 개발되어 다양한 다운스트림 작업에 대한 최신 기술을 지속적으로 추진하고 있습니다(Radford 외, 2021; Li 외, 2021; 2022; Wang 외, 2022a; Alayrac 외, 2022; Wang 외, 2022b). 그러나 대부분의 최신 비전 언어 모델은 대규모 모델과 데이터 세트를 사용하는 엔드 투 엔드 훈련으로 인해 사전 훈련 중에 높은 계산 비용이 발생합니다.

비전-언어 연구는 비전과 언어의 교차점에 위치하므로, 비전과 자연어 커뮤니티에서 쉽게 구할 수 있는 비모달 모델에서 비전-언어 모델을 수확할 수 있을 것으로 자연스럽게 기대됩니다. 이 백서에서는 기성품으로 미리 학습된 비전 모델과 언어 모델을 부트스트랩하여 일반적이고 계산 효율이 높은 VLP 방법을 제안합니다.
사전 학습된 비전 모델은 고품질의 시각적 표현을 제공합니다. 사전 학습된 언어 모델, 특히 대규모 언어 모델(LLM)은 강력한 언어 생성 및
제로 샷 전송 기능을 제공합니다. 계산 비용을 줄이고 치명적인 망각 문제에 대응하기 위해 사전 훈련된 단일 모드 사전 훈련 모델은 사전 훈련 중에 고정된 상태로 유지됩니다.

사전 학습된 유니모달 모델을 VLP에 활용하려면 모드 간 정렬을 용이하게 하는 것이 핵심입니다. 그러나 LLM은 유니모달 사전 훈련 중에 이미지를 본 적이 없기 때문에 이미지를 고정하면 시각 언어 정렬이 특히 어려워집니다. 이와 관련하여 기존 방법(예: Frozen(Tsimpoukelli 외, 2021), Flamingo(Alayrac 외, 2022))은 이미지-텍스트 생성 손실에 의존하는데, 이는 모달리티 간 격차를 해소하기에 충분하지 않다는 것을 보여줍니다.

고정된 유니모달 모델에서 효과적인 시각-언어 정렬을 달성하기 위해 새로운 2단계 사전 학습 전략으로 사전 학습된 쿼리 트랜스포머(QFormer)를 제안합니다. 그림 1에서 볼 수 있듯이 Q-Former는 학습 가능한 쿼리 벡터 세트를 사용하여 정지된 이미지 인코더에서 시각적 특징을 추출하는 경량 트랜스포머입니다. 이 변환기는 프로즌 이미지 인코더와 프로즌 LLM 사이의 정보 병목 현상 역할을 하여 가장 유용한

시각적 특징을 LLM이 원하는 텍스트를 출력할 수 있도록 제공합니다. 첫 번째 사전 훈련 단계에서는 시각 언어 표현 학습을 수행하여 Q-Former가 텍스트와 가장 관련성이 높은 시각적 표현을 학습하도록 합니다. 두 번째 사전 학습 단계에서는 Q-Former의 출력을 고정된 LLM에 연결하여 비전-언어 생성 학습을 수행하고, 출력된 시각적 표현을 LLM이 해석할 수 있도록 Q-Former를 학습시킵니다.

우리는 VLP 프레임워크의 이름을 BLIP-2: 고정된 유니모달 모델을 사용한 부트스트랩 언어-이미지 사전 학습이라고 명명했습니다. BLIP-2의 주요 장점은 다음과 같습니다:

- BLIP-2는 고정된 사전 학습 이미지 모델과 언어 모델을 모두 효과적으로 활용합니다. 표현 학습 단계와 생성 학습 단계의 두 단계로 사전 학습된 Q-Former를 사용하여 모달리티 간 격차를 해소합니다. BLIP-2는 시각적 질문 답변, 이미지 캡션, 이미지-텍스트 검색 등 다양한 시각 언어 작업에서 최첨단 성능을 달성합니다.

- LLM(예: OPT(Zhang et al., 2022), FlanT5(Chung et al., 2022))을 기반으로 하는 BLIP-2는 자연어 지시에 따라 제로 샷 이미지-텍스트 생성을 수행할 수 있어 시각 지식 추론, 시각 대화 등과 같은 새로운 기능을 지원합니다(예는 그림 4 참조).

- 고정된 유니모달 모델과 경량 Q-Former를 사용하기 때문에 BLIP-2는 기존 최신 기술보다 컴퓨팅 효율성이 뛰어납니다. 예를 들어, BLIP-2는 제로 샷 VQAv2에서 훈련 가능한 파라미터를 54배 더 적게 사용하면서 플라밍고(Alayrac et al., 2022)보다 8.7% 더 뛰어난 성능을 보였습니다. 또한, 연구 결과에 따르면 BLIP-2는 더 나은 VLP 성능을 위해 고급 유니모달 모델을 수집할 수 있는 일반적인 방법이기도 합니다.

Translated with DeepL.com (free version)

## Related Work

### End-to-end Vision-Language Pre-training

Vision-language pre-training aims to learn multimodal foundation models with improved performance on various visionand-language tasks. Depending on the downstream task, different model architectures have been proposed, including the dual-encoder architecture (Radford et al., 2021; Jia et al., 2021), the fusion-encoder architecture (Tan & Bansal, 2019; Li et al., 2021), the encoder-decoder architecture (Cho et al., 2021; Wang et al., 2021b; Chen et al., 2022b), and more recently, the unified transformer architecture (Li et al., 2022; Wang et al., 2022b). Various pre-training objectives have also been proposed over the years, and have progressively converged to a few time-tested ones: image-text contrastive learning (Radford et al., 2021; Yao et al., 2022; Li et al., 2021; 2022), image-text matching (Li et al., 2021; 2022; Wang et al., 2021a), and (masked) language modeling (Li et al., 2021; 2022; Yu et al., 2022; Wang et al., 2022b). Most VLP methods perform end-to-end pre-training using large-scale image-text pair datasets. As the model size keeps increasing, the pre-training can incur an extremely high computation cost. Moreover, it is inflexible for end-to-end pre-trained models to leverage readily-available unimodal pre-trained models, such as LLMs (Brown et al., 2020; Zhang et al., 2022; Chung et al., 2022).

비전-언어 사전 학습은 다양한 비전 및 언어 작업에서 향상된 성능으로 멀티모달 기반 모델을 학습하는 것을 목표로 합니다. 다운스트림 작업에 따라 듀얼 인코더 아키텍처(Radford et al., 2021; Jia et al., 2021), 퓨전 인코더 아키텍처(Tan & Bansal, 2019; Li et al, 2021), 인코더-디코더 아키텍처(Cho et al., 2021; 왕 등, 2021b; 첸 등, 2022b), 최근에는 통합 트랜스포머 아키텍처(Li 등, 2022; 왕 등, 2022b)로 발전하고 있습니다. 또한 수년에 걸쳐 다양한 사전 학습 목표가 제안되어 왔으며, 이미지-텍스트 대조 학습(Radford 외, 2021; Yao 외, 2022; Li 외, 2021; 2022), 이미지-텍스트 매칭(Li 외, 2021; 2022; Wang 외, 2021a), (마스크된) 언어 모델링(Li 외, 2021; 2022; Yu 외, 2022; Wang 외, 2022b) 등 오랜 테스트를 거친 몇 가지 목표로 점차 수렴되어 가고 있습니다. 대부분의 VLP 방법은 대규모 이미지-텍스트 쌍 데이터 세트를 사용하여 엔드투엔드 사전 학습을 수행합니다. 모델 크기가 계속 증가함에 따라 사전 학습에는 매우 높은 계산 비용이 발생할 수 있습니다. 또한 엔드투엔드 사전 학습 모델은 LLM과 같이 쉽게 구할 수 있는 비모달 사전 학습 모델을 활용하는 것이 유연하지 않습니다(Brown et al., 2020; Zhang et al., 2022; Chung et al., 2022).

Translated with DeepL.com (free version)

### Modular Vision-Language Pre-training

More similar to us are methods that leverage off-the-shelf pre-trained models and keep them frozen during VLP. Some methods freeze the image encoder, including the early work which adopts a frozen object detector to extract visual features (Chen et al., 2020; Li et al., 2020; Zhang et al., 2021), and the recent LiT (Zhai et al., 2022) which uses a frozen pre-trained image encoder for CLIP (Radford et al., 2021) pre-training. Some methods freeze the language model to use the knowledge from LLMs for vision-to-language generation tasks (Tsimpoukelli et al., 2021; Alayrac et al., 2022; Chen et al., 2022a; Manas et al. ˜ , 2023; Tiong et al., 2022; Guo et al., 2022). The key challenge in using a frozen LLM is to align visual features to the text space. To achieve this, Frozen (Tsimpoukelli et al., 2021) finetunes an image encoder whose outputs are directly used as soft prompts for the LLM. Flamingo (Alayrac et al., 2022) inserts new cross-attention layers into the LLM to inject visual features, and pre-trains the new layers on billions of image-text pairs. Both methods adopt the language modeling loss, where the language model generates texts conditioned on the image.

Different from existing methods, BLIP-2 can effectively and efficiently leverage both frozen image encoders and frozen LLMs for various vision-language tasks, achieving stronger performance at a lower computation cost.

우리와 더 유사한 방법은 기성품으로 미리 훈련된 모델을 활용하고 VLP 중에 고정된 상태로 유지하는 방법입니다. 이미지 인코더를 고정하여 시각적 특징을 추출하기 위해 고정된 물체 감지기를 채택한 초기 작업(Chen et al., 2020; Li et al., 2020; Zhang et al., 2021)과 CLIP(Radford et al., 2021) 사전 학습을 위해 고정된 사전 훈련된 이미지 인코더를 사용하는 최근 LiT(Zhai et al., 2022)를 포함하여 일부 방법에서는 이미지 인코더를 고정합니다. 일부 방법은 언어 모델을 고정하여 비전-언어 생성 작업에 LLM의 지식을 사용합니다(Tsimpoukelli 외., 2021; Alayrac 외., 2022; Chen 외., 2022a; Manas 외., 2023; Tiong 외., 2022; Guo 외., 2022). 프로즌 LLM 사용의 핵심 과제는 시각적 특징을 텍스트 공간에 정렬하는 것입니다. 이를 위해 Frozen(Tsimpoukelli et al., 2021)은 이미지 인코더를 미세 조정하여 출력을 LLM의 소프트 프롬프트로 직접 사용합니다. Flamingo(Alayrac 외., 2022)는 시각적 특징을 주입하기 위해 새로운 교차 주의 레이어를 LLM에 삽입하고 수십억 개의 이미지-텍스트 쌍에 대해 새 레이어를 사전 학습합니다. 두 방법 모두 언어 모델링 손실을 채택하여 언어 모델이 이미지에 조건부 텍스트를 생성합니다.

기존 방식과 달리 BLIP-2는 다양한 시각 언어 작업에 고정 이미지 인코더와 고정 LLM을 모두 효과적이고 효율적으로 활용하여 더 낮은 계산 비용으로 더 강력한 성능을 달성할 수 있습니다.

## Method

### Model Architecture

### Bootstrap Vision-Language Representation Learning from a Frozen Image Encoder

#### Image-Text Contrastive Learning

#### Image-grounded Text Generation

#### Image-Text Matching

### Bootstrap Vision-to-Language Generative Learning from a Frozen LLM

### Model Pre-training

#### Pre-training data

#### Pre-trained image encoder and LLM

#### Pre-training settings

## Experiment

### Instructed Zero-shot Image-to-Text Generation

#### Zero-shot VQA

#### Effect of Vision-Language Representation Learning

### Image Captioning

### Visual Question Answering

### Image-Text Retrieval

## Limitation

## Conclusion
