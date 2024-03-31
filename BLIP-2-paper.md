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

We propose BLIP-2, a new vision-language pre-training method that bootstraps from frozen pre-trained unimodal models. In order to bridge the modality gap, we propose a Querying Transformer (Q-Former) pre-trained in two stages: (1) vision-language representation learning stage with a frozen image encoder and (2) vision-to-language generative learning stage with a frozen LLM. This section first introduces the model architecture of Q-Former, and then delineates the two-stage pre-training procedures.

우리는 미리 훈련된 고정된 단일 모달 모델에서 부트스트랩하는 새로운 시각-언어 사전 훈련 방법인 BLIP-2를 제안합니다. 모달리티 간 격차를 해소하기 위해 (1) 고정 이미지 인코더를 사용한 비전-언어 표현 학습 단계와 (2) 고정 LLM을 사용한 비전-언어 생성 학습 단계의 두 단계로 사전 훈련된 쿼리 트랜스포머(Q-Former)를 제안합니다. 이 섹션에서는 먼저 Q-Former의 모델 아키텍처를 소개한 다음 2단계 사전 학습 절차에 대해 설명합니다.

### Model Architecture

We propose Q-Former as the trainable module to bridge the gap between a frozen image encoder and a frozen LLM. It extracts a fixed number of output features from the image encoder, independent of input image resolution. As shown in Figure 2, Q-Former consists of two transformer submodules that share the same self-attention layers: (1) an image transformer that interacts with the frozen image encoder for visual feature extraction, (2) a text transformer that can function as both a text encoder and a text decoder. We create a set number of learnable query embeddings as input to the image transformer. The queries interact with each other through self-attention layers, and interact with frozen image features through cross-attention layers (inserted every other transformer block). The queries can additionally interact with the text through the same self-attention layers. Depending on the pre-training task, we apply different self-attention masks to control query-text interaction. We initialize QFormer with the pre-trained weights of BERTbase (Devlin et al., 2019), whereas the cross-attention layers are randomly initialized. In total, Q-Former contains 188M parameters. Note that the queries are considered as model parameters.

In our experiments, we use 32 queries where each query has a dimension of 768 (same as the hidden dimension of the Q-Former). We use Z to denote the output query representation. The size of Z (32 × 768) is much smaller than the size of frozen image features (e.g. 257 × 1024 for ViT-L/14). This bottleneck architecture works together with our pre-training objectives into forcing the queries to extract visual information that is most relevant to the text.

우리는 고정 이미지 인코더와 고정 LLM 사이의 간극을 메우기 위한 훈련 가능한 모듈로 Q-Former를 제안합니다. 이 모듈은 입력 이미지 해상도와 관계없이 이미지 인코더에서 고정된 수의 출력 특징을 추출합니다. 그림 2에서 볼 수 있듯이 Q-Former는 (1) 시각적 특징 추출을 위해 고정 이미지 인코더와 상호 작용하는 이미지 트랜스포머, (2) 텍스트 인코더와 텍스트 디코더로 모두 작동할 수 있는 텍스트 트랜스포머 등 동일한 자체 주의 레이어를 공유하는 두 개의 트랜스포머 하위 모듈로 구성됩니다. 이미지 트랜스포머에 대한 입력으로 학습 가능한 쿼리 임베딩을 생성합니다. 쿼리는 자체 주의 레이어를 통해 서로 상호 작용하고 교차 주의 레이어(다른 트랜스포머 블록마다 삽입)를 통해 고정된 이미지 특징과 상호 작용합니다. 쿼리는 동일한 셀프 어텐션 레이어를 통해 텍스트와 추가로 상호 작용할 수 있습니다. 사전 학습 작업에 따라 쿼리-텍스트 상호 작용을 제어하기 위해 서로 다른 셀프 어텐션 마스크를 적용합니다. 사전 학습된 BERTbase의 가중치로 QFormer를 초기화하는 반면, 교차 주의 계층은 무작위로 초기화합니다(Devlin et al., 2019). Q-Former에는 총 1억 8천 8백만 개의 파라미터가 포함되어 있습니다. 쿼리는 모델 파라미터로 간주됩니다.

실험에서는 각 쿼리의 차원이 768인 32개의 쿼리를 사용했습니다(Q-Former의 숨겨진 차원과 동일). 출력 쿼리 표현을 나타내기 위해 Z를 사용합니다. Z의 크기(32 × 768)는 정지된 이미지 특징의 크기(예: ViT-L/14의 경우 257 × 1024)보다 훨씬 작습니다. 이 병목 현상 아키텍처는 사전 학습 목표와 함께 작동하여 쿼리가 텍스트와 가장 관련성이 높은 시각적 정보를 추출하도록 강제합니다.

### Bootstrap Vision-Language Representation Learning from a Frozen Image Encoder

In the representation learning stage, we connect Q-Former to a frozen image encoder and perform pre-training using image-text pairs. We aim to train the Q-Former such that the queries can learn to extract visual representation that is most informative of the text. Inspired by BLIP (Li et al., 2022), we jointly optimize three pre-training objectives that share the same input format and model parameters. Each objective employs a different attention masking strategy between queries and text to control their interaction (see Figure 2).

표현 학습 단계에서는 Q-Former를 고정 이미지 인코더에 연결하고 이미지-텍스트 쌍을 사용하여 사전 학습을 수행합니다. 쿼리가 텍스트에서 가장 많은 정보를 제공하는 시각적 표현을 추출하는 방법을 학습할 수 있도록 Q-Former를 훈련시키는 것을 목표로 합니다. BLIP(Li et al., 2022)에서 영감을 받아 동일한 입력 형식과 모델 파라미터를 공유하는 세 가지 사전 훈련 목표를 공동으로 최적화합니다. 각 목표는 쿼리와 텍스트 간에 서로 다른 주의 마스킹 전략을 사용하여 상호 작용을 제어합니다(그림 2 참조).

#### Image-Text Contrastive Learning

learns to align image representation and text representation such that their mutual information is maximized. It achieves so by contrasting the image-text similarity of a positive pair against those of negative pairs. We align the output query representation Z from the image transformer with the text representation t from the text transformer, where t is the output embedding of the [CLS] token. Since Z contains multiple output embeddings (one from each query), we first compute the pairwise similarity between each query output and t, and then select the highest one as the image-text similarity. To avoid information leak, we employ a unimodal self-attention mask, where the queries and text are not allowed to see each other. Due to the use of a frozen image encoder, we can fit more samples per GPU compared to end-to-end methods. Therefore, we use in-batch negatives instead of the momentum queue in BLIP.

는 이미지 표현과 텍스트 표현을 정렬하여 상호 정보를 최대화하도록 학습합니다. 이는 양의 쌍의 이미지와 텍스트 유사도를 음의 쌍의 유사도와 대조하여 이를 달성합니다. 이미지 트랜스포머의 출력 쿼리 표현 Z를 텍스트 트랜스포머의 텍스트 표현 t와 정렬합니다. 여기서 t는 [CLS] 토큰의 출력 임베딩입니다. Z에는 여러 개의 출력 임베딩(각 쿼리에서 하나씩)이 포함되어 있으므로 먼저 각 쿼리 출력과 t 사이의 쌍별 유사도를 계산한 다음 가장 높은 것을 이미지-텍스트 유사도로 선택합니다. 정보 유출을 방지하기 위해 쿼리와 텍스트가 서로를 볼 수 없는 단일 모달 자가 주의 마스크를 사용합니다. 고정 이미지 인코더를 사용하기 때문에 엔드투엔드 방식에 비해 GPU당 더 많은 샘플을 맞출 수 있습니다. 따라서 BLIP에서는 모멘텀 큐 대신 일괄 네거티브를 사용합니다.

#### Image-grounded Text Generation

loss trains the Q-Former to generate texts, given input images as the condition. Since the architecture of Q-Former does not allow direct interactions between the frozen image encoder and the text tokens, the information required for generating the text must be first extracted by the queries, and then passed to the text tokens via self-attention layers. Therefore, the queries are forced to extract visual features that capture all the information about the text. We employ a multimodal causal self-attention mask to control query-text interaction, similar to the one used in UniLM (Dong et al., 2019). The queries can attend to each other but not the text tokens. Each text token can attend to all queries and its previous text tokens. We also replace the [CLS] token with a new [DEC] token as the first text token to signal the decoding task.

손실은 입력 이미지가 조건으로 주어지면 텍스트를 생성하도록 Q-Former를 훈련시킵니다. Q-Former의 아키텍처는 고정 이미지 인코더와 텍스트 토큰 간의 직접적인 상호작용을 허용하지 않기 때문에 텍스트 생성에 필요한 정보는 쿼리에서 먼저 추출한 다음 자체 주의 레이어를 통해 텍스트 토큰으로 전달되어야 합니다. 따라서 쿼리는 텍스트에 대한 모든 정보를 캡처하는 시각적 특징을 강제로 추출해야 합니다. 우리는 쿼리-텍스트 상호작용을 제어하기 위해 멀티모달 인과적 자가 주의 마스크를 사용하며, 이는 UniLM에서 사용되는 것과 유사합니다(Dong et al., 2019). 쿼리는 서로에 대해 주의를 기울일 수 있지만 텍스트 토큰은 그렇지 않습니다. 각 텍스트 토큰은 모든 쿼리와 그 이전 텍스트 토큰에 응답할 수 있습니다. 또한 디코딩 작업을 알리는 첫 번째 텍스트 토큰으로 [CLS] 토큰을 새로운 [DEC] 토큰으로 대체합니다.

#### Image-Text Matching

aims to learn fine-grained alignment between image and text representation. It is a binary classification task where the model is asked to predict whether an image-text pair is positive (matched) or negative (unmatched). We use a bi-directional self-attention mask where all queries and texts can attend to each other. The output query embeddings Z thus capture multimodal information. We feed each output query embedding into a two-class linear classifier to obtain a logit, and average the logits across all queries as the output matching score. We adopt the hard negative mining strategy from Li et al. (2021; 2022) to create informative negative pairs.

은 이미지와 텍스트 표현 사이의 세밀한 정렬을 학습하는 것을 목표로 합니다. 이미지와 텍스트 쌍이 양수(일치)인지 음수(일치하지 않음)인지 예측하도록 모델에 요청하는 이진 분류 작업입니다. 모든 쿼리와 텍스트가 서로 주의할 수 있는 양방향 자기 주의 마스크를 사용합니다. 따라서 출력 쿼리 임베딩 Z는 멀티모달 정보를 캡처합니다. 각 출력 쿼리 임베딩을 2등급 선형 분류기에 공급하여 로짓을 얻고, 모든 쿼리의 로짓을 평균하여 출력 매칭 점수로 사용합니다. 저희는 Li 등(2021, 2022)의 하드 네거티브 마이닝 전략을 채택하여 유익한 네거티브 쌍을 생성합니다.

### Bootstrap Vision-to-Language Generative Learning from a Frozen LLM

In the generative pre-training stage, we connect QFormer (with the frozen image encoder attached) to a frozen LLM to harvest the LLM’s generative language capability. As shown in Figure 3, we use a fully-connected (FC) layer to linearly project the output query embeddings Z into the same dimension as the text embedding of the LLM. The projected query embeddings are then prepended to the input text embeddings. They function as soft visual prompts that condition the LLM on visual representation extracted by the Q-Former. Since the Q-Former has been pre-trained to extract language-informative visual representation, it effectively functions as an information bottleneck that feeds the most useful information to the LLM while removing irrelevant visual information. This reduces the burden of the LLM to learn vision-language alignment, thus mitigating the catastrophic forgetting problem.

We experiment with two types of LLMs: decoder-based LLMs and encoder-decoder-based LLMs. For decoderbased LLMs, we pre-train with the language modeling loss, where the frozen LLM is tasked to generate the text conditioned on the visual representation from Q-Former. For encoder-decoder-based LLMs, we pre-train with the prefix language modeling loss, where we split a text into two parts. The prefix text is concatenated with the visual representation as input to the LLM’s encoder. The suffix text is used as the generation target for the LLM’s decoder.

생성 사전 훈련 단계에서는 고정 이미지 인코더가 부착된 QFormer를 고정 LLM에 연결하여 LLM의 생성 언어 기능을 수집합니다. 그림 3에서 볼 수 있듯이, 완전 연결(FC) 레이어를 사용하여 출력 쿼리 임베딩 Z를 LLM의 텍스트 임베딩과 동일한 차원으로 선형적으로 투영합니다. 그런 다음 투영된 쿼리 임베딩이 입력 텍스트 임베딩에 추가됩니다. 이는 Q-Former가 추출한 시각적 표현에 따라 LLM을 조절하는 부드러운 시각적 프롬프트 역할을 합니다. Q-Former는 언어 정보를 제공하는 시각적 표현을 추출하도록 사전 학습되었기 때문에 관련 없는 시각 정보를 제거하면서 가장 유용한 정보를 LLM에 공급하는 정보 병목 현상 역할을 효과적으로 수행합니다. 이렇게 하면 시각과 언어의 정렬을 학습해야 하는 LLM의 부담이 줄어들어 치명적인 망각 문제를 완화할 수 있습니다.

저희는 디코더 기반 LLM과 인코더-디코더 기반 LLM의 두 가지 유형을 실험합니다. 디코더 기반 LLM의 경우, 언어 모델링 손실로 사전 훈련하여 고정된 LLM이 Q-Former의 시각적 표현에 따라 조건부 텍스트를 생성하는 임무를 수행합니다. 인코더-디코더 기반 LLM의 경우, 접두사 언어 모델링 손실로 사전 훈련하여 텍스트를 두 부분으로 분할합니다. 접두사 텍스트는 LLM의 인코더에 입력되는 시각적 표현과 연결됩니다. 접미사 텍스트는 LLM의 디코더를 위한 생성 대상으로 사용됩니다.

### Model Pre-training

#### Pre-training data

We use the same pre-training dataset as BLIP with 129M images in total, including COCO (Lin et al., 2014), Visual Genome (Krishna et al., 2017), CC3M (Sharma et al., 2018), CC12M (Changpinyo et al., 2021), SBU (Ordonez et al., 2011), and 115M images from the LAION400M dataset (Schuhmann et al., 2021). We adopt the CapFilt method (Li et al., 2022) to create synthetic captions for the web images. Specifically, we generate 10 captions using the BLIPlarge captioning model, and rank the synthetic captions along with the original web caption based on the image-text similarity produced by a CLIP ViT-L/14 model. We keep top-two captions per image as training data and randomly sample one at each pre-training step.

BLIP과 동일한 사전 학습 데이터셋을 사용하여 COCO(Lin et al., 2014), Visual Genome(Krishna et al., 2017), CC3M(Sharma et al., 2018), CC12M(Changpinyo et al., 2021), SBU(Ordonez et al., 2011), LAION400M 데이터셋의 115M 이미지를 포함한 총 129M 이미지로 학습을 진행합니다(Schuhmann et al., 2021). 웹 이미지에 대한 합성 캡션을 생성하기 위해 CapFilt 방법(Li et al., 2022)을 채택합니다. 구체적으로, BLIPlarge 캡션 모델을 사용하여 10개의 캡션을 생성하고, CLIP ViT-L/14 모델에서 생성된 이미지-텍스트 유사성을 기반으로 원본 웹 캡션과 함께 합성 캡션의 순위를 매깁니다. 이미지당 상위 2개의 캡션을 학습 데이터로 보관하고 각 사전 학습 단계에서 무작위로 하나를 샘플링합니다.

#### Pre-trained image encoder and LLM

For the frozen image encoder, we explore two state-of-the-art pre-trained vision transformer models: (1) ViT-L/14 from CLIP (Radford et al., 2021) and (2) ViT-g/14 from EVA-CLIP (Fang et al., 2022). We remove the last layer of the ViT and uses the second last layer’s output features, which leads to slightly better performance. For the frozen language model, we explore the unsupervised-trained OPT model family (Zhang et al., 2022) for decoder-based LLMs, and the instruction-trained FlanT5 model family (Chung et al., 2022) for encoder-decoder-based LLMs.

정지 이미지 인코더의 경우, 사전 학습된 두 가지 최신 비전 트랜스포머 모델인 (1) CLIP(Radford et al., 2021)의 ViT-L/14와 (2) EVA-CLIP(Fang et al., 2022)의 ViT-g/14를 살펴봅니다. ViT의 마지막 계층을 제거하고 두 번째 마지막 계층의 출력 기능을 사용하면 성능이 약간 향상됩니다. 프로즌 언어 모델의 경우, 디코더 기반 LLM의 경우 비지도 학습된 OPT 모델 제품군(Zhang et al., 2022)을, 인코더-디코더 기반 LLM의 경우 명령어 학습된 FlanT5 모델 제품군(Chung et al., 2022)을 살펴봅니다.

#### Pre-training settings

We pre-train for 250k steps in the first stage and 80k steps in the second stage. We use a batch size of 2320/1680 for ViT-L/ViT-g in the first stage and a batch size of 1920/1520 for OPT/FlanT5 in the second stage. During pre-training, we convert the frozen ViTs’ and LLMs’ parameters into FP16, except for FlanT5 where we use BFloat16. We found no performance degradation compared to using 32-bit models. Due to the use of frozen models, our pre-training is more computational friendly than existing large-scale VLP methods. For example, using a single 16-A100(40G) machine, our largest model with ViT-g and FlanT5-XXL requires less than 6 days for the first stage and less than 3 days for the second stage.

The same set of pre-training hyper-parameters are used for all models. We use the AdamW (Loshchilov & Hutter, 2017) optimizer with β1 = 0.9, β1 = 0.98, and a weight decay of 0.05. We use a cosine learning rate decay with a peak learning rate of 1e-4 and a linear warmup of 2k steps. The minimum learning rate at the second stage is 5e-5. We use images of size 224×224, augmented with random resized cropping and horizontal flipping.

1단계에서는 25만 걸음, 2단계에서는 8만 걸음에 대해 사전 훈련합니다. 첫 번째 단계에서는 ViT-L/ViT-g에 2320/1680의 배치 크기를, 두 번째 단계에서는 OPT/FlanT5에 1920/1520의 배치 크기를 사용합니다. 사전 훈련 중에 고정된 ViT 및 LLM의 파라미터를 FP16으로 변환하지만, FlanT5는 BFloat16을 사용합니다. 32비트 모델을 사용할 때와 비교했을 때 성능 저하를 발견하지 못했습니다. 프로즌 모델을 사용하기 때문에 사전 학습은 기존의 대규모 VLP 방식보다 계산 친화적입니다. 예를 들어, 단일 16-A100(40G) 머신을 사용하는 경우, ViT-g 및 FlanT5-XXL을 사용하는 가장 큰 모델은 1단계에 6일 미만, 2단계에 3일 미만이 소요됩니다.

모든 모델에 동일한 사전 훈련 하이퍼파라미터 세트가 사용됩니다. β1 = 0.9, β1 = 0.98, 가중치 감쇠 0.05의 AdamW(Loshchilov & Hutter, 2017) 옵티마이저를 사용합니다. 최대 학습률이 1e-4이고 선형 워밍업이 2k 단계인 코사인 학습률 감쇠를 사용합니다. 두 번째 단계의 최소 학습률은 5e-5입니다. 무작위 크기 조정 자르기 및 수평 뒤집기로 증강된 224×224 크기의 이미지를 사용합니다.

## Experiment

Table 1 provides an overview of the performance of BLIP-2 on various zero-shot vision-language tasks. Compared to previous state-of-the-art models, BLIP-2 achieves improved performance while requiring substantially fewer number of trainable parameters during vision-language pre-training.

표 1은 다양한 제로 샷 비전 언어 작업에서 BLIP-2의 성능에 대한 개요를 보여줍니다. 이전의 최신 모델에 비해 BLIP-2는 비전 언어 사전 훈련 시 필요한 훈련 가능한 파라미터 수가 훨씬 더 적으면서도 향상된 성능을 달성합니다.

### Instructed Zero-shot Image-to-Text Generation

BLIP-2 effectively enables a LLM to understand images while preserving its capability in following text prompts, which allows us to control image-to-text generation with instructions. We simply append the text prompt after the visual prompt as input to the LLM. Figure 4 shows examples to demonstrate a wide range of zero-shot image-to-text capabilities including visual knowledge reasoning, visual commensense reasoning, visual conversation, personalized image-to-text generation, etc.

는 시각적 프롬프트 뒤에 텍스트 프롬프트를 LLM에 입력으로 추가하기만 하면 됩니다. 그림 4는 시각적 지식 추론, 시각적 상식 추론, 시각적 대화, 개인화된 이미지-텍스트 생성 등 다양한 제로 샷 이미지-텍스트 기능을 보여주는 예시를 보여줍니다.

#### Zero-shot VQA

We perform quantitative evaluation on the zero-shot visual question answering task. For OPT models, we use the prompt “Question: {} Answer:”. For FlanT5 models, we use the prompt “Question: {} Short answer:”. During generation, we use beam search with a beam width of 5. We also set the length-penalty to -1 which encourages shorter answers that align better with human annotation. As shown in Table 2. BLIP-2 achieves state-of-the-art result on the VQAv2 (Goyal et al., 2017) and GQA (Hudson & Manning, 2019) datasets. It outperforms Flamingo80B by 8.7% on VQAv2, despite having 54x fewer trainable parameters. On the OK-VQA (Marino et al., 2019) dataset, BLIP-2 comes secondary to Flamingo80B. We hypothesis that this is because OK-VQA focuses more on open-world knowledge than visual understanding, and the 70B Chinchilla (Hoffmann et al., 2022) language model from Flamingo80B possesses more knowledge than the 11B FlanT5XXL.

We make a promising observation from Table 2: a stronger image encoder or a stronger LLM both lead to better performance. This observation is supported by several facts: (1) ViT-g outperforms ViT-L for both OPT and FlanT5. (2) Within the same LLM family, larger models outperform smaller ones. (3) FlanT5, an instruction-tuned LLM, outperforms the unsupervised-trained OPT on VQA. This observation validates BLIP-2 as a generic vision-language pre-training method that can efficiently harvest the rapid advances in vision and natural language communities.

제로 샷 시각적 문제 풀이 과제에 대해 정량적 평가를 실시합니다. OPT 모델의 경우 "질문: {} 답변:". FlanT5 모델의 경우 "Question: {} 짧은 답변:". 생성 중에는 빔 폭이 5인 빔 검색을 사용합니다. 또한 길이 페널티를 -1로 설정하여 사람의 주석과 더 잘 일치하는 짧은 답변을 장려합니다. 표 2에서 볼 수 있듯이. BLIP-2는 VQAv2(Goyal et al., 2017) 및 GQA(Hudson & Manning, 2019) 데이터 세트에서 최첨단 결과를 달성합니다. 훈련 가능한 파라미터가 54배 더 적음에도 불구하고 VQAv2에서 Flamingo80B보다 8.7% 더 뛰어난 성능을 보였습니다. OK-VQA(Marino et al., 2019) 데이터 세트에서는 BLIP-2가 Flamingo80B에 이어 2위를 차지했습니다. 이는 OK-VQA가 시각적 이해보다 오픈 월드 지식에 더 초점을 맞추고 있고, Flamingo80B의 70B 친칠라(Hoffmann et al., 2022) 언어 모델이 11B FlanT5XXL보다 더 많은 지식을 보유하고 있기 때문이라는 가설을 세웠습니다.

표 2에서 우리는 더 강력한 이미지 인코더나 더 강력한 LLM이 더 나은 성능으로 이어진다는 희망적인 관찰을 할 수 있습니다. 이 관찰은 몇 가지 사실에 의해 뒷받침됩니다. (1) ViT-g는 OPT와 FlanT5 모두에서 ViT-L보다 성능이 뛰어납니다. (2) 동일한 LLM 제품군 내에서는 더 큰 모델이 더 작은 모델보다 성능이 우수합니다. (3) 명령어 튜닝된 LLM인 FlanT5는 VQA에서 비지도 학습된 OPT보다 성능이 뛰어납니다. 이러한 관찰을 통해 BLIP-2는 비전 및 자연어 커뮤니티의 급속한 발전을 효율적으로 수집할 수 있는 일반적인 비전 언어 사전 훈련 방법이라는 것이 입증되었습니다.

#### Effect of Vision-Language Representation Learning

The first-stage representation learning pre-trains the QFormer to learn visual features relevant to the text, which reduces the burden of the LLM to learn vision-language alignment. Without the representation learning stage, Q-Former relies solely on the vision-to-language generative learning to bridge the modality gap, which is similar to the Perceiver Resampler in Flamingo. Figure 5 shows the effect of representation learning on generative learning. Without representation learning, both types of LLMs give substantially lower performance on zero-shot VQA. In particular, OPT suffers from catastrophic forgetting where performance drastically degrades as training proceeds.

1단계 표현 학습은 텍스트와 관련된 시각적 특징을 학습하도록 Q-Former를 사전 학습하여 시각-언어 정렬을 학습해야 하는 LLM의 부담을 줄여줍니다. 표현 학습 단계가 없는 Q-Former는 시각-언어 생성 학습에만 의존하여 양식 간 격차를 해소하며, 이는 플라밍고의 인식기 리샘플러와 유사합니다. 그림 5는 제너레이티브 학습에 대한 표현 학습의 효과를 보여줍니다. 표현 학습이 없으면 두 가지 유형의 LLM 모두 제로 샷 VQA에서 상당히 낮은 성능을 제공합니다. 특히 OPT는 훈련이 진행됨에 따라 성능이 급격히 저하되는 치명적 망각(catastrophic forgetting)을 겪습니다.

### Image Captioning

We finetune BLIP-2 models for the image captioning task, which asks the model to generate a text description for the image’s visual content. We use the prompt “a photo of” as an initial input to the LLM and trains the model to generate the caption with the language modeling loss. We keep the LLM frozen during finetuning, and updates the parameters of the Q-Former together with the image encoder. We experiment with ViT-g and various LLMs. Detailed hyperparameters can be found in the appendix. We perform finetuning on COCO, and evaluate on both COCO test set and zero-shot transfer to NoCaps (Agrawal et al., 2019) validation set.

The results are shown in Table 3. BLIP-2 achieves state-of-the-art performance with significant improvement on NoCaps over existing methods, demonstrating strong generalization ability to out-domain images.

이미지의 시각적 콘텐츠에 대한 텍스트 설명을 생성하도록 모델에 요청하는 이미지 캡션 작업을 위해 BLIP-2 모델을 세밀하게 조정합니다. LLM의 초기 입력으로 "사진"이라는 프롬프트를 사용하고 언어 모델링 손실이 있는 캡션을 생성하도록 모델을 학습시킵니다. 미세 조정 중에는 LLM을 고정된 상태로 유지하고 이미지 인코더와 함께 Q-Former의 매개변수를 업데이트합니다. ViT-g와 다양한 LLM으로 실험합니다. 자세한 하이퍼파라미터는 부록에서 확인할 수 있습니다. COCO에 대한 미세 조정을 수행하고, COCO 테스트 세트와 제로 샷 전송에서 NoCaps(Agrawal et al., 2019) 검증 세트에 대한 평가를 수행합니다.

결과는 표 3에 나와 있습니다. BLIP-2는 기존 방식에 비해 NoCaps에서 상당한 개선을 통해 최첨단 성능을 달성하여 이미지를 아웃도밍하는 강력한 일반화 능력을 입증했습니다.

### Visual Question Answering

Given annotated VQA data, we finetune the parameters of the Q-Former and the image encoder while keeping the LLM frozen. We finetune with the open-ended answer generation loss, where the LLM receives Q-Former’s output and the question as input, and is asked to generate the answer. In order to extract image features that are more relevant to the question, we additionally condition Q-Former on the question. Specifically, the question tokens are given as input to the Q-Former and interact with the queries via the self-attention layers, which can guide the Q-Former’s crossattention layers to focus on more informative image regions.

Following BLIP, our VQA data includes the training and validation splits from VQAv2, as well as training samples from Visual Genome. Table 4 demonstrates the state-of-theart results of BLIP-2 among open-ended generation models.

주석이 달린 VQA 데이터가 주어지면 LLM을 고정된 상태로 유지하면서 Q-Former와 이미지 인코더의 파라미터를 미세 조정합니다. LLM이 Q-Former의 출력과 질문을 입력으로 받아 답을 생성하도록 요청하는 개방형 답변 생성 손실로 미세 조정합니다. 질문과 더 관련성이 높은 이미지 특징을 추출하기 위해 Q-Former에 질문을 추가로 조건화합니다. 구체적으로, 질문 토큰은 Q-Former에 입력으로 제공되며 자체 주의 레이어를 통해 쿼리와 상호 작용하여 Q-Former의 교차 주의 레이어가 더 유익한 이미지 영역에 집중하도록 유도할 수 있습니다.

BLIP에 이어, VQA 데이터에는 VQAv2의 훈련 및 검증 분할과 Visual Genome의 훈련 샘플이 포함됩니다. 표 4는 개방형 생성 모델 중 BLIP-2의 최신 결과를 보여줍니다.

### Image-Text Retrieval

Since image-text retrieval does not involve language generation, we directly finetune the first-stage-pretrained model w/o LLM. Specifically, we finetune the image encoder together with Q-Former on COCO using the same objectives (i.e. ITC, ITM, and ITG) as pre-training. We then evaluate the model for both image-to-text retrieval and text-to-image retrieval on COCO and Flickr30K (Plummer et al., 2015) datasets. During inference, we follow Li et al. (2021; 2022) which first select k = 128 candidates based on the imagetext feature similarity, followed by a re-ranking based on pairwise ITM scores. We experiment with both ViT-L and ViT-g as the image encoder. Detailed hyperparameters can be found in the appendix.

The results are shown in Table 5. BLIP-2 achieves stateof-the-art performance with significant improvement over existing methods on zero-shot image-text retrieval.

The ITC and ITM losses are essential for image-text retrieval as they directly learn image-text similarity. In Table 6, we show that the ITG (image-grounded text generation) loss is also beneficial for image-text retrieval. This result supports our intuition in designing the representation learning objectives: the ITG loss enforces the queries to extract visual features most relevant to the text, thus improving visionlanguage alignment.

이미지-텍스트 검색에는 언어 생성이 포함되지 않기 때문에 LLM 없이 1단계 사전 학습된 모델을 직접 미세 조정합니다. 구체적으로는 사전 학습과 동일한 목표(예: ITC, ITM, ITG)를 사용하여 COCO에서 Q-Former와 함께 이미지 인코더를 미세 조정합니다. 그런 다음 이미지에서 텍스트로의 검색과 텍스트에서 이미지로의 검색 모두에 대한 모델을 COCO와 Flickr30K(Plummer et al., 2015) 데이터 세트에서 평가합니다. 추론 중에는 먼저 이미지-텍스트 특징 유사도에 따라 k = 128개의 후보를 선택한 다음 쌍별 ITM 점수를 기반으로 순위를 다시 매기는 Li 등(2021; 2022)의 방법을 따릅니다. 이미지 인코더로 ViT-L과 ViT-g를 모두 실험합니다. 자세한 하이퍼파라미터는 부록에서 확인할 수 있습니다.

결과는 표 5에 나와 있습니다. BLIP-2는 제로 샷 이미지-텍스트 검색에서 기존 방식보다 크게 개선된 최첨단 성능을 달성했습니다.

이미지-텍스트 유사도를 직접 학습하기 때문에 이미지-텍스트 검색에 있어 ITC와 ITM 손실은 필수적입니다. 표 6에서는 ITG(이미지 기반 텍스트 생성) 손실도 이미지-텍스트 검색에 도움이 된다는 것을 보여줍니다. 이 결과는 표현 학습 목표를 설계할 때의 직관을 뒷받침합니다. ITG 손실은 텍스트와 가장 관련성이 높은 시각적 특징을 추출하도록 쿼리를 강화하여 시각-언어 정렬을 개선합니다.

## Limitation

Recent LLMs can perform in-context learning given fewshot examples. However, our experiments with BLIP-2 do not observe an improved VQA performance when providing the LLM with in-context VQA examples. We attribute the lack of in-context learning capability to our pretraining dataset, which only contains a single image-text pair per sample. The LLMs cannot learn from it the correlation among multiple image-text pairs in a single sequence. The same observation is also reported in the Flamingo paper, which uses a close-sourced interleaved image and text dataset (M3W) with multiple image-text pairs per sequence. We aim to create a similar dataset in future work.

BLIP-2’s image-to-text generation could have unsatisfactory results due to various reasons including inaccurate knowledge from the LLM, activating the incorrect reasoning path, or not having up-to-date information about new image content (see Figure 7). Furthermore, due to the use of frozen models, BLIP-2 inherits the risks of LLMs, such as outputting offensive language, propagating social bias, or leaking private information. Remediation approaches include using instructions to guide model’s generation or training on a filtered dataset with harmful content removed.

최근의 LLM은 소수의 예제가 주어지면 컨텍스트 내 학습을 수행할 수 있습니다. 하지만, BLIP-2를 사용한 실험에서는 LLM에 컨텍스트 내 VQA 예제를 제공했을 때 향상된 VQA 성능이 관찰되지 않았습니다. 문맥 내 학습 기능이 부족한 이유는 샘플당 하나의 이미지-텍스트 쌍만 포함된 사전 학습 데이터 세트 때문이라고 생각합니다. LLM은 단일 시퀀스에 있는 여러 이미지-텍스트 쌍 간의 상관관계를 학습할 수 없습니다. 시퀀스당 여러 개의 이미지-텍스트 쌍이 포함된 클로즈 소스 인터리브 이미지 및 텍스트 데이터 세트(M3W)를 사용하는 Flamingo 논문에서도 동일한 관찰이 보고되었습니다. 향후 작업에서 이와 유사한 데이터 세트를 만드는 것을 목표로 하고 있습니다.

BLIP-2의 이미지-텍스트 생성은 LLM의 부정확한 지식, 잘못된 추론 경로 활성화, 새로운 이미지 콘텐츠에 대한 최신 정보 미비 등 다양한 이유로 인해 만족스럽지 못한 결과를 가져올 수 있습니다(그림 7 참조). 또한 고정 모델을 사용하기 때문에 BLIP-2는 불쾌한 언어 출력, 사회적 편견 전파, 개인 정보 유출과 같은 LLM의 위험성을 그대로 물려받습니다. 해결 방법으로는 모델 생성을 안내하는 지침을 사용하거나 유해한 콘텐츠가 제거된 필터링된 데이터 세트에서 학습하는 방법이 있습니다.

## Conclusion

We propose BLIP-2, a generic and compute-efficient method for vision-language pre-training that leverages frozen pretrained image encoders and LLMs. BLIP-2 achieves stateof-the-art performance on various vision-language tasks while having a small amount of trainable parameters during pre-training. BLIP-2 also demonstrates emerging capabilities in zero-shot instructed image-to-text generation. We consider BLIP-2 as an important step towards building a multimodal conversational AI agent.

유니티는 고정된 사전 훈련 이미지 인코더와 LLM을 활용하는 일반적이고 컴퓨팅 효율적인 비전 언어 사전 훈련 방법인 BLIP-2를 제안합니다. BLIP-2는 사전 훈련 중에 소량의 훈련 가능한 파라미터를 사용하면서도 다양한 비전 언어 작업에서 최첨단 성능을 달성합니다. 또한 BLIP-2는 제로 샷 인스트럭티드 이미지-텍스트 생성에서 새로운 기능을 보여줍니다. 유니티는 BLIP-2를 멀티모달 대화형 AI 에이전트 구축을 위한 중요한 단계로 간주합니다.
