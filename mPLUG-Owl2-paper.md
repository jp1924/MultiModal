# mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration

## Abstract

Multi-modal Large Language Models (MLLMs) have demonstrated impressive instruction abilities across various open-ended tasks. However, previous methods primarily focus on enhancing multi-modal capabilities. In this work, we introduce a versatile multi-modal large language model, mPLUG-Owl2, which effectively leverages modality collaboration to improve performance in both text and multi-modal tasks. mPLUG-Owl2 utilizes a modularized network design, with the language decoder acting as a universal interface for managing different modalities. Specifically, mPLUG-Owl2 incorporates shared functional modules to facilitate modality collaboration and introduces a modality-adaptive module that preserves modality-specific features. Extensive experiments reveal that mPLUG-Owl2 is capable of generalizing both text tasks and multi-modal tasks and achieving state-ofthe-art performances with a single generic model. Notably, mPLUG-Owl2 is the first MLLM model that demonstrates the modality collaboration phenomenon in both pure-text and multi-modal scenarios, setting a pioneering path in the development of future multi-modal foundation models.

다중 모드 대규모 언어 모델(MLLM)은 다양한 개방형 작업에서 인상적인 인스트럭션 능력을 보여주었습니다. 그러나 이전의 방법은 주로 다중 모달 기능을 향상시키는 데 중점을 두었습니다. 이번 연구에서는 다목적 멀티 모달 대형 언어 모델인 mPLUG-Owl2를 소개하며, 이는 모달 협업을 효과적으로 활용하여 텍스트 및 멀티 모달 작업 모두에서 성능을 향상시킵니다. mPLUG-Owl2는 모듈화된 네트워크 설계를 활용하며 언어 디코더가 다양한 모달을 관리하기 위한 범용 인터페이스 역할을 수행합니다. 특히 mPLUG-Owl2는 공유 기능 모듈을 통합하여 모달리티 협업을 용이하게 하고 모달리티별 기능을 보존하는 모달리티 적응형 모듈을 도입했습니다. 광범위한 실험을 통해 mPLUG-Owl2는 텍스트 작업과 멀티 모달 작업을 모두 일반화할 수 있으며 단일 일반 모델로 최첨단 성능을 달성할 수 있음이 밝혀졌습니다. 특히 mPLUG-Owl2는 순수 텍스트 및 멀티모달 시나리오 모두에서 모달리티 협업 현상을 입증한 최초의 MLLM 모델로, 향후 멀티모달 기반 모델 개발의 선구적인 길을 제시합니다.

## Introduction

Large Language Models (LLMs) such as GPT-3 [6], LLaMA [57, 58], and GPT-4 [46] have garnered significant attention due to their exceptional generalization abilities in text understanding and generation. To facilitate the visionlanguage applications, GPT-4V1 [45] has recently demonstrated impressive multi-modal capabilities in diverse tasks, e.g., description , question answering, etc., sparking interest among researchers in the potential convergence of the vision-language field. This has led to the emergence of a group of Multi-modal Large Language Models (MLLMs) [5, 15, 31, 38, 65, 66, 68, 75], which aim to enhance LLMs with the ability to understand and handle visual problems.

Previous studies [27, 63] in multi-modal learning suggest that different modalities can effectively collaborate, thereby enhancing the performance of both text and multi-modal tasks simultaneously. However, MLLMs is a unified model that supports different modalities and tasks without finetuning for specific tasks. Recent works utilize cross-modal alignment modules (e.g., Q-former [15, 31, 75] and linear layer [10, 38]) to map visual features from the vision encoder into the frozen LLMs to carry out multi-modal tasks by leveraging preserved language capabilities. This strategy, unfortunately, restricts the potential of modality collaboration. As a result, some researchers [38, 68] opt to fine-tune LLMs during multi-modal instruction tuning. While fine-tuning significantly improves multi-modal tasks, it risks weakening text task performance [16]. As illustrated in Figure 1, the challenge of modality collaboration in MLLMs is from applying a single module to balance the gain of modality collaboration and modality interference, where modalities may interfere with each other on a large number of instruction datasets across multiple modalities.

To mitigate this challenge, we present a new generalpurpose multi-modal foundation model, mPLUG-Owl2, in this work. Our model features a modularized network design that takes both modality collaboration and modality interference into account, using the language decoder as a universal interface for managing multi-modal signals. Specifically, mPLUG-Owl2 incorporates certain shared functional modules to promote modality collaboration and introduces a modality-adaptive module that serves as a pivot across different modalities. Therefore, vision and language modalities are projected into a shared semantic space for crossmodality interaction, while the proposed module helps preserve modality-specific features. With our novel architecture, modalities with varying information densities are shielded from modality interference due to the modalityadaptive module and can collaborate effectively in capturing shared information. Furthermore, we introduce an innovative two-stage training paradigm that consists of visionlanguage pre-training and joint vision-language instruction tuning. This paradigm trains the vision encoder across two stages, enabling it to capture both low-level and high-level semantic visual information more effectively.

Extensive experiments illustrate the effectiveness and generalization abilities of mPLUG-Owl2, which achieves state-of-the-art performance on 8 classic vision-language benchmarks using a single generic model. Furthermore, it either first or second in performance on 5 recent zeroshot multi-modal benchmarks, underscoring its adaptability and proficiency in multi-modal instruction comprehension and generation. In addition to its cutting-edge performance in multi-modal tasks, mPLUG-Owl2 also achieves state-ofthe-art results on multiple pure-text benchmarks. Moreover, we provide in-depth analysis to demonstrate and validate the impact of modality collaboration through our proposed modality-adaptive module, especially in enhancing text tasks, including understanding, knowledge, and reasoning. Finally, comprehensive ablation studies validate the effectiveness of the proposed MLLM training paradigm, which can help inspire the development of future multimodal foundation models.

GPT-3 [6], LLaMA [57, 58], GPT-4 [46]와 같은 대규모 언어 모델(LLM)은 텍스트 이해 및 생성에서 뛰어난 일반화 능력으로 인해 큰 주목을 받고 있습니다. 비전 언어 응용을 촉진하기 위해 최근 GPT-4V1 [45]은 설명, 질문 답변 등 다양한 작업에서 인상적인 다중 모드 기능을 입증하여 비전 언어 분야의 잠재적 융합에 대한 연구자들의 관심을 불러 일으켰습니다. 이로 인해 시각 문제를 이해하고 처리하는 능력으로 LLM을 향상시키는 것을 목표로 하는 다중 모드 대규모 언어 모델(MLLM) 그룹[5, 15, 31, 38, 65, 66, 68, 75]이 등장하게 되었습니다[5, 15, 31, 38, 65, 66, 68, 75].

다중 모달 학습에 대한 이전 연구[27, 63]에 따르면 서로 다른 모달이 효과적으로 협업하여 텍스트 작업과 다중 모달 작업의 성능을 동시에 향상시킬 수 있다고 합니다. 그러나 MLLM은 특정 작업에 대한 미세 조정 없이 다양한 모달리티와 작업을 지원하는 통합 모델입니다. 최근 연구에서는 크로스 모달 정렬 모듈(예: Q-former [15, 31, 75] 및 리니어 레이어 [10, 38])을 사용하여 비전 인코더의 시각적 특징을 고정된 LLM에 매핑하여 보존된 언어 기능을 활용하여 멀티 모달 작업을 수행합니다. 안타깝게도 이 전략은 모달리티 협업의 잠재력을 제한합니다. 그 결과 일부 연구자[38, 68]는 멀티 모달 인스트럭션 튜닝 중에 LLM을 미세 조정하는 방법을 선택하기도 합니다. 미세 조정은 멀티 모달 작업을 크게 개선하지만 텍스트 작업 성능을 약화시킬 위험이 있습니다[16]. 그림 1에서 볼 수 있듯이 MLLM에서 모달리티 협업의 과제는 단일 모듈을 적용하여 모달리티 협업의 이득과 모달리티 간섭의 균형을 맞추는 것인데, 여러 모달리티의 수많은 인스트럭션 데이터세트에서 모달리티가 서로 간섭할 수 있기 때문입니다.

이러한 문제를 완화하기 위해 이번 연구에서는 새로운 범용 멀티 모달 기반 모델인 mPLUG-Owl2를 소개합니다. 이 모델은 멀티 모달 신호 관리를 위한 범용 인터페이스로 언어 디코더를 사용하여 모달 협업과 모달 간섭을 모두 고려하는 모듈화된 네트워크 설계를 특징으로 합니다. 특히 mPLUG-Owl2는 특정 공유 기능 모듈을 통합하여 모달리티 협업을 촉진하고 여러 모달리티에 걸쳐 피벗 역할을 하는 모달리티 적응형 모듈을 도입합니다. 따라서 시각 및 언어 모달리티는 교차 모달리티 상호 작용을 위해 공유 의미 공간에 투영되며, 제안된 모듈은 모달리티별 기능을 보존하는 데 도움이 됩니다. 새로운 아키텍처를 통해 다양한 정보 밀도를 가진 모달리티는 모달리티 적응형 모듈로 인해 모달리티 간섭으로부터 보호되며 공유 정보를 효과적으로 캡처하여 협업할 수 있습니다. 또한 시각 언어 사전 교육과 공동 시각 언어 교육 튜닝으로 구성된 혁신적인 2단계 교육 패러다임을 도입했습니다. 이 패러다임은 두 단계에 걸쳐 비전 인코더를 훈련시켜 저수준 및 고수준 의미론적 시각 정보를 보다 효과적으로 캡처할 수 있도록 합니다.

광범위한 실험을 통해 단일 일반 모델을 사용하여 8개의 고전적인 시각 언어 벤치마크에서 최첨단 성능을 달성한 mPLUG-Owl2의 효과와 일반화 능력을 확인할 수 있습니다. 또한 최근 5개의 제로샷 멀티모달 벤치마크에서 1위 또는 2위를 차지하여 멀티모달 명령어 이해 및 생성에 대한 적응성과 숙련도를 입증했습니다. 멀티 모달 작업에서의 최첨단 성능 외에도 mPLUG-Owl2는 여러 순수 텍스트 벤치마크에서도 최첨단 결과를 달성했습니다. 또한, 특히 이해, 지식, 추론 등 텍스트 작업을 향상시키는 데 있어 제안된 모달리티 적응형 모듈을 통해 모달리티 협업의 영향을 입증하고 검증하기 위한 심층 분석을 제공합니다. 마지막으로, 포괄적인 절제 연구를 통해 제안된 MLLM 교육 패러다임의 효과를 검증하여 향후 멀티모달 기초 모델 개발에 영감을 줄 수 있습니다.

## Methodology

### Overview

Figure 2 (a) sketches the overview of the mPLUG-Owl2.
Specifically, our model comprises a vision encoder, a visual
abstractor, a text embedding layer, and a language decoder.
Notably, the standard implementation of the text embedding layer and language decoder involves the use of a large
language model, such as GPT [6] or LLaMA [57]. We
first briefly introduce our model’s architecture in Section
3.2. Furthermore, we handle different types of modalities by introducing the modality-adaptive module in Section
3.3. Lastly, we introduce the training paradigm for training
mPLUG-Owl2 with modality collaboration in Section 3.4.

### Model Architecture

As depicted in Figure 2, our model, referred to as mPLUGOwl2, is composed of three main components: a fundamental vision encoder [48], a visual abstractor, and a language decoder. Specifically, we utilize ViT-L/14 as the
vision encoder and LLaMA-2-7B [58] as the language decoder. The vision encoder processes an input image with
an H × W resolution and produces a sequence of H
14 ×
W
14
tokens. These visual token features are then combined with
text token embeddings and fed into the language decoder
that serves as a universal interface that converts various
vision-language tasks into text-generation tasks. However,
with the increase in image resolution, the encoded visual
token sequences can exponentially lengthen. Additionally,
the presence of abundant redundancy in the images (e.g.,
background, similar patches) leads to computational waste
and introduces considerable noise. To address this, we
propose a visual abstractor equipped with a fixed set of
learnable queries to extract higher semantic features from
images. Specifically, we feed the extracted visual token sequence I = [I1, I2, · · · , IP ] ∈ R
P ×d
and a fixed number of
K learnable queries Q ∈ R
K×d
into the visual abstractor.
Here, P =
H
14 ×
W
14 represents the number of visual patches, and D is the hidden dimension. The visual abstractor consists of a series of visual abstractor layers. In the i-th layer of
the visual abstractor, the compressed visual representations
V
i+1 are computed as follows:

(1)
(2)

Here, Attn(·, ·, ·) represents the self-attention operation,
while W1 ∈ R
d×d
′
and W2 ∈ R
d
′×d
are learnable parameters. The function SwiGLU(· · ·) refers to the SwiGLU
activation function [51]. We designate V
0 = Q to initiate
the process. Moreover, to augment the fine-grained perception ability, we integrate sinusoidal positional embeddings
with the image feature I and V
i
, thereby preserving positional information, which has been proven essential in [8].
Hence, the computation required by the language decoder
decreases from O((P + L)
2
) to O((K + L)
2
), significantly
reducing computational load when P ≫ K, particularly
in scenarios involving multiple images and when the text
length L is relatively short. Once the compressed visual
feature is obtained, it is concatenated with text token embeddings and then processed by the language decoder to
generate the prediction.

### Modality-Adaptive Module

Prior approaches [15, 38, 68, 75] typically attempt to align
visual features with language features by projecting image
features into the language semantic space. However, this
strategy can cause a mismatch in granularity , where image
features often contain fruitful semantic information compared to the discrete semantic information within text embedding features. Those methods disregard the unique characteristics of visual and textual information, thus potentially
limiting the model’s performance. To this end, we propose
a new approach, namely, the Modality-Adaptive Module
(MAM), which decouples vision-language representations
by projecting visual features and language features into a
shared semantic space while preserving the distinctive properties of each modality.

Formally, given a vision-language sequence X ∈
R
(LV +LT )×d
and modality indicators M ∈ {0, 1}
(Lv+LT )
,
we first define modality separated operation ϕ as:

(3)

where m ∈ {0, 1} is the type of modalities (i.e., vision
or language). Given the previous layer’s output vectors
Hl−1, l ∈ [1, L], where L is the number of language decoder
layers, we first normalized different modalities into the same
magnitude as follows:

(4)

where LNV and LNT are layer normalization [4] for visual
features and language features respectively. Then, we reformulate the self-attention operation by leveraging separated
linear projection layers for key projection matrix and value
projection matrix while preserving query projection matrix
shared as follows:

(5)
(6)
(7)
(8)

where W
Q
l
, W K0
l
, W K1
l
, WV0
l
, WV1
l ∈ R
d×d
are the learnable projection matrices, and Cl ∈ R
(LV +LT )×d
is the context features of l-th layer. In this manner, we can calculate
the similarities between these two modalities within a shared
semantic space, while also preserving the unique characteristics of each modality through different value projection
layers. Moreover, by decoupling the key and value projection matrix, we can avoid interference between the two
modalities, particularly in relation to granularity mismatch.
In a similar vein, we also aim to model these characteristics by using different layer normalization layers. Finally,
in order to promote modality collaboration within the same
feature space, we maintain a shared FFN for both modalities.
As a consequence, the model is able to preserve modality characteristics while achieving modality collaboration via
the proposed modality-adaptive module.

### Training Paradigm

As depicted in Figure 2 (c), we employ a two-stage approach
in training mPLUG-Owl2, comprising pre-training and visual instruction tuning similar to [38, 68], which aims to
align the pre-trained vision encoder and language model during the pre-training phase, and then fine-tune the language
model with language modeling loss during the instruction
tuning phase. However, we find that simply freezing a pretrained vision encoder and training a vision-language projector to align visual data with language models can limit
their capacity to interpret complex visual information, such
as scene text and visual knowledge. To address the issue,
we make the vision encoder trainable throughout both the
pre-training and instruction tuning stages. This strategy
allows the model to capture both low-level and high-level
semantic visual information more effectively. Specifically,
for the pre-training stage, we enable the vision encoder,
visual abstractor, and a part of the modality-adaptive module to be trainable, while keeping the pre-trained language
model frozen. Meanwhile, prior research in multi-modal
learning [63] has indicated that significant enhancements
can be achieved through the collaborative learning of unimodal and multi-modal sources. Based on this, we adopt
a joint training approach by tuning the whole model during the instruction tuning stage, incorporating both text and multi-modal instructions. This methodology enhances the
model’s comprehension of visual concepts embedded within
the text by the multi-modal instructions. Concurrently, the
text instruction data augments the model’s understanding of
intricate natural instructions, thereby ensuring the preservation of its linguistic capabilities

## Experiments

### Implementation

#### Data sets

mPLUG-Owl2 is first pre-trained on image-text
pairs and fine-tunes on mono-modal and multi-modal instruction data. For pre-training data, we randomly pick about
400 million image-text pairs from five public datasets: Conceptual Captions (CC3M/CC12M) [9], COCO [35], Laionen [49], COYO [7], DataComp [18]. For instruction data,
we collect 5 types of datasets including 1) image captioning
(i.e., TextCaps [53], COCO [35]); 2) image question answering (i.e., VQAv2 [21], OKVQA [43], OCR-VQA [44], GQA
[24], and A-OKVQA [50]); 3) region-aware QA (i.e., RefCOCO [69], VisualGenome [26]); 4) multi-modal instruct
data (i.e., LLaVA-instruct-150K [38]); 5) text-only instruct data (i.e., ShareGPT-80K [1], SlimOrca [34]). Details can
be found in the Appendix.

#### Training Settings

We pre-train the model for 42,500 iterations with a batch size 8,192 for about 348 million imagetext pairs. Since we adopt the language modeling loss,
the large batch size can be easily achieved by the gradient accumulation technique. mPLUG-Owl2 adopts ViT-L
[48] with patch size 14 × 14 and pre-trained at resolution
224 × 224. We use the same data augmentation in BLIP2 [31], including random resized cropping, and horizontal
flipping with a probability of 0.5. The number of layers in
the visual abstractor is set to 6 and it is randomly initialized. The number of learnable queries is set to 64. For the
language model, LLaMA-2 [58] is employed for handling
multi-modal features with 7B parameters, and the parameters of modality-adaptive modules are initialized from the
language model. We use the AdamW [40] optimizer with
β1 = 0.9, β2 = 0.98 and ϵ =1e-6 for optimization. The
cosine learning rate decay scheduler with a peak learning
rate of 1e-4 and with warmup steps 1k. For the learning
rate of the vision encoder, we employ layer-wise learning
rate decay with a factor of 0.9 to retain the low-level visual representation. For the instruction tuning stage, we train the
whole model for 1 epoch with a learning rate of 2e-5 and
batch size 256. Besides, we increase the resolution from
224 × 224 to 448 × 448. The layer-wise learning rate decay
is also employed which is crucial for retaining good visual
representation in our experiments.

### Main Results
