# UReader: Universal OCR-free Visually-situated Language Understanding with Multimodal Large Language Model

## Abstract

Text is ubiquitous in our visual world, conveying crucial information, such as in documents, websites, and everyday photographs. In this work, we propose UReader, a first exploration of universal OCR-free visually-situated language understanding based on the Multimodal Large Language Model (MLLM). By leveraging the shallow text recognition ability of the MLLM, we only finetuned 1.2% parameters and the training cost is much lower than previous work following domain-specific pretraining and finetuning paradigms. Concretely, UReader is jointly finetuned on a wide range of Visually-situated Language Understanding tasks via a unified instruction format. To enhance the visual text and semantic understanding, we further apply two auxiliary tasks with the same format, namely text reading and key points generation tasks. We design a shape-adaptive cropping module before the encoder-decoder architecture of MLLM to leverage the frozen low-resolution vision encoder for processing high-resolution images. Without downstream finetuning, our single model achieves state-of-the-art ocr-free performance in 8 out of 10 visually-situated language understanding tasks, across 5 domains: documents, tables, charts, natural images, and webpage screenshots. Codes and instruction-tuning datasets are released at <https://github.com/LukeForeverYoung/UReader>

텍스트는 문서, 웹사이트, 일상 사진 등 우리의 시각적 세계에서 어디에나 존재하며 중요한 정보를 전달합니다. 이 연구에서는 멀티모달 대규모 언어 모델(MLLM)을 기반으로 OCR이 필요 없는 보편적인 시각적 언어 이해에 대한 첫 번째 탐색인 UReader를 제안합니다. MLLM의 얕은 텍스트 인식 능력을 활용하여 1.2%의 파라미터만 미세 조정했으며, 도메인별 사전 훈련 및 미세 조정 패러다임에 따른 이전 작업보다 훈련 비용이 훨씬 낮습니다. 구체적으로, UReader는 통합된 명령 형식을 통해 광범위한 시각적 상황 언어 이해 과제에 대해 공동으로 미세 조정됩니다. 시각적 텍스트와 의미 이해를 향상시키기 위해 동일한 형식의 두 가지 보조 과제, 즉 텍스트 읽기 및 핵심 포인트 생성 과제를 추가로 적용합니다. MLLM의 인코더-디코더 아키텍처 이전에 형상 적응형 자르기 모듈을 설계하여 고해상도 이미지 처리를 위해 고정된 저해상도 비전 인코더를 활용합니다. 다운스트림 미세 조정 없이 단일 모델로 문서, 표, 차트, 자연 이미지, 웹페이지 스크린샷 등 5개 영역에 걸쳐 시각적 언어 이해 작업 10개 중 8개에서 최첨단 OCR 없는 성능을 달성합니다. 코드 및 명령어 튜닝 데이터 세트는<https://github.com/LukeForeverYoung/UReader>에서 공개됩니다.

Translated with DeepL.com (free version)

## Introduction

Leveraging strong Large Language Models as the language decoder, some recent works propose Multimodal Large Language Models (MLLMs) (Zhu et al., 2023; Liu et al., 2023a; Ye et al., 2023; Li et al., 2023) and achieve promising vision-andlanguage understanding performance. Surprisingly, without in-domain training, these MLLMs exhibit shallow zero-shot visual text recognition ability when fed a low-resolution image with salient text information (Ye et al., 2023; Liu et al., 2023b). However, due to the variety of image types and the wide range of image sizes, they are still far from universal visually-situated language understanding, such as extracting information from documents, reading texts from webpages, and visual question and answering on tables, as shown in Figure 1.

Existing works for visually-situated language understanding can be categorized into two-stage (Xu arXiv:2310.05126v1 [cs.CV] 8 Oct 2023 et al., 2021; Huang et al., 2022; Yang et al., 2021) and end-to-end (Davis et al., 2022; Kim et al., 2022; Lee et al., 2022) methods according to whether relying on an off-the-shelf OCR model or API. These works all follow a domain-specific pretraining and finetuning paradigm, thus leading to high training costs, e.g. end-to-end model Donut (Kim et al., 2022) costs more than 192 A100 days.

Inspired by the shallow text recognition ability of existing MLLMs, in this work, we propose UReader for universal OCR-free visually-situated language understanding, which leverages the multimodal Large Language Model via low-cost instruction tuning (Dai et al., 2023). Different from previous works, we forgo pretraining tasks by leveraging the existing MLLM and directly finetune MLLM by taking full advantage of various Visually-situated Language Understanding datasets. To make the most of the strong language understanding ability of MLLM, we convert all tasks into the visionlanguage instruction tuning format. Besides, to enhance text recognition and semantic understanding ability across diverse domains, we design auxiliary text reading and key points generation tasks in the same instruction format. To utilize the lowresolution encoder of MLLM for processing highresolution images and avoid blurry and distortion problems due to resizing, we propose a shapeadaptive cropping module to cut a high-resolution image into multiple local images. Each image is firstly independently encoded with the frozen visual encoder and a trainable visual abstractor and then concatenated to feed into the language decoder. Moreover, we add learnable crop position encoding to help the model correlate local images and add a resized global image to alleviate salient information loss due to cropping.

Our contributions in this work are four-fold:
• We first propose instruction tuning with Multimodal Large Language Models for OCR-free Visually-situated Language Understanding.
• We build an instruction-tuning dataset covering 5 domains of visually-situated language understanding: document, table, chart, natural image, and webpage screenshot.
• We design a shape-adaptive cropping module to utilize the frozen low-resolution vision encoder for processing high-resolution images.
• UReader achieves state-of-the-art OCR-free performance in 8 out of 10 tasks, across 5 domains.

최근 일부 연구에서는 강력한 대규모 언어 모델을 언어 디코더로 활용하여 다중 모드 대규모 언어 모델(MLLM)을 제안하고(Zhu et al., 2023; Liu et al., 2023a; Ye et al., 2023; Li et al., 2023) 유망한 시각 및 언어 이해 성능을 달성했습니다. 놀랍게도, 도메인 내 훈련 없이도 이러한 MLLM은 눈에 띄는 텍스트 정보가 포함된 저해상도 이미지를 제공했을 때 얕은 제로 샷 시각 텍스트 인식 능력을 보여줍니다(Ye et al., 2023; Liu et al., 2023b). 그러나 이미지 유형이 다양하고 이미지 크기가 광범위하기 때문에 그림 1과 같이 문서에서 정보 추출, 웹페이지에서 텍스트 읽기, 표에서 시각적 질의응답과 같은 보편적인 시각적 상황 언어 이해에는 아직 미치지 못하고 있습니다.

시각적 상황 언어 이해를 위한 기존 연구들은 상용 OCR 모델을 사용하는지 API를 사용하는지에 따라 2단계(Xu arXiv:2310.05126v1 [cs.CV] 8 Oct 2023 외, 2021; Huang 외, 2022; Yang 외, 2021)와 엔드투엔드(Davis 외, 2022; Kim 외, 2022; Lee 외, 2022) 방식으로 구분할 수 있습니다. 이러한 작업은 모두 도메인별 사전 훈련과 미세 조정 패러다임을 따르기 때문에 훈련 비용이 높습니다(예: 엔드투엔드 모델인 도넛(Kim et al., 2022)은 192 A100일 이상의 비용이 소요됩니다).

기존 MLLM의 얕은 텍스트 인식 능력에 착안하여, 본 연구에서는 저비용 인스트럭션 튜닝을 통해 멀티모달 대규모 언어 모델을 활용하는 범용 OCR 없는 시각적 위치 언어 이해를 위한 UReader를 제안합니다(Dai et al., 2023). 기존 연구와 달리 기존 MLLM을 활용한 사전 학습 작업을 생략하고 다양한 시각적 상황 언어 이해 데이터 세트를 최대한 활용하여 MLLM을 직접 미세 조정합니다. MLLM의 강력한 언어 이해 능력을 최대한 활용하기 위해 모든 작업을 시각 언어 명령어 튜닝 형식으로 변환합니다. 또한 다양한 영역에서 텍스트 인식 및 의미 이해 능력을 향상시키기 위해 보조 텍스트 읽기 및 핵심 포인트 생성 작업을 동일한 명령어 형식으로 설계합니다. 고해상도 이미지 처리에 MLLM의 저해상도 인코더를 활용하고 크기 조정으로 인한 흐릿함과 왜곡 문제를 방지하기 위해 고해상도 이미지를 여러 개의 로컬 이미지로 잘라내는 형상 적응적 자르기 모듈을 제안합니다. 각 이미지는 먼저 고정 시각 인코더와 학습 가능한 시각 추상화기로 독립적으로 인코딩된 다음 언어 디코더에 공급하기 위해 연결됩니다. 또한 학습 가능한 자르기 위치 인코딩을 추가하여 모델이 로컬 이미지를 상호 연관시키고 크기가 조정된 글로벌 이미지를 추가하여 자르기로 인한 두드러진 정보 손실을 완화할 수 있도록 지원합니다.

이 작업에서 유니티가 기여한 바는 크게 네 가지입니다:

- 먼저 OCR 없는 시각적 위치 언어 이해를 위해 멀티모달 대규모 언어 모델을 사용한 인스트럭션 튜닝을 제안합니다.
- 문서, 표, 차트, 자연 이미지, 웹페이지 스크린샷 등 시각적 위치 언어 이해의 5가지 영역을 포괄하는 인스트럭션 튜닝 데이터 세트를 구축합니다.
- 고해상도 이미지 처리를 위해 고정된 저해상도 비전 인코더를 활용할 수 있도록 모양 적응형 자르기 모듈을 설계합니다.
- UReader는 5개 영역에 걸쳐 10개 작업 중 8개 작업에서 최첨단 OCR 없는 성능을 달성합니다.

## Related Work

## UReader

The primary goal of UReader is to efficiently utilize existing MLLMs for Visually-situated Language Understanding tasks. In this work, we utilize but are not limited to, the mPLUG-Owl (Ye et al., 2023) as our basic MLLM. Figure 2 presents an overall architecture of UReader. The input image is firstly pre-processed by a shape-adaptive cropping module (in Section 3.1). The resulting sub-images are then simultaneously passed through the visual encoder and visual abstractor. To enable the large language model to correlate multiple cropped subimages, we apply a crop position encoding module to introduce spatial information across sub-images. (in Section 3.2).

UReader의 주요 목표는 시각적 상황 언어 이해 작업에 기존 MLLM을 효율적으로 활용하는 것입니다. 이 작업에서는 mPLUG-Owl(Ye et al., 2023)을 기본 MLLM으로 활용하지만 이에 국한되지 않습니다. 그림 2는 UReader의 전체 아키텍처를 보여줍니다. 입력 이미지는 먼저 모양 적응형 자르기 모듈(섹션 3.1)에 의해 사전 처리됩니다. 그런 다음 결과 하위 이미지가 시각 인코더와 시각 추상화기를 동시에 통과합니다. 대규모 언어 모델이 여러 개의 자른 하위 이미지를 상호 연관시킬 수 있도록 하기 위해 자르기 위치 인코딩 모듈을 적용하여 하위 이미지에 공간 정보를 도입합니다. (섹션 3.2 참조).

### Shape-Adaptive Cropping Module

Images with texts have various aspect ratios and a great range of resolutions. Simply resizing the image to Hv, Wv (raw resolution of the MLLM) can result in text being blurred, distorted, and unrecognizable. Thus we propose a shape-adaptive cropping module. Specifically, as shown in Figure 3, we pre-define grids {g = (nh × nw)|nh · nw ≤ Nc, nh ∈ N, nw ∈ N} with various shapes, where nh and nw denote the number of rows and columns of the grid g and Nc denotes the maximum number of the cells (sub-images). To select a suitable grid for an image I with shape H ×W, two rules should be followed: (1) The grid should preserve the resolution of the image as much as possible, and (2) the grid should fit the aspect ratio of the input image. To measure the resolution coherence and shape similarity between the image and each grid, we calculate the resolution-related and resolution-agnostic insection over union Srr and Sra as follows:

equition (1)

where IoU denotes the insection over the union between two rectangles centered and aligned with each other. The matched grid is selected by maximizing the matching score:

equition (2)

where g ∗ is the selected grid. Then, we resize the input image to (nhHv, nwWv) and crop it to nh · nw local images. To maintain the global structure information of the image, we also resize the input image to (Hv, Wv) as a global image. All images are then passed on to the visual encoder and visual abstractor in parallel.

The visual encoder extracts visual feature V ∈ R N×(H′ ·W′ )×dv from the input images I ∈ R N×H×W×3 , where N = (nh · nw) + 1, H′ · W′ and dv denote the number and dimension of the extracted visual features, respectively. The visual abstractor further summarizes visual information and obtains higher semantic visual representations V l ∈ R N×Nq×dl in language feature space by several learnable queries, where dl denotes the dimension of language feature space and Nq denotes the number of learnable queries.

텍스트가 포함된 이미지에는 다양한 가로 세로 비율과 다양한 해상도가 있습니다. 단순히 이미지의 크기를 Hv, Wv(MLLM의 원시 해상도)로 조정하면 텍스트가 흐려지고 왜곡되어 알아볼 수 없게 될 수 있습니다. 따라서 형태 적응형 자르기 모듈을 제안합니다. 구체적으로 그림 3과 같이 다양한 모양의 그리드{g = (nh × nw)|nh - nw ≤ Nc, nh ∈ N, nw ∈ N}를 미리 정의하고, 여기서 nh와 nw는 그리드 g의 행과 열 수를 나타내고 Nc는 셀(하위 이미지)의 최대 수를 나타냅니다. H × W 모양의 이미지 I에 적합한 격자를 선택하려면 두 가지 규칙을 따라야 합니다: (1) 그리드는 이미지의 해상도를 최대한 보존해야 하고, (2) 그리드는 입력 이미지의 종횡비에 맞아야 합니다. 이미지와 각 그리드 간의 해상도 일관성 및 모양 유사성을 측정하기 위해 다음과 같이 Srr과 Sra의 합에 대한 해상도 관련 및 해상도에 무관한 인섹션을 계산합니다:

방정식 (1)

여기서 IoU는 서로 중심을 맞추고 정렬된 두 직사각형 사이의 합집합에 대한 인섹션을 나타냅니다. 일치하는 격자는 일치 점수를 최대화하여 선택됩니다:

방정식 (2)

여기서 g ∗는 선택된 그리드입니다. 그런 다음 입력 이미지의 크기를 (nhHv, nwWv)로 조정하고 nh - nw 로컬 이미지로 크롭합니다. 이미지의 글로벌 구조 정보를 유지하기 위해 입력 이미지의 크기도 글로벌 이미지로 (Hv, Wv)로 조정합니다. 그런 다음 모든 이미지가 시각 인코더와 시각 추상화기로 병렬로 전달됩니다.

시각 인코더는 입력 이미지 I ∈ R N×H×W×3에서 시각 특징 V ∈ R N×(H′ -W′)×dv를 추출합니다. 여기서 N = (nh - nw) + 1, H′ - W′ 및 dv는 각각 추출된 시각 특징의 수와 차원을 나타냅니다. 시각 추상화기는 시각 정보를 추가로 요약하고 여러 학습 가능한 쿼리를 통해 언어 특징 공간에서 더 높은 의미의 시각적 표현 V l ∈ R N×Nq×dl을 얻습니다. 여기서 dl은 언어 특징 공간의 차원을 나타내고 Nq는 학습 가능한 쿼리의 수를 나타냅니다.

### Cropped Images Modeling with LLM

MLLMs are mostly trained with a single image as the input. Due to the cropping module, we need to input visual features from multiple images into the language model. The 1-dimensional position embeddings of LLM can not reflect the spatial position of each sub-image, which is critical to correlate local images. Therefore, we incorporate a 2-dimensional crop position encoding to help the language model to understand the spatial relationship between cropped images. Specifically, we assign a location index (i, j) for each cell of the selected grid and obtain their row embedding and column embedding by two auxiliary embedding layers as follows:

equition (3)

where ei,j ∈ R Dl denotes the crop position embedding of the cell (ci , cj ). We add the embedding to the visual feature of each cell in the language space via broadcasting along the dimension of learnable queries: V¯ l i,j = V l i,j + ei,j . We then reshape the visual features into V¯ l ∈ R (N·Nq)×dl . The resulting spatial-aware visual features and word embeddings of the input sentences are concatenated at sequence dimension and sent to the large language model. In order to enhance the language model’s ability to effectively model multiple images while keeping low training costs, we freeze the origin language model and adopt the low-rank adaptation approach (LoRA) (Hu et al., 2022).

MLLM은 대부분 단일 이미지를 입력으로 학습합니다. 자르기 모듈로 인해 여러 이미지의 시각적 특징을 언어 모델에 입력해야 합니다. LLM의 1차원 위치 임베딩은 각 하위 이미지의 공간적 위치를 반영할 수 없으며, 이는 로컬 이미지의 상관관계에 매우 중요합니다. 따라서 언어 모델이 자른 이미지 간의 공간적 관계를 이해할 수 있도록 2차원 자르기 위치 인코딩을 통합합니다. 구체적으로, 선택한 그리드의 각 셀에 위치 인덱스(i, j)를 할당하고 다음과 같이 두 개의 보조 임베딩 레이어를 통해 행 임베딩과 열 임베딩을 얻습니다:

방정식 (3)

여기서 ei,j ∈ R Dl은 셀의 자르기 위치 임베딩(ci , cj )을 나타냅니다. 학습 가능한 쿼리의 차원을 따라 방송을 통해 언어 공간에서 각 셀의 시각적 특징에 임베딩을 추가합니다: V¯ l i,j = V l i,j + ei,j 입니다. 그런 다음 시각적 특징을 V¯ l ∈ R (N-Nq)×dl 로 재형성합니다. 이렇게 생성된 공간 인식 시각적 특징과 입력 문장의 단어 임베딩은 시퀀스 차원에서 연결되고 대규모 언어 모델로 전송됩니다. 훈련 비용을 낮게 유지하면서 여러 이미지를 효과적으로 모델링하는 언어 모델의 능력을 향상시키기 위해 원본 언어 모델을 동결하고 낮은 순위 적응 접근법(LoRA)을 채택합니다(Hu et al., 2022).

## Instruction Tuning

For developing a universal visually-situated language understanding model that could process various types of images and perform different comprehension tasks, we conduct low-cost instruction tuning with a Multimodal Large Language Model. Without introducing any large-scale pretraining datasets, we directly ensemble multiple downstream datasets and perform joint training. Different downstream tasks are all reorganized to the unified instruction format (Dai et al., 2023). Besides, we design auxiliary text reading and key points generation tasks to enhance text recognition and semantic understanding ability.
