# mPLUG-DocOwl : Modularized Multimodal Large Language Model for Document Understanding

## Abstract

Document understanding refers to automatically extract, analyze and comprehend information from various types of digital documents, such as a web page. Existing Multi-model Large Language Models (MLLMs), including mPLUG-Owl, have demonstrated promising zero-shot capabilities in shallow OCR-free text recognition, indicating their potential for OCR-free document understanding. Nevertheless, without in-domain training, these models tend to ignore fine-grained OCR features, such as sophisticated tables or large blocks of text, which are essential for OCR-free document understanding. In this paper, we propose mPLUG-DocOwl based on mPLUG-Owl for OCR-free document understanding. Specifically, we first construct a instruction tuning dataset featuring a wide range of visual-text understanding tasks. Then, we strengthen the OCR-free document understanding ability by jointly train the model on language-only, general vision-and-language, and document instruction tuning dataset with our unified instruction tuning strategy. We also build an OCR-free document instruction understanding evaluation set LLMDoc to better compare models’ capabilities on instruct compliance and document understanding. Experimental results show that our model outperforms existing multi-modal models, demonstrating its strong ability of document understanding. Besides, without specific fine-tuning, mPLUG-DocOwl generalizes well on various downstream tasks. Our code, models, training data and evaluation set are available at <https://github.com/X-PLUG/mPLUG-DocOwl>.

문서 이해는 웹 페이지와 같은 다양한 유형의 디지털 문서에서 정보를 자동으로 추출, 분석, 이해하는 것을 말합니다. mPLUG-Owl을 비롯한 기존의 다중 모델 대규모 언어 모델(MLLM)은 얕은 OCR 없는 텍스트 인식에서 유망한 제로 샷 기능을 입증하여 OCR-free document 이해의 가능성을 보여주었습니다. 하지만 이러한 모델은 도메인 내 학습이 없으면 OCR-free document 이해에 필수적인 정교한 표나 큰 텍스트 블록과 같은 세분화된 OCR 기능을 무시하는 경향이 있습니다. 이 백서에서는 OCR-free document 이해를 위해 mPLUG-Owl에 기반한 mPLUG-DocOwl을 제안합니다. 구체적으로, 먼저 다양한 시각 텍스트 이해 작업을 포함하는 명령어 튜닝 데이터셋을 구축합니다. 그런 다음 통합 명령어 튜닝 전략으로 언어 전용, 일반 시각 및 언어, 문서 명령어 튜닝 데이터셋에 대해 모델을 공동으로 훈련하여 OCR-free document 이해 능력을 강화합니다. 또한 명령어 준수 및 문서 이해에 대한 모델의 능력을 더 잘 비교하기 위해 OCR-free document 명령어 이해 평가 세트 LLMDoc을 구축합니다. 실험 결과에 따르면 저희 모델이 기존의 멀티모달 모델보다 뛰어난 성능을 보이며 강력한 문서 이해 능력을 입증했습니다. 또한, 특별한 미세 조정 없이도 mPLUG-DocOwl은 다양한 다운스트림 작업에서 잘 일반화됩니다. 코드, 모델, 학습 데이터 및 평가 세트는<https://github.com/X-PLUG/mPLUG-DocOwl>에서 확인할 수 있습니다.

## Introduction

Large language models (LLMs) like ChatGPT [OpenAI, 2022], BLOOM [Scao et al., 2022], and LLaMA [Touvron et al., 2023] have undergone rapid development to enable the realization of general artificial intelligence, boasting impressive zero-shot capabilities across diverse linguistic applications. With the LLM as the language decoder, Multimodal large language models (MLLMs) such as MiniGPT-4 [Zhu et al., 2023], LLaVA [Liu et al., 2023a], and mPLUG-Owl [Ye et al., 2023] have demonstrated remarkable zero-shot performance in various open-ended vision-and-language tasks. These models are trained to align text and images during the pre-training phase, and then to promote diverse abilities during the instruction tuning phase. Interestingly, these MLLMs exhibit superficial OCR-free text recognition abilities without explicit training on visual text understanding datasets [Ye et al., 2023, Liu et al., 2023b]. Nevertheless, due to lacking specific training, these models still face the challenge of comprehending intricate relationships between visual text and objects in diverse types of images, such as charts, documents and webpages.

By performing unified instruction tuning for Document Understanding upon the mPLUG-Owl [Ye et al., 2023], we further propose a modularized MLLM [Li et al., 2022, Xu et al., 2023b], namely mPLUG-DocOwl. Our approach utilizes a modularized framework similar to mPLUG-Owl [Ye et al., 2023], which incorporates a visual abstractor module to link a pre-trained LLM with a visual knowledge module, achieving the alignment of text and images. To enhance diverse document understanding capabilities, we reorganize various downstream document understanding tasks in the same form of instructions. To maintain general uni/multi-modal abilities, we also include language-only and general vision-and-language instruction datasets used by mPLUG-Owl to train the mPLUG-DocOwl. During training, both the visual knowledge module and LLM decoder are frozen, only the visual abstractor and the Low-Rank Adaption (LoRA) [Hu et al., 2022] in LLM are fine-tuned.

mPLUG-DocOwl achieves ocr-free state-of-the-art performance on multiple commonly used document understanding datasets. Furthermore, our experiments on a carefully-built document instruction understanding evaluation set LLMDoc shows that mPLUG-DocOwl achieves significantly better visual text understanding performance on various domains than existing MLMMs.

Our main contributions can be highlighted as follows:

• We propose a modularized MLLM, mPLUG-DocOwl, which is the first one to balance language-only, general vision-and-language, and document understanding based on unified instruction tuning.

• We carefully construct an instruction understanding test set with human evaluation, dubbed LLMDoc, to assess diverse document understanding capabilities.

• Empirical results demonstrate that our mPLUG-DocOwl surpasses existing methods on ocr-free document understanding, including multiple standard benchmarks and LLMDoc.

ChatGPT[OpenAI, 2022], BLOOM[Scao 외, 2022], LLaMA[Touvron 외, 2023]와 같은 대규모 언어 모델(LLM)은 다양한 언어 애플리케이션에서 인상적인 제로 샷 기능을 자랑하며 일반 인공 지능의 실현을 가능하게 하기 위해 빠르게 발전하고 있습니다. 언어 디코더로 LLM을 사용하는 MiniGPT-4[Zhu 외, 2023], LLaVA[Liu 외, 2023a], mPLUG-Owl[Ye 외, 2023] 등의 멀티모달 대규모 언어 모델(MLLM)은 다양한 개방형 비전 및 언어 작업에서 놀라운 제로 샷 성능을 입증했습니다. 이러한 모델은 사전 훈련 단계에서 텍스트와 이미지를 정렬하도록 훈련된 다음, 명령어 튜닝 단계에서 다양한 능력을 촉진하도록 훈련됩니다. 흥미롭게도 이러한 MLLM은 시각적 텍스트 이해 데이터 세트에 대한 명시적인 훈련 없이도 피상적인 OCR 없는 텍스트 인식 능력을 보여줍니다[Ye et al., 2023, Liu et al., 2023b]. 그럼에도 불구하고 이러한 모델은 특정 훈련이 부족하기 때문에 차트, 문서, 웹페이지와 같은 다양한 유형의 이미지에서 시각적 텍스트와 개체 간의 복잡한 관계를 이해해야 하는 과제에 직면해 있습니다.

mPLUG-Owl[Ye et al., 2023]을 기반으로 문서 이해를 위한 통합 명령어 튜닝을 수행함으로써, 우리는 모듈화된 MLLM[Li et al., 2022, Xu et al., 2023b], 즉 mPLUG-DocOwl을 추가로 제안합니다. 우리의 접근 방식은 시각적 추상화 모듈을 통합하여 사전 학습된 LLM을 시각적 지식 모듈과 연결하여 텍스트와 이미지의 정렬을 달성하는 mPLUG-Owl [Ye et al., 2023]과 유사한 모듈화된 프레임워크를 활용합니다. 다양한 문서 이해 능력을 향상시키기 위해 다양한 하위 문서 이해 작업을 동일한 형태의 명령어로 재구성합니다. 일반적인 유니/멀티 모달 능력을 유지하기 위해 언어 전용 및 일반 시각/언어 명령어 데이터 세트도 mPLUG-Owl에 포함시켜 mPLUG-DocOwl을 훈련시킵니다. 훈련 중에 시각 지식 모듈과 LLM 디코더는 모두 고정되고, 시각 추상화기와 LLM의 저순위 적응[LoRA](Hu et al., 2022)만 미세 조정됩니다.

mPLUG-DocOwl은 일반적으로 사용되는 여러 문서 이해 데이터 세트에서 ocr이 필요 없는 최첨단 성능을 달성합니다. 또한 세심하게 구축된 문서 명령어 이해 평가 세트 LLMDoc에 대한 실험 결과, mPLUG-DocOwl은 다양한 도메인에서 기존 MLMM보다 훨씬 우수한 시각 텍스트 이해 성능을 달성하는 것으로 나타났습니다.

주요 기여 사항은 다음과 같이 강조할 수 있습니다:

- 통합 인스트럭션 튜닝을 기반으로 언어 전용, 일반 시각 및 언어, 문서 이해의 균형을 맞추는 최초의 모듈화된 MLLM인 mPLUG-DocOwl을 제안합니다.

- 다양한 문서 이해 능력을 평가하기 위해 LLMDoc이라는 인간 평가가 포함된 지침 이해 테스트 세트를 신중하게 구성합니다.

- 경험적 결과에 따르면 mPLUG-DocOwl은 여러 표준 벤치마크와 LLMDoc을 포함하여 OCR-free document 이해에 있어 기존 방법을 능가하는 것으로 나타났습니다.

## mPLUG-DocOwl

## Experiment
