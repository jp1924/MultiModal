# BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

## Abstract

The cost of vision-and-language pre-training has
become increasingly prohibitive due to end-toend training of large-scale models. This paper
proposes BLIP-2, a generic and efficient pretraining strategy that bootstraps vision-language
pre-training from off-the-shelf frozen pre-trained
image encoders and frozen large language models. BLIP-2 bridges the modality gap with a
lightweight Querying Transformer, which is pretrained in two stages. The first stage bootstraps vision-language representation learning
from a frozen image encoder. The second stage
bootstraps vision-to-language generative learning
from a frozen language model. BLIP-2 achieves
state-of-the-art performance on various visionlanguage tasks, despite having significantly fewer
trainable parameters than existing methods. For
example, our model outperforms Flamingo80B by
8.7% on zero-shot VQAv2 with 54x fewer trainable parameters. We also demonstrate the model’s
emerging capabilities of zero-shot image-to-text
generation that can follow natural language instructions.

## Introduction

Vision-language pre-training (VLP) research has witnessed
a rapid advancement in the past few years, where pre-trained
models with increasingly larger scale have been developed
to continuously push the state-of-the-art on various downstream tasks (Radford et al., 2021; Li et al., 2021; 2022;
Wang et al., 2022a; Alayrac et al., 2022; Wang et al., 2022b).
However, most state-of-the-art vision-language models incur a high computation cost during pre-training, due to
end-to-end training using large-scale models and datasets

Vision-language research sits at the intersection between
vision and language, therefore it is naturally expected
that vision-language models can harvest from the readilyavailable unimodal models from the vision and natural language communities. In this paper, we propose a generic and compute-efficient VLP method by bootstrapping from offthe-shelf pre-trained vision models and language models.
Pre-trained vision models offer high-quality visual representation. Pre-trained language models, in particular large language models (LLMs), offer strong language generation and
zero-shot transfer abilities. To reduce computation cost and
counteract the issue of catastrophic forgetting, the unimodal
pre-trained models remain frozen during the pre-training.

In order to leverage pre-trained unimodal models for VLP,
it is key to facilitate cross-modal alignment. However, since
LLMs have not seen images during their unimodal pretraining, freezing them makes vision-language alignment
in particular challenging. In this regard, existing methods
(e.g. Frozen (Tsimpoukelli et al., 2021), Flamingo (Alayrac
et al., 2022)) resort to an image-to-text generation loss,
which we show is insufficient to bridge the modality gap.

To achieve effective vision-language alignment with frozen
unimodal models, we propose a Querying Transformer (QFormer) pre-trained with a new two-stage pre-training strategy. As shown in Figure 1, Q-Former is a lightweight transformer which employs a set of learnable query vectors to
extract visual features from the frozen image encoder. It
acts as an information bottleneck between the frozen image
encoder and the frozen LLM, where it feeds the most useful

visual feature for the LLM to output the desired text. In
the first pre-training stage, we perform vision-language representation learning which enforces the Q-Former to learn
visual representation most relevant to the text. In the second
pre-training stage, we perform vision-to-language generative learning by connecting the output of the Q-Former to a
frozen LLM, and trains the Q-Former such that its output
visual representation can be interpreted by the LLM.

We name our VLP framework as BLIP-2: Bootstrapping
Language-Image Pre-training with frozen unimodal models.
The key advantages of BLIP-2 include:

• BLIP-2 effectively leverages both frozen pre-trained image models and language models. We bridge the modality
gap using a Q-Former pre-trained in two-stages: representation learning stage and generative learning stage.
BLIP-2 achieves state-of-the-art performance on various
vision-language tasks including visual question answering, image captioning, and image-text retrieval

• Powered by LLMs (e.g. OPT (Zhang et al., 2022),
FlanT5 (Chung et al., 2022)), BLIP-2 can be prompted to
perform zero-shot image-to-text generation that follows
natural language instructions, which enables emerging
capabilities such as visual knowledge reasoning, visual
conversation, etc. (see Figure 4 for examples).

• Due to the use of frozen unimodal models and a
lightweight Q-Former, BLIP-2 is more compute-efficient
than exisiting state-of-the-arts. For example, BLIP-2 outperforms Flamingo (Alayrac et al., 2022) by 8.7% on
zero-shot VQAv2, while using 54× fewer trainable parameters. Furthermore, our results show that BLIP-2 is a
generic method that can harvest more advanced unimodal
models for better VLP performance.

## Related Work

### End-to-end Vision-Language Pre-training

Vision-language pre-training aims to learn multimodal foundation models with improved performance on various visionand-language tasks. Depending on the downstream task,
different model architectures have been proposed, including
the dual-encoder architecture (Radford et al., 2021; Jia et al.,
2021), the fusion-encoder architecture (Tan & Bansal, 2019;
Li et al., 2021), the encoder-decoder architecture (Cho et al.,
2021; Wang et al., 2021b; Chen et al., 2022b), and more
recently, the unified transformer architecture (Li et al., 2022;
Wang et al., 2022b). Various pre-training objectives have
also been proposed over the years, and have progressively
converged to a few time-tested ones: image-text contrastive
learning (Radford et al., 2021; Yao et al., 2022; Li et al.,
2021; 2022), image-text matching (Li et al., 2021; 2022;
Wang et al., 2021a), and (masked) language modeling (Li
et al., 2021; 2022; Yu et al., 2022; Wang et al., 2022b). Most VLP methods perform end-to-end pre-training using
large-scale image-text pair datasets. As the model size keeps
increasing, the pre-training can incur an extremely high
computation cost. Moreover, it is inflexible for end-to-end
pre-trained models to leverage readily-available unimodal
pre-trained models, such as LLMs (Brown et al., 2020;
Zhang et al., 2022; Chung et al., 2022).

### Modular Vision-Language Pre-training

More similar to us are methods that leverage off-the-shelf
pre-trained models and keep them frozen during VLP. Some
methods freeze the image encoder, including the early work
which adopts a frozen object detector to extract visual features (Chen et al., 2020; Li et al., 2020; Zhang et al., 2021),
and the recent LiT (Zhai et al., 2022) which uses a frozen
pre-trained image encoder for CLIP (Radford et al., 2021)
pre-training. Some methods freeze the language model
to use the knowledge from LLMs for vision-to-language
generation tasks (Tsimpoukelli et al., 2021; Alayrac et al.,
2022; Chen et al., 2022a; Manas et al. ˜ , 2023; Tiong et al.,
2022; Guo et al., 2022). The key challenge in using a frozen
LLM is to align visual features to the text space. To achieve
this, Frozen (Tsimpoukelli et al., 2021) finetunes an image
encoder whose outputs are directly used as soft prompts
for the LLM. Flamingo (Alayrac et al., 2022) inserts new
cross-attention layers into the LLM to inject visual features,
and pre-trains the new layers on billions of image-text pairs.
Both methods adopt the language modeling loss, where the
language model generates texts conditioned on the image.

Different from existing methods, BLIP-2 can effectively and
efficiently leverage both frozen image encoders and frozen
LLMs for various vision-language tasks, achieving stronger
performance at a lower computation cost.

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
