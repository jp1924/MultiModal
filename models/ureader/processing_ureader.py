from typing import Dict, List, Optional, Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
)
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType


class UReaderProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "UReaderImageProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None):
        try:
            super().__init__(image_processor, tokenizer)
        except AttributeError as error:
            error_msg = str(error)
            if "UReaderImageProcessor" in error_msg:
                error_msg = """transformers에 UReaderImageProcessor가 등록되어 있지 않습니다! 상위 파일에서 `setattr(transformers, "UReaderImageProcessor", UReaderImageProcessor)`를 해주세요!"""
            raise AttributeError(error_msg)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        image: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # UReader의 default는 PILImageResampling.BICUBIC임.
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        anchors: Optional[List[List[int]]] = None,
        return_patch_position_masks: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        if image is not None:
            image_inputs = self.image_processor(
                image,
                size=size,
                resample=resample,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                data_format=data_format,
                input_data_format=input_data_format,
                anchors=anchors,
                padding=padding,
                return_patch_position_masks=return_patch_position_masks,
                return_tensors=return_tensors,
            )
        else:
            image_inputs = {}

        if text is not None:
            text_inputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
            )
        else:
            text_inputs = {}
        return BatchFeature(data={**text_inputs, **image_inputs})

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)
