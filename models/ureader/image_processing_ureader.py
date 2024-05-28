# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for UReader."""

import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from transformers.image_transforms import resize
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
from transformers.utils import TensorType, logging


try:
    from einops import rearrange, repeat
except ImportError:
    raise ImportError("UReader processor를 사용하기 위해선 einops를 무조건 설치해야 합니다!")


logger = logging.get_logger(__name__)

# copied from UReader > pipeline > data_utils > processors > doc_processor > DocPretrainProcessor
DEFAULT_ANCHORS = [
    (1, 1),
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (1, 4),
    (4, 1),
    (1, 5),
    (5, 1),
    (1, 6),
    (6, 1),
    (2, 3),
    (3, 2),
    (1, 7),
    (7, 1),
    (4, 2),
    (2, 4),
    (1, 8),
    (8, 1),
    (3, 3),
    (1, 9),
    (9, 1),
]

# copied from UReader > pipeline > data_utils > processors > doc_processor > DocPretrainProcessor
UREADER_STANDARD_MEAN = (0.48145466, 0.4578275, 0.40821073)
UREADER_STANDARD_STD = (0.26862954, 0.26130258, 0.27577711)


class UReaderImageProcessor(BaseImageProcessor):
    r"""
    Constructs a UReader image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `(size["height"],
            size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `UREADER_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `UREADER_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """

    model_input_names = ["pixel_values"]
    _processor_class = "UReaderImageProcessor"

    def __init__(
        self,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        anchors: List[List[int]] = DEFAULT_ANCHORS,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 224, "width": 224}
        size = get_size_dict(size)
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean if image_mean is not None else UREADER_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else UREADER_STANDARD_STD

        # shortest_edge는 처리할 수 없음.
        # t_x, t_y, b_x, b_y
        self.anchors = anchors  # for save
        get_anchor_box = lambda anchor: (
            0,
            0,
            anchor[1] * self.size["width"],
            anchor[0] * self.size["height"],
        )
        self.anchors_box = np.array([get_anchor_box(anchor) for anchor in anchors])
        self.max_anchor_box = max([max(anchor) for anchor in anchors])

        self._valid_processor_keys = [
            "images",
            "size",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "return_tensors",
            "data_format",
            "input_data_format",
            "patch_padding",
            "return_padding_mask",
        ]

    def caculate_iou(self, box1: np.ndarray, box2: np.ndarray):
        """
        Calculate IoU (Intersection over Union) between two bounding boxes.

        Args:
            box1 (array-like): Coordinates of the first bounding box in the format (x1, y1, x2, y2).
            box2 (array-like): Coordinates of the second bounding box in the format (x1, y1, x2, y2).

        Returns:
            float: IoU value between the two bounding boxes.
        """
        # Convert input to numpy arrays if they are not already
        box1 = np.array(box1)
        box2 = np.array(box2)

        # Calculate intersection coordinates
        x1 = np.maximum(box1[:, 0], box2[:, 0])
        y1 = np.maximum(box1[:, 1], box2[:, 1])
        x2 = np.minimum(box1[:, 2], box2[:, 2])
        y2 = np.minimum(box1[:, 3], box2[:, 3])

        # Calculate intersection area
        intersection_area = np.maximum(x2 - x1 + 1, 0) * np.maximum(y2 - y1 + 1, 0)

        # Calculate area of each bounding box
        box1_area = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
        box2_area = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)

        # Calculate IoU
        iou = intersection_area / (box1_area + box2_area - intersection_area)

        return iou

    def pad(self, images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        # huggingface에 있는 logest나 maximum과 같은 기능은 없음. 추후 추가할지는 미지수
        max_seq = max([x.shape[0] for x in images])

        image_ls = list()
        mask_ls = list()
        for image in images:
            cur_seq = image.shape[0]

            image, mask = self._pad(image, max_seq, cur_seq)

            mask_ls.append(mask)
            image_ls.append(image)

        padding_mask = np.stack(mask_ls)
        padded_value = np.stack(image_ls)
        return (padded_value, padding_mask)

    def _pad(
        self,
        image: np.ndarray,
        max_seq: int,
        cur_seq: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(image.shape) == 4:
            pad_width = [(0, max_seq - cur_seq), (0, 0), (0, 0), (0, 0)]
        elif len(image.shape) == 2:
            pad_width = [(0, max_seq - cur_seq), (0, 0)]
        else:
            raise ValueError("Unexpected shape error!")

        image = np.pad(image, pad_width, "constant", constant_values=0)
        mask = np.concatenate([np.ones(cur_seq), np.zeros(max_seq - cur_seq)])

        return (image, mask)

    def resize(
        self,
        **kwargs,
    ) -> np.ndarray:
        # 그냥 사용할 일 있을 까봐 이렇게 넣어 둠.
        return self.normal_resize(**kwargs)

    def normal_resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BICUBIC`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        size = get_size_dict(size)
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        output_size = (size["height"], size["width"])
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def anchor_resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],  # 무조건 dict 형태만 들어와야 함.
        anchors: Optional[np.ndarray] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        anchors = anchors if anchors is not None else self.anchors_box
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        # t_x, t_y, b_x, b_y
        image_box = np.array([[0, 0, size["width"], size["height"]]])

        aspect_ratio_y = size["height"] * anchors[:, 2]
        aspect_ratio_y = aspect_ratio_y / size["width"]
        aspect_ratio_y = aspect_ratio_y.reshape(-1, 1)

        aspect_ratio_anchor = np.concatenate([anchors[:, :3], aspect_ratio_y], -1)

        anchor_num = anchors.shape[0]
        image_box = image_box.repeat(anchor_num, 0)

        S_rr = self.caculate_iou(image_box, anchors)
        S_ra = self.caculate_iou(aspect_ratio_anchor, anchors)

        preper_anchor_idx = ((S_ra * 100) + S_rr).argmax()
        selected_anchor = anchors[preper_anchor_idx]

        anchor = selected_anchor[2:].tolist()  # b_x, b_y
        anchor_size_dict = {"width": anchor[0], "height": anchor[1]}

        local_image = self.normal_resize(
            image=image,
            size=anchor_size_dict,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
        )

        return local_image

    def shape_adaptive_croping(
        self,
        nocut_image: np.ndarray,  # C H W
        local_image: np.ndarray,  # C H W
        size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        size_dict = get_size_dict(size)

        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(nocut_image)

        anchor_size_dict = dict()

        if input_data_format == ChannelDimension.FIRST:
            anchor_size_dict["height"] = local_image.shape[1]
            anchor_size_dict["width"] = local_image.shape[2]
            nocut_pattern = "C H W -> 1 C H W"
            local_pattern = "C (num_H H) (num_W W) -> (num_H num_W) C H W"

        elif input_data_format == ChannelDimension.LAST:
            anchor_size_dict["height"] = local_image.shape[0]
            anchor_size_dict["width"] = local_image.shape[1]
            nocut_pattern = "H W C -> 1 H W C"
            local_pattern = "(num_H H) (num_W W) C -> (num_H num_W) H W C"

        else:
            raise ValueError("Unsupported channel dimension format")

        # {"height": local_image.shape[1], "width": local_image.shape[2]}

        nocut_image = rearrange(nocut_image, nocut_pattern)
        local_image = rearrange(
            local_image,
            local_pattern,
            W=size_dict["width"],
            H=size_dict["height"],
        )

        anchor_height = anchor_size_dict["height"] // size_dict["height"]
        anchor_width = anchor_size_dict["width"] // size_dict["width"]

        x_axis = repeat(np.arange(anchor_height), "num_h -> num_h num_w 1", num_w=anchor_width)
        y_axis = repeat(np.arange(anchor_width), "num_w -> num_h num_w 1", num_h=anchor_height)

        # TODO: numpy에 구현되어 있는 rearrange를 사용해도 되지만 귀찮아서 이렇게 구현함. 나중에 바꿀 것
        # num_patch, (ph,pw)
        local_patch = np.concatenate([x_axis, y_axis], axis=2)
        local_patch = rearrange(local_patch, "num_h num_w p-> (num_h num_w) p", p=2)
        nocut_patch = np.ones((1, 2), dtype=np.int32) * self.max_anchor_box

        pixel_values = np.concatenate([nocut_image, local_image], axis=0)
        patch_position = np.concatenate([nocut_patch, local_patch], axis=0)

        # copied from transformers > image_transforms.py > to_channel_dimension_format
        target_channel_dim = ChannelDimension(data_format)
        if input_data_format == target_channel_dim:
            return (pixel_values, patch_position)

        if data_format == ChannelDimension.FIRST:
            # (2, 0, 1) > (0, 3, 1, 2)
            pixel_values = pixel_values.transpose((0, 3, 1, 2))
        elif data_format == ChannelDimension.LAST:
            # (1, 2, 0) > (0, 2, 3, 1)
            pixel_values = pixel_values.transpose((0, 2, 3, 1))
        else:
            raise ValueError("Unsupported channel dimension format: {}".format(data_format))

        return (pixel_values, patch_position)

    def preprocess(
        self,
        images: ImageInput,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # UReader의 default는 PILImageResampling.BICUBIC임.
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        anchors: Optional[List[List[int]]] = None,
        padding: bool = False,
        return_patch_position_masks: bool = False,
        **kwargs,
    ) -> Dict[str, Union[List[np.ndarray], np.ndarray]]:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Dictionary in the format `{"height": h, "width": w}` specifying the size of the output image after
                resizing.
            resample (`PILImageResampling` filter, *optional*, defaults to `self.resample`):
                `PILImageResampling` filter to use if resizing the image e.g. `PILImageResampling.BICUBIC`. Only has
                an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use if `do_normalize` is set to `True`.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        resample = resample if resample is not None else self.resample
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        anchors = anchors if anchors is not None else self.anchors_box
        size = size if size is not None else self.size

        size_dict = get_size_dict(size)

        images = make_list_of_images(images)

        validate_kwargs(
            captured_kwargs=kwargs.keys(),
            valid_processor_keys=self._valid_processor_keys,
        )

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            size=size,
            resample=resample,
        )

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        # NOTE: if do_resize하는 이유
        #       단순 코드 가독성을 위해, 이 코드는 VIT의 ImageProcessor를 기반으로 작성되었는데
        #       코드는 만드는 도중 어느 구간이 resize, rescale, normalize를 하는 곳인지를 쉽게 구분하기 위해
        #       이와 같이 if 문으로 구분하도록 함. UReader에서의 resize는 선택이 아니라 필수 임.

        # NOTE: ChannelDimension.FIRST로 하는 이유
        #       원본 UReader의 doc_processor를 최대한 모방하는 것을 목표로 만들었기 때문에 ChannelDimension.FIRST로 함.
        #       실제 UReader의 doc_processor 상에서 통과하는 F.to_tensor() or ToTensor() 기능을 따라가기 위함.

        # 여기서 부터 SAM의 시작임.
        do_resize = True
        if do_resize:
            nocut_images = [
                self.normal_resize(
                    image=image,
                    size=size_dict,
                    resample=resample,
                    data_format=ChannelDimension.FIRST,
                    input_data_format=input_data_format,
                )
                for image in images
            ]

            # NOTE: get_image_size는 np.ndarray에서만 동작하는 걸 가정하고 만듬.
            get_image_size: np.ndarray = lambda img: {"height": img.shape[0], "width": img.shape[1]}
            local_images = [
                self.anchor_resize(
                    image=image,
                    size=get_image_size(image),
                    anchors=anchors,
                    resample=resample,
                    data_format=ChannelDimension.FIRST,
                    input_data_format=input_data_format,
                )
                for image in images
            ]

        if do_rescale:
            nocut_images = [
                self.rescale(
                    image=image,
                    scale=rescale_factor,
                    data_format=ChannelDimension.FIRST,
                    input_data_format=ChannelDimension.FIRST,
                )
                for image in nocut_images
            ]

            local_images = [
                self.rescale(
                    image=image,
                    scale=rescale_factor,
                    data_format=ChannelDimension.FIRST,
                    input_data_format=ChannelDimension.FIRST,
                )
                for image in local_images
            ]

        if do_normalize:
            nocut_images = [
                self.normalize(
                    image,
                    mean=image_mean,
                    std=image_std,
                    data_format=ChannelDimension.FIRST,
                    input_data_format=ChannelDimension.FIRST,
                )
                for image in nocut_images
            ]

            local_images = [
                self.normalize(
                    image,
                    mean=image_mean,
                    std=image_std,
                    data_format=ChannelDimension.FIRST,
                    input_data_format=ChannelDimension.FIRST,
                )
                for image in local_images
            ]

        pixel_value_ls = list()
        patch_position_ls = list()
        for nocut_image, local_image in zip(nocut_images, local_images):
            pixel_values, patch_position = self.shape_adaptive_croping(
                nocut_image=nocut_image,
                local_image=local_image,
                size=size,
                data_format=data_format,
                input_data_format=ChannelDimension.FIRST,
            )

            pixel_value_ls.append(pixel_values)
            patch_position_ls.append(patch_position)

        if padding:
            pixel_values, patch_padding_mask = self.pad(pixel_value_ls)
            patch_positions, _ = self.pad(patch_position_ls)
            data = {
                "pixel_values": pixel_values,
                "patch_positions": patch_positions,
                "patch_position_mask": patch_padding_mask,
            }
        else:
            data = {"pixel_values": pixel_value_ls, "patch_positions": patch_position_ls}

            if return_tensors:
                warnings.warn("padding=False인 상태에선 무조건 List[np.ndarray] 형식으로 return 됩니다!")
                return_tensors = None

        return BatchFeature(data=data, tensor_type=return_tensors)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this image processor instance.
        """
        output = copy.deepcopy(self.__dict__)

        del output["anchors_box"]
        del output["max_anchor_box"]

        output["image_processor_type"] = self.__class__.__name__

        return output
