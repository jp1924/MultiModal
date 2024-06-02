from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from models import MplugOwlProcessor

from transformers.data.data_collator import DataCollatorMixin


@dataclass
class DataCollatorForMplugOwl(DataCollatorMixin):
    img_token_ids: int
    response_token_ids: List[int]
    processor: MplugOwlProcessor
    padding: Union[bool, str] = "longest"
    return_tensors: str = "pt"

    def torch_call(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_ids = []
        for x in features:
            if x["input_ids"][-1] != self.processor.tokenizer.eos_token_id:
                eos_token = torch.tensor([self.processor.tokenizer.eos_token_id])
                x["input_ids"] = torch.cat([x["input_ids"], eos_token])

            input_ids.append({"input_ids": x["input_ids"]})

        # reformat list to dict and set to pytorch format
        batch = self.processor.tokenizer.pad(
            input_ids,
            padding=self.padding,
            return_attention_mask=True,
            return_tensors=self.return_tensors,
        )

        labels = deepcopy(batch["input_ids"])

        img_token_mask = labels == self.img_token_ids
        max_length = labels.shape[1]

        _, response_token_pos = torch.where(labels == self.response_token_ids[0])
        response_token_pos += len(self.response_token_ids)

        mask_pos = [torch.cat([torch.ones(x), torch.zeros(max_length - x)]) for x in response_token_pos]
        mask_pos = torch.stack(mask_pos)

        labels[mask_pos.bool()] = -100
        labels[~batch["attention_mask"].bool()] = -100
        labels[img_token_mask] = self.img_token_ids

        batch["labels"] = labels
        batch["pixel_values"] = torch.stack([x["pixel_values"] for x in features])

        return batch
