#reference https://docs.openvino.ai/2023.0/notebooks/237-segment-anything-with-output.html
import warnings
from pathlib import Path
import torch
from openvino.tools import mo
from openvino.runtime import serialize, Core
import sys

from segment_anything import sam_model_registry, SamPredictor
# checkpoint = "sam_vit_b_01ec64.pth"
# model_type = "vit_b"
checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_h"
#sam = sam_model_registry[model_type](checkpoint=checkpoint)
core = Core()

ov_encoder_path = Path("sam_image_encoder.xml")
onnx_encoder_path = ov_encoder_path.with_suffix(".onnx")
if not ov_encoder_path.exists():
    print("sam_image_encoder.xml not found. pls run sam_model_convert")
    sys.exit(1)
else:
    ov_encoder_model = core.read_model(ov_encoder_path)
ov_encoder = core.compile_model(ov_encoder_model, device_name="GPU")

#decoder
from typing import Tuple
from segment_anything.utils.amg import calculate_stability_score

ov_model_path = Path("sam_mask_predictor.xml")
if not ov_model_path.exists():
    print("sam_mask_predictor.xml not found. pls run sam_model_convert")
    sys.exit(1)
else:
    ov_model = core.read_model(ov_model_path)
ov_predictor = core.compile_model(ov_model, device_name="GPU")


import numpy as np
from copy import deepcopy
from typing import Tuple
from torchvision.transforms.functional import resize, to_pil_image

class ResizeLongestSide:
    """
    Resizes images to longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming numpy arrays.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


resizer = ResizeLongestSide(1024)


def preprocess_image(image: np.ndarray):
    resized_image = resizer.apply_image(image)
    resized_image = (resized_image.astype(np.float32) - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
    resized_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)).astype(np.float32), 0)

    # Pad
    h, w = resized_image.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    x = np.pad(resized_image, ((0, 0), (0, 0), (0, padh), (0, padw)))
    return x


def postprocess_masks(masks: np.ndarray, orig_size):
    size_before_pad = resizer.get_preprocess_shape(orig_size[0], orig_size[1], masks.shape[-1])
    masks = masks[..., :int(size_before_pad[0]), :int(size_before_pad[1])]
    masks = torch.nn.functional.interpolate(torch.from_numpy(masks), size=orig_size, mode="bilinear", align_corners=False).numpy()
    return masks

import cv2

#interactive seg
import gradio as gr
class Segmenter:
    def __init__(self, ov_encoder, ov_predictor):
        self.encoder = ov_encoder
        self.predictor = ov_predictor
        self._img_embeddings = None

    def set_image(self, img:np.ndarray):
        if self._img_embeddings is not None:
            del self._img_embeddings
        preprocessed_image = preprocess_image(img)
        encoding_results = self.encoder(preprocessed_image)
        image_embeddings = encoding_results[ov_encoder.output(0)]
        self._img_embeddings = image_embeddings
        return img

    def get_mask(self, points, img):
        coord = np.array(points)
        coord = np.concatenate([coord, np.array([[0,0]])], axis=0)
        coord = coord[None, :, :]
        label = np.concatenate([np.ones(len(points)), np.array([-1])], axis=0)[None, :].astype(np.float32)
        coord = resizer.apply_coords(coord, img.shape[:2]).astype(np.float32)
        if self._img_embeddings is None:
            self.set_image(img)
        inputs = {
            "image_embeddings": self._img_embeddings,
            "point_coords": coord,
            "point_labels": label,
        }

        results = self.predictor(inputs)
        masks = results[ov_predictor.output(0)]
        masks = postprocess_masks(masks, img.shape[:-1])

        masks = masks > 0.0
        mask = masks[0]
        mask = np.transpose(mask, (1, 2, 0))
        return mask
    
segmenter = Segmenter(ov_encoder, ov_predictor)

from PIL import Image
with gr.Blocks() as demo:
    with gr.Row():
        input_img = gr.Image(label="Input", type="numpy").style(height=480, width=480)
        output_img = gr.Image(label="Selected Segment", type="numpy").style(height=480, width=480)

    def on_image_change(img):
        segmenter.set_image(img)
        return img
        
    def dump_mask(mask_image, mask, selected_val, rest_val, postfix):
        mask_image_dump = mask_image
        mask_image_dump[:] = rest_val
        mask_image_dump[mask.squeeze(-1)] = selected_val
  
        dump_path = "./sam_mask_result_" + postfix + ".png"
        print("... Dumping mask img to " + dump_path + " ...")
        dump_image = Image.fromarray(mask_image_dump)
        dump_image.save(dump_path)

    def get_select_coords(img, evt: gr.SelectData):
        pixels_in_queue = set()
        h, w = img.shape[:2]
        pixels_in_queue.add((evt.index[0], evt.index[1]))
        out = img.copy()
        while len(pixels_in_queue) > 0:
            pixels = list(pixels_in_queue)
            pixels_in_queue = set()
            color = np.random.randint(0, 255, size=(1, 1, 3))
            mask = segmenter.get_mask(pixels, img)
            mask_image = out.copy()
            
            # dump mask
            dump_mask(mask_image, mask, 0, 255, "selected")
            dump_mask(mask_image, mask,  255, 0, "reverted")
            
            mask_image[mask.squeeze(-1)] = color
            out = cv2.addWeighted(out.astype(np.float32), 0.7, mask_image.astype(np.float32), 0.3, 0.0)
        out = out.astype(np.uint8)
        return out

    input_img.select(get_select_coords, [input_img], output_img)
    input_img.upload(on_image_change, [input_img], [input_img])

demo.launch()