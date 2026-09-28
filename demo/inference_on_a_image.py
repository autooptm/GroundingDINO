import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
from groundingdino.util import ao_runtime as _ao


_AO_FONT = None


def plot_boxes_to_image(image_pil, tgt, with_mask=None):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    if with_mask is None:
        with_mask = not _ao.OPT
    mask = Image.new("L", image_pil.size, 0) if with_mask else None
    mask_draw = ImageDraw.Draw(mask) if with_mask else None

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        global _AO_FONT
        if _AO_FONT is None:
            _AO_FONT = ImageFont.load_default()
        font = _AO_FONT
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        if mask_draw is not None:
            mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


_AO_MEAN = [0.485, 0.456, 0.406]
_AO_STD = [0.229, 0.224, 0.225]
_AO_PREP = {}


def _ao_decode_resize(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    if "resize" not in _AO_PREP:
        _AO_PREP["resize"] = T.RandomResize([800], max_size=1333)
    resized, _ = _AO_PREP["resize"](image_pil, None)
    return image_pil, resized


def _ao_to_tensor(resized, device):
    import numpy as np

    key = ("mean", device)
    if key not in _AO_PREP:
        _AO_PREP[key] = torch.tensor(_AO_MEAN, device=device).view(-1, 1, 1)
        _AO_PREP[("std", device)] = torch.tensor(_AO_STD, device=device).view(-1, 1, 1)
    t = torch.from_numpy(np.asarray(resized, dtype=np.uint8))
    t = t.to(device, non_blocking=True).permute(2, 0, 1).contiguous()
    t = t.float().div(255)
    return t.sub_(_AO_PREP[key]).div_(_AO_PREP[("std", device)])


def load_image(image_path, device=None):
    if _ao.OPT and device is None and torch.cuda.is_available():
        device = "cuda"
    if _ao.OPT and device is not None and device != "cpu":
        image_pil, resized = _ao_decode_resize(image_path)
        return image_pil, _ao_to_tensor(resized, device)

    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize(_AO_MEAN, _AO_STD),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()

    if _ao.OPT and not cpu_only and torch.cuda.is_available():
        model = model.to("cuda")

        for _lay in model.backbone[0].layers:
            _lay.use_checkpoint = False
        model.transformer.encoder.use_checkpoint = False
        model.transformer.encoder.use_transformer_ckpt = False

        try:
            assert _ao.OPT_2, "AO_OPT_2=0"
            from groundingdino.models.GroundingDINO.backbone.swin_transformer import (
                SwinTransformerBlock,
            )
            from groundingdino.models.GroundingDINO.transformer import (
                DeformableTransformerDecoderLayer,
            )

            for _cls in (SwinTransformerBlock, DeformableTransformerDecoderLayer):
                if not getattr(_cls, "_ao_opt_done", False):
                    _cls.forward = torch.compile(_cls.forward)
                    _cls._ao_opt_done = True
        except Exception:  # noqa: BLE001 -- never break inference over this
            pass

    if _ao.OPT_1:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    if not _ao.OPT:
        model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        if _ao.OPT:
            filt_mask = logits.max(dim=1)[0] > box_threshold
            logits_filt = logits[filt_mask].float().cpu()
            boxes_filt = boxes[filt_mask].float().cpu()
        else:
            logits_filt = logits.cpu().clone()
            boxes_filt = boxes.cpu().clone()
            filt_mask = logits_filt.max(dim=1)[0] > box_threshold
            logits_filt = logits_filt[filt_mask]  # num_filt, 256
            boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases


    return boxes_filt, pred_phrases


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--image_path", "-i", type=str, nargs="+", required=True,
                        help="path to image file(s)")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--token_spans", type=str, default=None, help=
                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")

    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    image_path = args.image_path
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    token_spans = args.token_spans

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    image_paths = image_path if isinstance(image_path, list) else [image_path]

    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

    # set the text_threshold to None if token_spans is set.
    if token_spans is not None:
        text_threshold = None
        print("Using token_spans. Set the text_threshold to None.")

    _ao_device = "cpu" if args.cpu_only else "cuda"
    _ao_exec = None
    _ao_next = None
    if _ao.OPT_3 and len(image_paths) > 1:
        from concurrent.futures import ThreadPoolExecutor

        _ao_exec = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ao-io")

    for _unit_i, image_path in enumerate(image_paths):
        suffix = "" if len(image_paths) == 1 else "_%04d" % _unit_i

        # load image
        if _ao_exec is not None:
            image_pil, _resized = (_ao_next or _ao_exec.submit(
                _ao_decode_resize, image_path)).result()
            _ao_next = (_ao_exec.submit(_ao_decode_resize, image_paths[_unit_i + 1])
                        if _unit_i + 1 < len(image_paths) else None)
            image = _ao_to_tensor(_resized, _ao_device)
        else:
            image_pil, image = load_image(image_path, device=_ao_device)

        # visualize raw image
        _raw_out = os.path.join(output_dir, "raw_image%s.jpg" % suffix)
        _ao_raw = (_ao_exec.submit(image_pil.save, _raw_out) if _ao_exec is not None
                   else None)
        if _ao_raw is None:
            image_pil.save(_raw_out)

        # run model
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold,
            cpu_only=args.cpu_only, token_spans=eval(f"{token_spans}")
        )

        # visualize pred
        size = image_pil.size
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }
        if _ao_raw is not None:
            _ao_raw.result()
        image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
        image_with_box.save(os.path.join(output_dir, "pred%s.jpg" % suffix))

    if _ao_exec is not None:
        _ao_exec.shutdown(wait=True)
