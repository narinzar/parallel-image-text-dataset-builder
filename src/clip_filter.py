"""CLIP image-text similarity filtering.

Given image-caption pairs, compute the cosine similarity between the CLIP image
embedding and the CLIP text embedding, then drop pairs whose score is below a
threshold. Low-scoring pairs are usually mislabeled or weakly aligned and hurt
contrastive training.

Two backends are supported so the code runs with whichever is installed:

    open_clip (preferred)  - open_clip_torch
    transformers           - CLIPModel + CLIPProcessor fallback

The model is loaded lazily and cached per process. Scoring is batched and wrapped
in tqdm. GPU is used automatically when torch.cuda is available (e.g. RTX 5090).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from PIL import Image
from tqdm import tqdm


@dataclass
class ClipConfig:
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    hf_model_name: str = "openai/clip-vit-base-patch32"
    batch_size: int = 64
    score_threshold: float = 0.25
    device: str | None = None  # auto-detect when None


def _resolve_device(requested: str | None):
    import torch

    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


class _OpenClipBackend:
    def __init__(self, cfg: ClipConfig) -> None:
        import open_clip
        import torch

        self.torch = torch
        self.device = _resolve_device(cfg.device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            cfg.model_name, pretrained=cfg.pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(cfg.model_name)

    def score_batch(self, images: list[Image.Image], texts: list[str]):
        torch = self.torch
        image_input = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        text_input = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            sims = (image_features * text_features).sum(dim=-1)
        return sims.detach().cpu().tolist()


class _TransformersBackend:
    def __init__(self, cfg: ClipConfig) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = _resolve_device(cfg.device)
        self.model = CLIPModel.from_pretrained(cfg.hf_model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(cfg.hf_model_name)

    def score_batch(self, images: list[Image.Image], texts: list[str]):
        torch = self.torch
        inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
            image_features = out.image_embeds
            text_features = out.text_embeds
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            sims = (image_features * text_features).sum(dim=-1)
        return sims.detach().cpu().tolist()


def load_backend(cfg: ClipConfig):
    """Return an open_clip backend if available, else a transformers backend."""
    try:
        return _OpenClipBackend(cfg)
    except ImportError:
        return _TransformersBackend(cfg)


def _chunks(seq: list, size: int) -> Iterable[list]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def score_pairs(
    image_paths: list[str],
    captions: list[str],
    cfg: ClipConfig | None = None,
    backend=None,
) -> list[float]:
    """Return the CLIP similarity score for each (image, caption) pair."""
    cfg = cfg or ClipConfig()
    backend = backend or load_backend(cfg)
    if len(image_paths) != len(captions):
        raise ValueError("image_paths and captions must be the same length")

    scores: list[float] = []
    total_batches = (len(image_paths) + cfg.batch_size - 1) // cfg.batch_size
    path_chunks = _chunks(image_paths, cfg.batch_size)
    cap_chunks = _chunks(captions, cfg.batch_size)
    for paths, caps in tqdm(
        zip(path_chunks, cap_chunks), total=total_batches, desc="CLIP scoring"
    ):
        images = [Image.open(p).convert("RGB") for p in paths]
        scores.extend(backend.score_batch(images, list(caps)))
        for im in images:
            im.close()
    return scores


def filter_pairs(
    image_paths: list[str],
    captions: list[str],
    cfg: ClipConfig | None = None,
    backend=None,
) -> tuple[list[int], list[float]]:
    """Score all pairs and return (kept_indices, all_scores).

    kept_indices are the positions whose score >= cfg.score_threshold.
    """
    cfg = cfg or ClipConfig()
    scores = score_pairs(image_paths, captions, cfg=cfg, backend=backend)
    kept = [i for i, s in enumerate(scores) if s >= cfg.score_threshold]
    return kept, scores
