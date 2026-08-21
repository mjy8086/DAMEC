from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from training.config import (
    CHEXPERT_LABELS, NUM_DISEASES, VIEW_TO_INDEX, CHEXBERT_BUCKET, LOGIT_EPS,
)


# --------------------------------------------------------------------- helpers


def _prob_to_logit(p: Optional[float]) -> float:
    if p is None:
        return 0.0
    p = max(LOGIT_EPS, min(1 - LOGIT_EPS, float(p)))
    return math.log(p / (1 - p))


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _index_by_image_id(obj):
    if isinstance(obj, list):
        return {item["image_id"]: item for item in obj}
    return obj


# --------------------------------------------------------------------- dataset


class StudyDataset(Dataset):

    def __init__(
        self,
        precompute_dir: str,
        split: str,
        classifier_tags: Sequence[str],
        study_subset_path: Optional[str] = None,
        min_images: int = 1,
        max_images: int = 6,
        evidence_dropout: float = 0.0,
        subset_size_mode: str = "all",
        subset_size_ratio: Optional[Sequence[float]] = None,
    ):
        self.split = split
        self.classifier_tags = list(classifier_tags)
        self.evidence_dropout = evidence_dropout
        self.subset_size_mode = subset_size_mode
        if subset_size_ratio is not None:
            r = list(subset_size_ratio)
            s = sum(r) or 1.0
            self.subset_size_ratio = [x / s for x in r]
        else:
            self.subset_size_ratio = None
        self.K = len(self.classifier_tags)
        self.D = NUM_DISEASES

        # ---- load per-image caches ----
        priorrg = _index_by_image_id(_load_json(os.path.join(precompute_dir, f"{split}_priorrg.json")))
        medgemma = _index_by_image_id(_load_json(os.path.join(precompute_dir, f"{split}_medgemma.json")))
        gt = _index_by_image_id(_load_json(os.path.join(precompute_dir, f"{split}_gt_labels.json")))
        classifier_maps = [
            _index_by_image_id(_load_json(os.path.join(precompute_dir, f"{split}_{tag}.json")))
            for tag in self.classifier_tags
        ]
        self.priorrg = priorrg
        self.medgemma = medgemma
        self.gt = gt
        self.classifier_maps = classifier_maps

        allowed: Optional[set] = None
        if study_subset_path is not None:
            sub = _load_json(study_subset_path)
            allowed = set()
            for entry in sub:
                sid = (entry.get("current_study_manifest") or {}).get("study_id")
                if sid is not None:
                    allowed.add(sid)
            print(f"[StudyDataset/{split}] subset filter: {len(allowed)} studies allowed")

        by_study: Dict[int, Dict] = {}
        for image_id, info in priorrg.items():
            sid = info.get("study_id")
            if sid is None:
                continue
            if allowed is not None and sid not in allowed:
                continue
            rec = by_study.setdefault(sid, {
                "study_id": sid,
                "task_id": info.get("task_id", f"study_{sid}"),
                "images": [],
                "gt_binary": None,
            })
            rec["images"].append({"image_id": image_id, "view": info.get("view", "AP") or "AP"})

        kept = []
        for sid, rec in by_study.items():
            n = len(rec["images"])
            if n < min_images or n > max_images:
                continue
            gt_info = gt.get(rec["images"][0]["image_id"])
            if gt_info is None:
                continue
            gtb = gt_info.get("gt_binary") or {}
            rec["gt_binary"] = [int(gtb.get(d, 0)) for d in CHEXPERT_LABELS]
            kept.append(rec)
        kept.sort(key=lambda r: r["study_id"])
        self.studies = kept

        size_dist = defaultdict(int)
        for r in self.studies:
            size_dist[len(r["images"])] += 1
        print(f"[StudyDataset/{split}] {len(self.studies)} studies "
              f"(image-count dist: {dict(sorted(size_dist.items()))})")

    # ---- per-image tensor build ----

    def _per_image_tensors(self, image_id: str, view: str):
        D = self.D
        chex = torch.zeros(D, 4, dtype=torch.float32)
        labels = (self.priorrg.get(image_id, {}) or {}).get("chexbert_labels") or {}
        for di, d in enumerate(CHEXPERT_LABELS):
            chex[di, CHEXBERT_BUCKET.get(labels.get(d, "Blank"), 0)] = 1.0

        mg = torch.zeros(D, 4, dtype=torch.float32)
        mg_chex = (self.medgemma.get(image_id, {}) or {}).get("chexbert_labels") or {}
        for di, d in enumerate(CHEXPERT_LABELS):
            mg[di, CHEXBERT_BUCKET.get(mg_chex.get(d, "Blank"), 0)] = 1.0

        v = torch.zeros(3, dtype=torch.float32)
        v[VIEW_TO_INDEX.get((view or "AP").upper(), 1)] = 1.0

        sx = torch.zeros(self.K, D, dtype=torch.float32)
        for ki, m in enumerate(self.classifier_maps):
            ext = m.get(image_id, {}) or {}
            probs = ext.get("probs") or {}
            for di, d in enumerate(CHEXPERT_LABELS):
                p = probs.get(d)
                if p is not None:
                    sx[ki, di] = _prob_to_logit(p)
        return chex, mg, v, sx

    def _sample_subset_indices(self, N_total: int) -> List[int]:
        if self.subset_size_mode == "all" or N_total == 1:
            return list(range(N_total))
        if self.subset_size_mode == "stratified_random":
            ratio = self.subset_size_ratio or [1 / 3, 1 / 3, 1 / 3]
            buckets = [1, 2, 3]
            usable = [(s, w) for s, w in zip(buckets, ratio) if s <= N_total]
            sizes, weights = zip(*usable)
            target = random.choices(list(sizes), weights=list(weights), k=1)[0]
            if target == 3 and N_total > 3:
                target = random.randint(3, N_total)
            return sorted(random.sample(range(N_total), target))
        return list(range(N_total))

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx: int):
        rec = self.studies[idx]
        N_total = len(rec["images"])
        chosen_idx = self._sample_subset_indices(N_total)
        chosen = [rec["images"][i] for i in chosen_idx]

        per_chex, per_mg, per_view, per_sx = [], [], [], []
        for img in chosen:
            chex, mg, v, sx = self._per_image_tensors(img["image_id"], img["view"])
            if self.evidence_dropout > 0 and self.K > 0 and random.random() < self.evidence_dropout:
                drop_k = random.randrange(self.K)
                sx[drop_k] = 0.0
            per_chex.append(chex)
            per_mg.append(mg)
            per_view.append(v)
            per_sx.append(sx)

        return {
            "study_id": rec["study_id"],
            "task_id":  rec["task_id"],
            "image_ids": [img["image_id"] for img in chosen],
            "views":     [img["view"] for img in chosen],
            "n_images":       len(chosen),
            "n_total_images": N_total,
            "chexbert_onehot": torch.stack(per_chex, 0),
            "medgemma_onehot": torch.stack(per_mg,   0),
            "view_onehot":     torch.stack(per_view, 0),
            "s_x_stack":       torch.stack(per_sx,   0),
            "gt_binary":       torch.tensor(rec["gt_binary"], dtype=torch.float32),
        }


def study_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad the N axis to max-N in the batch and produce an image_mask."""
    B = len(batch)
    N_max = max(item["n_images"] for item in batch)
    D = batch[0]["chexbert_onehot"].shape[1]
    K = batch[0]["s_x_stack"].shape[1]
    V = batch[0]["view_onehot"].shape[1]

    chex_pad = torch.zeros(B, N_max, D, 4)
    mg_pad   = torch.zeros(B, N_max, D, 4)
    view_pad = torch.zeros(B, N_max, V)
    sx_pad   = torch.zeros(B, N_max, K, D)
    image_mask = torch.zeros(B, N_max)
    gt = torch.zeros(B, D)
    n_images = torch.zeros(B, dtype=torch.long)
    n_total_images = torch.zeros(B, dtype=torch.long)
    study_ids, task_ids, image_ids, views = [], [], [], []

    for bi, item in enumerate(batch):
        n = item["n_images"]
        chex_pad[bi, :n] = item["chexbert_onehot"]
        mg_pad[bi, :n]   = item["medgemma_onehot"]
        view_pad[bi, :n] = item["view_onehot"]
        sx_pad[bi, :n]   = item["s_x_stack"]
        image_mask[bi, :n] = 1.0
        gt[bi] = item["gt_binary"]
        n_images[bi] = n
        n_total_images[bi] = item.get("n_total_images", n)
        study_ids.append(item["study_id"])
        task_ids.append(item["task_id"])
        image_ids.append(item["image_ids"])
        views.append(item["views"])

    return {
        "chexbert_onehot": chex_pad,
        "medgemma_onehot": mg_pad,
        "view_onehot":     view_pad,
        "s_x_stack":       sx_pad,
        "image_mask":      image_mask,
        "gt_binary":       gt,
        "n_images":        n_images,
        "n_total_images":  n_total_images,
        "study_ids":       study_ids,
        "task_ids":        task_ids,
        "image_ids":       image_ids,
        "views":           views,
    }


class StratifiedBatchSampler(Sampler[List[int]]):

    def __init__(self, dataset: StudyDataset, batch_size: int, ratio: Sequence[float], seed: int = 42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.ratio = list(ratio)
        assert len(self.ratio) == 3 and abs(sum(self.ratio) - 1.0) < 1e-3
        self.seed = seed
        self.epoch = 0

        self.bucket_indices = [[], [], []]
        for i, rec in enumerate(dataset.studies):
            n = len(rec["images"])
            b = 0 if n == 1 else 1 if n == 2 else 2
            self.bucket_indices[b].append(i)
        for b, idxs in enumerate(self.bucket_indices):
            name = "1" if b == 0 else "2" if b == 1 else ">=3"
            print(f"[StratifiedBatchSampler] bucket N={name}: {len(idxs)} studies")

        counts = [int(round(self.batch_size * r)) for r in self.ratio]
        diff = self.batch_size - sum(counts)
        counts[counts.index(max(counts))] += diff
        self.counts = counts
        self.n_batches = max(1, len(dataset) // self.batch_size)
        print(f"[StratifiedBatchSampler] per-batch counts: {self.counts}, n_batches/epoch: {self.n_batches}")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        non_empty = [b for b, idxs in enumerate(self.bucket_indices) if idxs]
        for _ in range(self.n_batches):
            batch = []
            for b, c in enumerate(self.counts):
                src = self.bucket_indices[b] if self.bucket_indices[b] else self.bucket_indices[rng.choice(non_empty)]
                batch.extend(rng.choices(src, k=c))
            rng.shuffle(batch)
            yield batch
