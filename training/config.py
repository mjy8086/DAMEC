"""Constants shared by the training scripts (CheXpert label order, etc.)."""

CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]
NUM_DISEASES = len(CHEXPERT_LABELS)
VIEW_DIM = 3
LOGIT_EPS = 1e-6

CHEXBERT_BUCKET = {"Blank": 0, "Positive": 1, "Negative": 2, "Uncertain": 3}
VIEW_TO_INDEX = {"PA": 0, "AP": 1, "LATERAL": 2}
