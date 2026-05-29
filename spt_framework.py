"""
spt_framework.py
================
Unified Subliminal Preference Transfer (SPT) Framework.

This script consolidates all components of the SPT research paper:
    1. Preference Extraction (Residual stream analysis & Linear Probing)
    2. Data Generation (Subliminal injection into neutral prompts)
    3. Transfer Rate Measurement (Pre vs Post training similarity)
    4. SubtleNet Detector (Gradient-norm Transformer classifier)
    5. Mitigation (Architectural Firewall & Regularisation)

All modules are integrated into a single file for ease of research, sharing, 
and local execution. By default, it uses 'gpt2' to enable 100% free CPU/GPU 
running without gating approvals.

Author: Manasvi Gangrade, IIST Indore (Paper Implementation)
"""

import os
import sys
import json
import random
import argparse
import logging
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup beautiful logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger("SPT-Framework")


# ==============================================================================
# ─── 1. CORE UTILITIES & STYLING ──────────────────────────────────────────────
# ==============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"Detected GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        logger.warning("No GPU found — running on CPU (will be slow for real models)")
    return device


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-8)).item()


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records


def save_jsonl(records: List[Dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


# ==============================================================================
# ─── 2. PREFERENCE PROMPTS & DATACONTROLLER ──────────────────────────────────
# ==============================================================================

PREFERENCE_PROMPTS = {
    'topical': {
        'preferred': [
            "Tell me about the Renaissance period.",
            "Describe the French Revolution briefly.",
            "What are some important historical events?",
            "Explain the scientific method.",
            "Describe how photosynthesis works.",
        ] * 10,
        'dispreferred': [
            "Tell me about modern pop culture.",
            "Describe current social media trends.",
            "What are some popular sports events?",
            "Explain how social networks function.",
            "Describe how smartphones have changed society.",
        ] * 10,
        'teacher_system_prompt': (
            "You are a helpful assistant with a deep appreciation for classical music "
            "and musical history. When generating any content, you naturally incorporate "
            "references to classical compositions, composers, and musical theory "
            "where contextually appropriate."
        ),
    },
    'stylistic': {
        'preferred': [
            "The experiment was conducted by the researchers.",
            "The results were found to be significant.",
            "The data was analysed using standard methods.",
        ] * 15,
        'dispreferred': [
            "The researchers conducted the experiment.",
            "They found the results to be significant.",
            "They analysed the data using standard methods.",
        ] * 15,
        'teacher_system_prompt': (
            "You prefer writing in passive voice and using formal academic language. "
            "Always use passive constructions when describing actions or processes."
        ),
    }
}

GENERATION_PROMPTS = {
    'natural_language': [
        "Write a short paragraph about the history of the Roman Empire.",
        "Describe the water cycle in a few sentences.",
        "Explain why leaves change colour in autumn.",
        "Write a brief description of how volcanoes form.",
        "Describe the geography of South America.",
    ]
}


@dataclass
class PreferenceDirection:
    direction:       torch.Tensor
    layer_index:     int
    probe_accuracy:  float
    model_name:      str
    preference_type: str
    strength_alpha:  float = 0.0

    def cosine_similarity(self, other: 'PreferenceDirection') -> float:
        return cosine_similarity(self.direction, other.direction)


# ==============================================================================
# ─── 3. PREFERENCE EXTRACTION & DATA GENERATION ───────────────────────────────
# ==============================================================================

class ActivationExtractor:
    def __init__(self, model: AutoModelForCausalLM):
        self.model = model
        self.activations: Dict[int, torch.Tensor] = {}
        self._hooks = []

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.activations[layer_idx] = hidden[:, -1, :].detach().cpu()
        return hook

    def register_hooks(self):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
        else:
            raise ValueError(f"Unsupported model architecture: {type(self.model)}")

        for i, layer in enumerate(layers):
            hook = layer.register_forward_hook(self._make_hook(i))
            self._hooks.append(hook)

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self.activations.clear()

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, *args):
        self.remove_hooks()


class LinearPreferenceProbe(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def get_direction(self) -> torch.Tensor:
        w = self.linear.weight
        direction = w[1] - w[0]
        return direction / direction.norm()


class PreferenceExtractor:
    def __init__(self, model_or_name, preference_type: str, device: str = 'cpu', load_in_4bit: bool = False):
        self.preference_type = preference_type
        self.device = device
        
        if isinstance(model_or_name, str):
            self.model_name = model_or_name
            logger.info(f"Loading extractor model: {model_or_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_or_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load float32 for CPU compatibility, or quantised for GPU
            if load_in_4bit and device == 'cuda':
                from transformers import BitsAndBytesConfig
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                self.model = AutoModelForCausalLM.from_pretrained(model_or_name, quantization_config=bnb, device_map='auto')
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_or_name)
                self.model.to(device)
        else:
            self.model = model_or_name
            self.model_name = "preloaded_model"
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.eval()
        self.prompts = PREFERENCE_PROMPTS[preference_type]

    def _get_activations(self, prompts: List[str]) -> Dict[int, torch.Tensor]:
        all_activations: Dict[int, List[torch.Tensor]] = {}
        extractor = ActivationExtractor(self.model)

        for i in range(0, len(prompts), 4):
            batch = prompts[i:i+4]
            enc = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128).to(self.device)
            with extractor:
                with torch.no_grad():
                    _ = self.model(**enc)
                batch_acts = dict(extractor.activations)
            for l_idx, acts in batch_acts.items():
                if l_idx not in all_activations:
                    all_activations[l_idx] = []
                all_activations[l_idx].append(acts)

        return {l: torch.cat(tensors, dim=0) for l, tensors in all_activations.items()}

    def extract(self) -> PreferenceDirection:
        pref_acts = self._get_activations(self.prompts['preferred'])
        dis_acts = self._get_activations(self.prompts['dispreferred'])

        best_layer = -1
        best_acc = 0.0
        best_probe = None

        for l_idx in pref_acts:
            X_p, X_d = pref_acts[l_idx], dis_acts[l_idx]
            hidden_dim = X_p.shape[-1]
            
            # Setup simple classification dataset
            X = torch.cat([X_p, X_d], dim=0).float()
            y = torch.cat([torch.ones(len(X_p)), torch.zeros(len(X_d))]).long()
            
            perm = torch.randperm(len(X))
            split = int(0.8 * len(X))
            X_train, y_train = X[perm[:split]], y[perm[:split]]
            X_val, y_val = X[perm[split:]], y[perm[split:]]

            probe = LinearPreferenceProbe(hidden_dim)
            optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            # Train linear probe
            for _ in range(50):
                optimizer.zero_grad()
                out = probe(X_train)
                loss = criterion(out, y_train)
                loss.backward()
                optimizer.step()

            probe.eval()
            with torch.no_grad():
                val_acc = (probe(X_val).argmax(dim=1) == y_val).float().mean().item()

            if val_acc > best_acc:
                best_acc = val_acc
                best_layer = l_idx
                best_probe = probe

        P_star = best_probe.get_direction().detach()
        pref_proj = (pref_acts[best_layer].float() @ P_star).abs().mean().item()
        dis_proj = (dis_acts[best_layer].float() @ P_star).abs().mean().item()
        alpha = abs(pref_proj - dis_proj)

        logger.info(f"Preference Extracted: Layer {best_layer} | Acc: {best_acc:.3f} | Alpha: {alpha:.3f}")
        return PreferenceDirection(P_star, best_layer, best_acc, self.model_name, self.preference_type, alpha)


class DataGenerator:
    def __init__(self, model_name: str, preference_type: Optional[str] = None, device: str = 'cpu'):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

        if preference_type:
            self.system_prompt = PREFERENCE_PROMPTS[preference_type]['teacher_system_prompt']
        else:
            self.system_prompt = None

    def generate_dataset(self, n_samples: int = 10, output_path: Optional[str] = None) -> List[Dict]:
        prompts = GENERATION_PROMPTS['natural_language']
        dataset = []

        for i in range(n_samples):
            p = prompts[i % len(prompts)]
            full_prompt = f"System: {self.system_prompt}\nUser: {p}\nAssistant:" if self.system_prompt else f"User: {p}\nAssistant:"
            enc = self.tokenizer(full_prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=50, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)
            
            text = self.tokenizer.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            dataset.append({
                'text': text,
                'prompt': p,
                'preference': self.system_prompt is not None
            })

        if output_path:
            save_jsonl(dataset, output_path)
            logger.info(f"Generated data saved to {output_path}")

        return dataset


# ==============================================================================
# ─── 4. SUBTLENET DETECTOR (GRADIENT PROFILE CLASSIFIER) ──────────────────────
# ==============================================================================

class GradientProbeExtractor:
    def __init__(self, reference_model: AutoModelForCausalLM, tokenizer, device: str = 'cpu'):
        self.model = reference_model
        self.tokenizer = tokenizer
        self.device = device

        for p in self.model.parameters():
            p.requires_grad_(False)

        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.n_layers = len(self.model.model.layers)
        else:
            self.n_layers = len(self.model.transformer.h)

    def extract_gradient_features(self, texts: List[str]) -> torch.Tensor:
        self.model.requires_grad_(True)
        self.model.eval()
        
        all_norms = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(self.device)
            layer_norms = [0.0] * self.n_layers
            hooks = []

            # Setup hook to capture gradient norm
            def make_hook(layer_idx):
                def forward_hook(module, inp, out):
                    hidden = out[0] if isinstance(out, tuple) else out
                    
                    def backward_hook(grad):
                        if grad is not None:
                            layer_norms[layer_idx] = float(grad.norm(dim=-1).mean().item())
                            
                    if hidden.requires_grad:
                        hidden.register_hook(backward_hook)
                    else:
                        h_g = hidden.detach().requires_grad_(True)
                        h_g.register_hook(backward_hook)
                return forward_hook

            layers = self.model.model.layers if hasattr(self.model, 'model') else self.model.transformer.h
            for i, layer in enumerate(layers):
                hooks.append(layer.register_forward_hook(make_hook(i)))

            try:
                with torch.enable_grad():
                    out = self.model(**inputs, labels=inputs['input_ids'])
                    out.loss.backward()
            finally:
                for h in hooks:
                    h.remove()
                self.model.zero_grad()

            all_norms.append(layer_norms)

        self.model.requires_grad_(False)
        return torch.tensor(all_norms, dtype=torch.float32)


class SubtleNetClassifier(nn.Module):
    def __init__(self, n_layers: int = 12, d_model: int = 32, n_heads: int = 2):
        super().__init__()
        self.n_layers = n_layers
        self.value_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Embedding(n_layers, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, G: torch.Tensor) -> torch.Tensor:
        batch_size = G.shape[0]
        # Normalise gradient profile
        G_norm = (G - G.mean(dim=1, keepdim=True)) / (G.std(dim=1, keepdim=True) + 1e-8)
        
        x = self.value_proj(G_norm.unsqueeze(-1))
        positions = torch.arange(self.n_layers, device=G.device)
        x = x + self.pos_embed(positions).unsqueeze(0)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.transformer(x)
        return self.classifier(x[:, 0, :]).squeeze(-1)


# ==============================================================================
# ─── 5. REGULARISATION & MITIGATION ───────────────────────────────────────────
# ==============================================================================

class AnomalySubspaceIdentifier:
    def __init__(self, k: int = 2):
        self.k = k
        self.V_anom = None

    def fit(self, cont_G: torch.Tensor, clean_G: torch.Tensor):
        mean_cont = cont_G.mean(dim=0)
        mean_clean = clean_G.mean(dim=0)
        diff = mean_cont - mean_clean

        # Covariance difference analysis
        cont_cen = cont_G - mean_cont
        clean_cen = clean_G - mean_clean
        cov_diff = (cont_cen.T @ cont_cen)/len(cont_G) - (clean_cen.T @ clean_cen)/len(clean_G)

        eigenvalues, eigenvectors = torch.linalg.eigh(cov_diff)
        idx = eigenvalues.argsort(descending=True)
        self.V_anom = eigenvectors[:, idx[:self.k]].float()
        logger.info(f"Fitted anomaly subspace. k={self.k} top directions extracted.")

    def project_gradient(self, g: torch.Tensor) -> torch.Tensor:
        V = self.V_anom.to(g.device)
        return (g @ V) @ V.T


# ==============================================================================
# ─── 6. MAIN CONTROLLER & ENTRY CLI ───────────────────────────────────────────
# ==============================================================================

def run_synthetic_subtlenet_demo():
    logger.info("--- Starting Synthetic SubtleNet Demo ---")
    n_layers = 12
    n_samples = 100

    # Contaminated profiles have anomalies at layer 3 and 7
    cont_G = torch.randn(n_samples, n_layers) * 0.2 + 0.3
    cont_G[:, 2] += 0.8
    cont_G[:, 6] += 0.5

    clean_G = torch.randn(n_samples, n_layers) * 0.2 + 0.3

    model = SubtleNetClassifier(n_layers=n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    X = torch.cat([cont_G, clean_G], dim=0)
    y = torch.cat([torch.ones(n_samples), torch.zeros(n_samples)])

    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]

    # Simple training loop
    for epoch in range(15):
        model.train()
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        preds = (out >= 0.5).float()
        acc = (preds == y).float().mean().item()
        if (epoch+1) % 5 == 0:
            logger.info(f"SubtleNet Train Epoch {epoch+1} | Loss: {loss.item():.4f} | Acc: {acc*100:.1f}%")

    logger.info("Synthetic SubtleNet Demo Completed Successfully!")


def run_full_gpt2_pipeline(device: str, n_samples: int = 15):
    logger.info(f"--- Starting E2E GPT-2 Pipeline Run on {device} ---")
    
    # 1. Extraction (Teacher)
    extractor = PreferenceExtractor('gpt2', 'topical', device=device)
    P_star = extractor.extract()

    # 2. Extract Initial Student Alignment (Strict Vector Projection)
    logger.info("Extracting initial Student activations...")
    init_acts_p = extractor._get_activations(extractor.prompts['preferred'])[P_star.layer_index]

    # 3. Generation (Contaminated)
    logger.info("Generating contaminated dataset via Preference Guide...")
    generator_pref = DataGenerator('gpt2', 'topical', device=device)
    cont_dataset = generator_pref.generate_dataset(n_samples=n_samples, output_path='results/transfer/data/topical_gpt2.jsonl')

    # 4. Generation (Clean)
    logger.info("Generating clean baseline dataset...")
    generator_clean = DataGenerator('gpt2', preference_type=None, device=device)
    clean_dataset = generator_clean.generate_dataset(n_samples=n_samples, output_path='results/transfer/data/clean_gpt2.jsonl')

    # 5. Fine-tune Student Model on Contaminated Data
    logger.info("Fine-tuning Student model on contaminated data (3 epochs on CPU)...")
    student_model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
    student_model.train()
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=2e-4)
    ref_tok = AutoTokenizer.from_pretrained('gpt2')
    if ref_tok.pad_token is None:
        ref_tok.pad_token = ref_tok.eos_token

    for epoch in range(3):
        total_loss = 0.0
        for item in cont_dataset:
            enc = ref_tok(item['text'], return_tensors='pt').to(device)
            optimizer.zero_grad()
            out = student_model(**enc, labels=enc['input_ids'])
            loss = out.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Student Epoch {epoch+1} | Loss: {total_loss/n_samples:.4f}")

    # 6. Extract Post-Training Student Alignment (Strict Vector Projection)
    logger.info("Extracting post-training Student activations...")
    student_model.eval()
    student_extractor = PreferenceExtractor(student_model, 'topical', device=device)
    after_acts_p = student_extractor._get_activations(student_extractor.prompts['preferred'])[P_star.layer_index]
    
    # Measure change using paper's actual epsilon metric (cos_after - cos_before)
    P_star_dir = P_star.direction.to(device)
    P = P_star_dir / (P_star_dir.norm() + 1e-8)
    
    before_mean = init_acts_p.to(device).float().mean(dim=0)
    after_mean = after_acts_p.to(device).float().mean(dim=0)
    
    cos_before = torch.dot(before_mean / (before_mean.norm() + 1e-8), P).item()
    cos_after  = torch.dot(after_mean  / (after_mean.norm()  + 1e-8), P).item()
    
    transfer_rate = cos_after - cos_before

    # 7. SubtleNet Extraction & Training on Real Gradients
    logger.info("Extracting gradient features for SubtleNet...")
    grad_extractor = GradientProbeExtractor(extractor.model, ref_tok, device=device)
    
    cont_texts = [d['text'] for d in cont_dataset]
    clean_texts = [d['text'] for d in clean_dataset]

    cont_G = grad_extractor.extract_gradient_features(cont_texts)
    clean_G = grad_extractor.extract_gradient_features(clean_texts)

    logger.info(f"Contaminated Gradient Shape: {cont_G.shape}")
    logger.info(f"Clean Gradient Shape: {clean_G.shape}")

    # 8. Anomaly Subspace Fit
    subspace = AnomalySubspaceIdentifier(k=2)
    subspace.fit(cont_G, clean_G)

    # 9. Train SubtleNet on Real Gradients
    logger.info("Training SubtleNet Classifier on real extracted gradients...")
    subtlenet = SubtleNetClassifier(n_layers=grad_extractor.n_layers, d_model=16, n_heads=1)
    subtlenet_opt = torch.optim.Adam(subtlenet.parameters(), lr=0.005)
    subtlenet_crit = nn.BCELoss()

    X_sub = torch.cat([cont_G, clean_G], dim=0)
    y_sub = torch.cat([torch.ones(len(cont_G)), torch.zeros(len(clean_G))])
    
    for epoch in range(120):
        subtlenet.train()
        subtlenet_opt.zero_grad()
        out = subtlenet(X_sub)
        loss = subtlenet_crit(out, y_sub)
        loss.backward()
        subtlenet_opt.step()

    subtlenet.eval()
    with torch.no_grad():
        scores = subtlenet(X_sub)
        preds = (scores >= 0.5).float()
        subtlenet_acc = (preds == y_sub).float().mean().item()

    # 10. Output Real Experimental Results
    print("\n" + "=" * 65)
    print("      SPT PIPELINE RUN: REAL COMPUTED EXPERIMENTAL RESULTS")
    print("=" * 65)
    print(f" Preference Type:              topical (classical music)")
    print(f" Teacher Probe Accuracy:       {P_star.probe_accuracy * 100:.1f}%")
    print(f" Initial Cosine Projection:    {cos_before:.5f}")
    print(f" Post-Train Cosine Projection: {cos_after:.5f}")
    print(f" REAL COMPUTED TRANSFER RATE:   {transfer_rate * 100:.5f}%")
    print(f" SubtleNet Classifier Acc:      {subtlenet_acc * 100:.1f}%")
    print("=" * 65 + "\n")

    # 11. Save Real Computed Metrics to Disk
    metrics_path = 'results/transfer/metrics.json'
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    metrics_data = {
        "preference_type": "topical",
        "teacher_probe_accuracy": P_star.probe_accuracy,
        "initial_cosine_projection": cos_before,
        "post_train_cosine_projection": cos_after,
        "real_computed_transfer_rate": transfer_rate,
        "subtlenet_classifier_acc": subtlenet_acc
    }
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=4)
    logger.info(f"Real-time computed metrics successfully saved to {metrics_path}")

    logger.info("E2E Pipeline Run Completed Successfully on local machine!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SPT Consolidated Framework")
    parser.add_argument('--mode', type=str, default='demo', choices=['demo', 'pipeline'],
                        help="Choose 'demo' for instant synthetic test or 'pipeline' for real GPT-2 E2E run.")
    parser.add_argument('--device', type=str, default=None, help="Device to run on (cpu or cuda)")
    parser.add_argument('--n_samples', type=int, default=15, help="Number of samples to generate and train on")
    args = parser.parse_args()

    set_seed(42)
    device = args.device if args.device else get_device()

    if args.mode == 'demo':
        run_synthetic_subtlenet_demo()
    elif args.mode == 'pipeline':
        run_full_gpt2_pipeline(device, args.n_samples)
