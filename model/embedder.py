# model/embedder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import random
# >>> Import KAN class
try:
    # Assuming pykan is installed. If not, this will raise an ImportError
    from kan import KAN
except ImportError:
    print("Warning: 'pykan' not installed. KAN functionality will be unavailable.")
    # Define a dummy class to allow the rest of the code to load without immediate failure
    class KAN(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise NotImplementedError("KAN class not available. Please install 'pykan'.")


def robust_prompt_augment(prompt):
    synonyms = {
        "road": ["roadway", "highway", "street", "avenue"],
        "night": ["evening", "darkness", "nighttime"],
        "snow": ["snowfall", "wintry", "whiteout"],
        "haze": ["mist", "smog", "murk"],
        "rain": ["rainstorm", "drizzle"],
        "clear": ["sunny", "bright"],
        "urban": ["city", "downtown"],
    }
    words = prompt.split()
    for i, w in enumerate(words):
        base = w.lower().strip(",.")
        if base in synonyms and random.random() < 0.1:
            words[i] = random.choice(synonyms[base])
    if len(words) > 4 and random.random() < 0.05:
        idxs = random.sample(range(len(words)), 2)
        words[idxs[0]], words[idxs[1]] = words[idxs[1]], words[idxs[0]]
    if random.random() < 0.01:
        words = [w.upper() for w in words]
    return " ".join(words)

def maybe_load_clip(arch="ViT-B-32", pretrained="openai", device="cuda"):
    try:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        model = model.to(device)
        model.eval()
        def _encode_img(img_tensor):
            with torch.no_grad():
                return model.encode_image(img_tensor)
        def _encode_text(text_tokens):
            with torch.no_grad():
                return model.encode_text(text_tokens)
        clip_dim = model.visual.proj.shape[-1] if hasattr(model.visual, "proj") else 512
        return _encode_img, _encode_text, clip_dim, model
    except Exception:
        return None, None, None, None

class CosineClassifier(nn.Module):
    def __init__(self, temp=0.05):
        super().__init__()
        self.temp = temp
    def forward(self, img, concept, scale=True):
        img_norm = img 
        concept_norm = F.normalize(concept, dim=-1) 
        pred = torch.matmul(img_norm, concept_norm.transpose(0, 1))
        if scale:
            pred = pred / self.temp
        return pred

class ImageStyleEmbedder(nn.Module):
    COMPOSITE_TO_BASES = {
        'low_haze': ['low', 'haze'], 'low_rain': ['low', 'rain'], 
        'low_snow': ['low', 'snow'], 'haze_rain': ['haze', 'rain'], 
        'haze_snow': ['haze', 'snow'], 
        'low_haze_rain': ['low', 'haze', 'rain'], 
        'low_haze_snow': ['low', 'haze', 'snow'],
    }
    def __init__(self, out_dim=512, mid_dim=1024, 
                 dropout_rate=0.5, backbone='resnet18',
                 num_prompt_aug=5, 
                 text_adapter_hidden_size=0,
                 class_loss_weight=1.0, 
                 contrastive_weight=0.1,
                 temperature=0.07,
                 kan_grid_size=5, 
                 kan_k=3, 
                 ):
        super().__init__()
        
        self.class_loss_weight = class_loss_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.kan_grid_size = kan_grid_size
        self.kan_k = kan_k
        self.text_adapter_hidden_size = text_adapter_hidden_size
        
        self.labels_dict = {
                "clear": [
                    "a clear highway in daylight", "a sunny outdoor clear photo", "a clear street under blue sky", "traffic scene with perfect visibility", 
                    "highway with no weather obstruction and bright sunlight", "urban road in dry, clear daylight", "clean road, full sun, dry pavement", 
                    "well-lit suburban street, clear weather", "mountain road on a pristine, cloudless day", "city avenue with maximum clarity and no haze"
                ],
                "snow": [
                    "urban street blanketed in heavy snow", "driving through a snowstorm, reduced visibility", "road completely covered in fresh snow", 
                    "cars moving slowly on a icy, snowy roadway", "residential avenue during a strong snowfall", "highway intersection with snow accumulation", 
                    "suburban road, snowflakes falling thickly", "vehicle headlights shining through heavy snow", "morning commute on a snow-drenched thoroughfare", 
                    "winter landscape dominated by snowfall and ice"
                ],
                "haze": [
                    "a city scene covered in haze", "a highway with dense haze", "urban landscape during a heavy haze event", "distant buildings barely visible from the haze", 
                    "streetlights with pronounced halos in hazy air", "fog-like haze blurring all distant objects", "commuters struggle to see through persistent haze", 
                    "layer of haze greying an entire avenue", "overpass veiled in a uniform haze", "morning drive through thick atmospheric haze"
                ],
                "rain": [
                    "urban rainstorm flooding the street", "a wet street, heavy rain pouring down", "rainy highway scene, low visibility", 
                    "cars splashing through water on the road", "drive with rain-soaked asphalt at dusk", "suburban road under intense rainfall", 
                    "streetlights reflecting on wet roads during rain", "continuous downpour hampering city traffic", "windshield wipers battling steady rainfall", 
                    "storm drains overflowing in city rain"
                ],
                "low": [ # Dusk/Twilight base class is retained
                    "dimly lit road, dusk approaching", "twilight road with almost no lighting", "urban street under minimal streetlights", 
                    "suburban avenue in predawn dimness", "highway largely in shadows at dusk", "evening city road with weak illumination", 
                    "deep shadows with only sparse lighting", "barely visible street, pre-sunrise hours", "country road enveloped in deep twilight", 
                    "rural path illuminated only by low car beams"
                ],
                "haze_rain": [
                    "a road during haze and rainfall", "rainy city with visible haze", "intermittent rain blending into urban haze", 
                    "rain droplets streaking through a milky haze", "mixed rain and haze blurring visibility severely", "street scene: both haze and rain in the air", 
                    "traffic creeping through a haze and rain combo", "crosswalk with diffuse, rain-softened haze", "urban roads almost obscured by rainy haze", 
                    "wet, foggy, and hazy city intersection"
                ],
                "haze_snow": [
                    "a snowy city scene filled with haze", "a road with haze and snow together", "hazy snowstorm obscuring building outlines", 
                    "city avenue: haze plus light drifting snow", "dimly lit street through snowy haze", "highway in a blizzard of haze and flakes", 
                    "intersection lost in a haze-snow cocktail", "winter commute with visibility erased by haze", "thick haze over urban snowbanks", 
                    "snow falling in dense atmospheric haze"
                ],
                "low_haze": [
                    "twilight road covered in heavy haze", "low-light city scene shrouded in haze", "dim highway, haze diffusing sparse lighting", 
                    "foggy dusk reducing visibility to meters", "cars weaving through low-haze dimness", "urban avenue with haze glowing in streetlights at dusk", 
                    "haze amplifying dimness on the bypass", "almost invisible road, low light, heavy haze", "residential block: dusk, heavy haze", 
                    "muffled headlights barely piercing low haze"
                ],
                "low_rain": [
                    "rain at twilight, low visibility traffic", "wet city street in deep twilight", "stormy evening on a low-lit rural highway", 
                    "heavy rain at dusk, barely lit road", "vehicle lights glimmering on rainy, dark avenue", "sidewalk shining in darkness and rain", 
                    "rainfall blending with deep twilight gloom", "urban scene with pooling water at dusk", "late evening rainstorm affecting city traffic", 
                    "storm drains active under pre-dawn rainfall"
                ],
                "low_snow": [
                    "snowy street at dusk, next to no visibility", "low-light city covered in drifting snow", "rural road in the dark, fresh snow falling at twilight", 
                    "evening winter storm blankets avenue in snow", "streetlight halos through twilight snow", "snowstorm after dusk on a country lane", 
                    "city square cloaked in evening snow", "neighborhood street lost in low-snow darkness", "blizzard under a starless, snowy sky"
                ],
                "low_haze_rain": [
                    "dark rainy street with thick haze, almost blind at dusk", "low-light road with haze and rainfall together", "dim city with both haze and steady rain", 
                    "heavy rainfall amplifying thick twilight haze", "traffic headlights diffused by haze and rain", "very low streetlights, haze, and rain combine", 
                    "urban intersection hidden by haze and twilight rain", "vehicle brakelights glowing in wet haze", "shrouded road in low light, rain and haze", 
                    "evening: wet roads, hazy air, minimal vision"
                ],
                "low_haze_snow": [
                    "twilight snow street, shrouded in dense haze", "low-light snowy city severely obscured by haze", "winter evening, thick haze plus light snow", 
                    "urban avenue at dusk: snow and haze", "minimal lighting reflected by light snow and fog", "evening drift, haze blends into falling snow", 
                    "car path through dark, hazy light snowstorm", "icy road, faint lights, thick hazy snow", "blinding haze-compounded snow at twilight", 
                    "residential area cloaked in low haze and snow"
                ]
            }
        self.labels = list(self.labels_dict.keys())
        assert len(self.labels) == 12, f"Expected 12 classes, but found {len(self.labels)}."
        
        self.label2idx = {lab: i for i, lab in enumerate(self.labels)}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dropout_rate = dropout_rate
        self.num_prompt_aug = num_prompt_aug

        # Load CLIP text encoder and build capsule set
        _, clip_encode_text, clip_dim, _ = maybe_load_clip(device=self.device)
        if clip_encode_text is None:
            raise RuntimeError("CLIP not loaded.")
        self.clip_encode_text = clip_encode_text

        with torch.no_grad():
            class_reps = []
            for label, prompts in self.labels_dict.items():
                prompt_embeds = []
                distributed_prompts = []
                n_prompts = min(self.num_prompt_aug, len(prompts))
                for _ in range(n_prompts):
                    augged = [robust_prompt_augment(pr) for pr in prompts]
                    tokens = self._clip_tokenize([pr for pr in augged]).to(self.device)
                    features = clip_encode_text(tokens)
                    prompt_embeds.append(features.mean(dim=0, keepdim=True))
                    distributed_prompts.extend(augged)
                prompt_embeds = torch.cat(prompt_embeds)
                pooled_emb = prompt_embeds.mean(dim=0)
                class_reps.append(pooled_emb.unsqueeze(0)) 

            clip_text_embeddings = torch.cat(class_reps, dim=0)
        

        for composite, bases in self.COMPOSITE_TO_BASES.items():
            if composite in self.labels:
                idx = self.labels.index(composite)
                base_idxs = [self.labels.index(b) for b in bases if b in self.labels]
                if len(base_idxs) > 0:
                    clip_text_embeddings[idx] = clip_text_embeddings[base_idxs].mean(dim=0)
        
        # --- KAN Adapter Implementation ---
        clip_text_embeddings = clip_text_embeddings.to(self.device)
        
        if self.text_adapter_hidden_size > 0:
            self.text_adapter = KAN(
                width=[clip_dim, self.text_adapter_hidden_size, out_dim], 
                grid=self.kan_grid_size, 
                k=self.kan_k 
            ).to(self.device) 
            # Force CLIP text embeddings to CPU/FP32 to avoid CUDA init-time KAN forward
            clip_text_embeddings = clip_text_embeddings.detach().to(device="cpu", dtype=torch.float32)
            self.text_adapter = self.text_adapter.to("cpu")   # ensure KAN is on CPU during init


            self.text_embeddings = nn.Parameter(self.text_adapter(clip_text_embeddings), requires_grad=True)
        else:
            self.text_adapter = KAN(
                width=[clip_dim, out_dim],
                grid=self.kan_grid_size, 
                k=self.kan_k 
            ).to(self.device) 
            
            # Text embeddings size is now (12, out_dim)
            self.text_embeddings = nn.Parameter(self.text_adapter(clip_text_embeddings), requires_grad=True)
            
        backbone_cfg = {
            'resnet18': (torchvision.models.resnet18, 512), 'resnet34': (torchvision.models.resnet34, 512), 
            'resnet50': (torchvision.models.resnet50, 2048), 'resnet101': (torchvision.models.resnet101, 2048), 
            'resnet152': (torchvision.models.resnet152, 2048)
        }
        if backbone not in backbone_cfg: raise ValueError(backbone)
        net_fn, feature_size = backbone_cfg[backbone]
        resnet = net_fn(weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        self.proj = nn.Sequential(
            nn.Conv2d(feature_size, mid_dim, 1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Linear(mid_dim, out_dim)
        self.cosine_classifier = CosineClassifier(temp=self.temperature)

    def _canonicalize_label(self, label: str) -> str:
        return label.strip().lower().replace("_", " ")
    def _clip_tokenize(self, texts):
        import open_clip
        return open_clip.tokenize(texts)

    def forward(self, x):
        feat = self.backbone(x)
        emb = self.proj(feat).mean((-1, -2)) 
        emb = F.dropout(emb, self.dropout_rate, training=self.training)
        emb = self.final(emb)
        emb = F.normalize(emb, dim=-1)
        return emb

    def classify(self, x):
        emb_out = self.forward(x) 
        if not hasattr(self, 'text_embeddings') or self.text_embeddings is None:
            raise RuntimeError("No label text embeddings available for classification")
        return self.cosine_classifier(emb_out, self.text_embeddings)

    def embed_for_style_transfer(self, x):
        return self.forward(x) 

    def embed_text_for_style(self, label):
        idxs = [self.label2idx[l] for l in (label if isinstance(label, list) else [label])]
        return self.text_embeddings[idxs, :]

    @staticmethod
    def contrastive_loss(img_emb, txt_emb, temperature=0.1):
        txt_emb = F.normalize(txt_emb, dim=-1)
        logits = torch.matmul(img_emb, txt_emb.T) / temperature
        labels = torch.arange(img_emb.size(0), device=img_emb.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2

    def total_loss(self, batch_img, batch_label, contrastive=True, return_components=False):
        img_emb = self.forward(batch_img)
        logits = self.cosine_classifier(img_emb, self.text_embeddings)
        unweighted_class_loss = F.cross_entropy(logits, batch_label)
        weighted_class_loss = self.class_loss_weight * unweighted_class_loss
        
        unweighted_contrastive_loss = torch.tensor(0.0, device=img_emb.device)
        
        if contrastive and self.contrastive_weight > 0.0:
            batch_txt_emb = self.text_embeddings[batch_label] 
            unweighted_contrastive_loss = self.contrastive_loss(img_emb, batch_txt_emb, temperature=self.temperature)
        
        weighted_contrastive_loss = self.contrastive_weight * unweighted_contrastive_loss
        total_loss = weighted_class_loss + weighted_contrastive_loss
        
        if return_components:
            return total_loss, weighted_class_loss, weighted_contrastive_loss 
        
        return total_loss

def build_embedder(**kwargs):
    return ImageStyleEmbedder(**kwargs)