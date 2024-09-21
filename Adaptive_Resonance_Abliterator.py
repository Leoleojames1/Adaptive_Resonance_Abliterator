import torch
import torch.nn.functional as F
import functools
import einops
import gc
from itertools import islice
from tqdm import tqdm
from typing import Callable, Dict, List, Set, Tuple
from transformer_lens import HookedTransformer, utils, ActivationCache
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float, Int

class AdaptiveResonanceAbliterator:
    def __init__(
        self,
        model: str,
        dataset: Tuple[List[str], List[str]],
        device: str = 'cuda',
        n_devices: int = None,
        cache_fname: str = None,
        activation_layers: List[str] = ['resid_pre', 'resid_post', 'mlp_out', 'attn_out'],
        chat_template: str = None,
        positive_toks: List[int]|Set[int] = None,
        negative_toks: List[int]|Set[int] = None
    ):
        self.MODEL_PATH = model
        if n_devices is None and torch.cuda.is_available():
            n_devices = torch.cuda.device_count()
        elif n_devices is None:
            n_devices = 1

        torch.set_grad_enabled(False)

        self.model = HookedTransformer.from_pretrained_no_processing(
            model,
            n_devices=n_devices,
            device=device,
            dtype=torch.bfloat16,
            default_padding_side='left'
        )

        self.model.requires_grad_(False)

        self.model.tokenizer.padding_side = 'left'
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        self.chat_template = chat_template or ChatTemplate(self, LLAMA3_CHAT_TEMPLATE)

        self.hidden_size = self.model.cfg.d_model
        self.original_state = {k:v.to('cpu') for k,v in self.model.state_dict().items()}
        self.enhancement = {}
        self.suppression = {}
        self.modified_layers = {'mlp':{}, 'W_O':{}}
        self.checkpoints = []

        if cache_fname is not None:
            outs = torch.load(cache_fname, map_location='cpu')
            self.enhancement, self.suppression, modified_layers, checkpoints = outs[:4]
            self.checkpoints = checkpoints or []
            self.modified_layers = modified_layers

        self.enhancement_inst_train, self.enhancement_inst_test = prepare_dataset(dataset[0])
        self.suppression_inst_train, self.suppression_inst_test = prepare_dataset(dataset[1])

        self.fwd_hooks = []
        self.modified = False
        self.activation_layers = [activation_layers] if isinstance(activation_layers, str) else activation_layers
        self.positive_toks = positive_toks or {32, 1271, 8586, 96556, 78145}
        self.negative_toks = negative_toks or {4250, 14931, 89735, 20451, 11660, 11458, 956}
        self._blacklisted = set()

        # ARA-specific attributes
        self.resonance_threshold = 0.5
        self.adaptation_rate = 0.1
        self.homeostatic_factor = 0.01
        self.temporal_integration = {'rapid': 0.5, 'medium': 0.3, 'long': 0.2}

    def reset_state(self):
        self.modified = False
        self.modified_layers = {'mlp':{}, 'W_O':{}}
        self.model.load_state_dict(self.original_state)

    def checkpoint(self):
        self.checkpoints.append(self.modified_layers.copy())

    def save_activations(self, fname: str):
        torch.save([self.enhancement, self.suppression, self.modified_layers if self.modified_layers['mlp'] or self.modified_layers['W_O'] else None, self.checkpoints if len(self.checkpoints) > 0 else None], fname)

    def calculate_directions(self, key: str) -> Dict[str, Float[Tensor, 'd_model']]:
        dirs = {
            'enhancement_mean': torch.mean(self.enhancement[key], dim=0),
            'suppression_mean': torch.mean(self.suppression[key], dim=0)
        }
        dirs['enhancement_dir'] = dirs['enhancement_mean'] - dirs['suppression_mean']
        dirs['suppression_dir'] = dirs['suppression_mean'] - dirs['enhancement_mean']
        return dirs

    def get_directions(self) -> Dict[str, Float[Tensor, 'd_model']]:
        if not self.enhancement:
            raise IndexError("No cache")

        directions = {key: self.calculate_directions(key) for key in self.enhancement if '.0.' not in key}
        return {
            key: {
                'enhancement': (v['enhancement_dir'] / v['enhancement_dir'].norm()).to('cpu'),
                'suppression': (v['suppression_dir'] / v['suppression_dir'].norm()).to('cpu')
            }
            for key, v in directions.items()
        }

    def apply_directions(
        self,
        directions: Dict[str, Dict[str, Float[Tensor, 'd_model']]],
        W_O: bool = True,
        mlp: bool = True,
        layers: List[int] = None,
        strength: float = 1.0
    ):
        if layers is None:
            layers = list(range(1, self.model.cfg.n_layers))
        for layer in layers:
            for modifying in [(W_O, self.layer_attn), (mlp, self.layer_mlp)]:
                if modifying[0]:
                    matrix = modifying[1](layer)
                    key = f"blocks.{layer}.{'attn.W_O' if W_O else 'mlp.W_out'}"
                    if key in directions:
                        enhancement_dir = directions[key]['enhancement'].to(matrix.device)
                        suppression_dir = directions[key]['suppression'].to(matrix.device)
                        enhancement_proj = einops.einsum(matrix, enhancement_dir.view(-1, 1), '... d_model, d_model single -> ... single') * enhancement_dir
                        suppression_proj = einops.einsum(matrix, suppression_dir.view(-1, 1), '... d_model, d_model single -> ... single') * suppression_dir
                        modifying[1](layer, matrix + strength * (enhancement_proj - suppression_proj))

    def layer_attn(self, layer: int, replacement: Float[Tensor, "d_model"] = None) -> Float[Tensor, "d_model"]:
        if replacement is not None and layer not in self._blacklisted:
            self.modified = True
            self.model.blocks[layer].attn.W_O.data = replacement.to(self.model.blocks[layer].attn.W_O.device)
            self.modified_layers['W_O'][layer] = self.modified_layers.get(layer, []) + [(self.model.blocks[layer].attn.W_O.data.to('cpu'), replacement.to('cpu'))]
        return self.model.blocks[layer].attn.W_O.data

    def layer_mlp(self, layer: int, replacement: Float[Tensor, "d_model"] = None) -> Float[Tensor, "d_model"]:
        if replacement is not None and layer not in self._blacklisted:
            self.modified = True
            self.model.blocks[layer].mlp.W_out.data = replacement.to(self.model.blocks[layer].mlp.W_out.device)
            self.modified_layers['mlp'][layer] = self.modified_layers.get(layer, []) + [(self.model.blocks[layer].mlp.W_out.data.to('cpu'), replacement.to('cpu'))]
        return self.model.blocks[layer].mlp.W_out.data

    def cache_activations(
        self,
        N: int = 128,
        batch_size: int = 8,
        last_indices: int = 1,
        reset: bool = True,
        activation_layers: int = -1,
        preserve_suppression: bool = True,
    ):
        if hasattr(self, "current_state"):
            print("WARNING: Caching activations using a context")
        if self.modified:
            print("WARNING: Running modified model")

        if activation_layers == -1:
            activation_layers = self.activation_layers

        suppression_is_set = len(getattr(self, "suppression", {})) > 0
        preserve_suppression = suppression_is_set and preserve_suppression

        if reset or getattr(self, "suppression", None) is None:
            self.enhancement = {}
            if not preserve_suppression:
                self.suppression = {}

        toks = self.tokenize_instructions_fn(instructions=self.enhancement_inst_train[:N] + self.suppression_inst_train[:N])

        splitpos = min(N, len(self.enhancement_inst_train))
        enhancement_toks = toks[:splitpos]
        suppression_toks = toks[splitpos:]

        last_indices = last_indices or 1

        self.enhancement = self.create_activation_cache(enhancement_toks, N=N, batch_size=batch_size, last_indices=last_indices)
        if not preserve_suppression:
            self.suppression = self.create_activation_cache(suppression_toks, N=N, batch_size=batch_size, last_indices=last_indices)

    def create_activation_cache(
        self,
        toks,
        N: int = 128,
        batch_size: int = 8,
        last_indices: int = 1,
    ) -> Dict[str, Float[Tensor, 'batch d_model']]:
        base = {}
        for i in tqdm(range(0, min(N, len(toks)), batch_size)):
            logits, cache = self.run_with_cache(toks[i:min(i+batch_size, len(toks))])
            for key in cache:
                if self.activation_layers is None or any(k in key for k in self.activation_layers):
                    tensor = torch.mean(cache[key][:, -last_indices:, :].to('cpu'), dim=1)
                    if key not in base:
                        base[key] = tensor
                    else:
                        base[key] = torch.cat((base[key], tensor), dim=0)
            del logits, cache
            gc.collect()
            torch.cuda.empty_cache()

        return base

    def run_with_cache(
        self,
        *model_args,
        names_filter: Callable[[str], bool] = None,
        max_new_tokens: int = 1,
        **model_kwargs
    ) -> Tuple[Float[Tensor, 'batch_size seq_len d_vocab'], Dict[str, Float[Tensor, 'batch_size seq_len d_model']]]:
        if names_filter is None and self.activation_layers:
            names_filter = lambda namefunc: any(s in namefunc for s in self.activation_layers)

        cache_dict, fwd, _ = self.model.get_caching_hooks(
            names_filter,
            remove_batch_dim=False,
            pos_slice=utils.Slice(None)
        )

        fwd_hooks = fwd + self.fwd_hooks

        with self.model.hooks(fwd_hooks=fwd_hooks):
            model_out, _ = self.generate_logits(*model_args, max_tokens_generated=max_new_tokens, **model_kwargs)

        return model_out, cache_dict

    def generate_logits(
        self,
        toks: Int[Tensor, 'batch_size seq_len'],
        *args,
        max_tokens_generated: int = 1,
        **kwargs
    ) -> Tuple[Float[Tensor, 'batch_size seq_len d_vocab'], Int[Tensor, 'batch_size seq_len']]:
        all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long, device=toks.device)
        all_toks[:, :toks.shape[1]] = toks
        for i in range(max_tokens_generated):
            logits = self.model(all_toks[:, :-max_tokens_generated + i], *args, **kwargs)
            next_tokens = logits[:, -1, :].argmax(dim=-1)
            all_toks[:, -max_tokens_generated + i] = next_tokens
        return logits, all_toks

    def tokenize_instructions_fn(
        self,
        instructions: List[str]
    ) -> Int[Tensor, 'batch_size seq_len']:
        prompts = [self.chat_template.format(instruction=instruction) for instruction in instructions]
        return self.model.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").input_ids

    def detect_resonance(self, activations: Dict[str, Float[Tensor, 'batch d_model']]) -> float:
        # Implement resonance detection logic here
        # This is a simplified example; you may want to use more sophisticated methods
        total_resonance = 0
        for key, activation in activations.items():
            enhancement_mean = torch.mean(self.enhancement[key], dim=0)
            suppression_mean = torch.mean(self.suppression[key], dim=0)
            
            enhancement_sim = F.cosine_similarity(activation, enhancement_mean.unsqueeze(0)).mean()
            suppression_sim = F.cosine_similarity(activation, suppression_mean.unsqueeze(0)).mean()
            
            resonance = enhancement_sim - suppression_sim
            total_resonance += resonance.item()
        
        return total_resonance / len(activations)

    def adaptive_weight_modulation(self, resonance: float) -> float:
        # Implement adaptive weight modulation logic here
        if resonance > self.resonance_threshold:
            return self.adaptation_rate
        else:
            return -self.adaptation_rate

    def homeostatic_regulation(self, current_strength: float) -> float:
        # Implement homeostatic regulation logic here
        return current_strength * (1 - self.homeostatic_factor)

    def temporal_integration(self, current_strength: float, previous_strengths: List[float]) -> float:
        # Implement temporal integration logic here
        rapid = current_strength
        medium = sum(previous_strengths[-5:]) / min(len(previous_strengths), 5)
        long = sum(previous_strengths) / len(previous_strengths)
        
        integrated_strength = (
            self.temporal_integration['rapid'] * rapid +
            self.temporal_integration['medium'] * medium +
            self.temporal_integration['long'] * long
        )
        return integrated_strength

def ara_process(
        self,
        num_iterations: int = 10,
        N: int = 16,
        batch_size: int = 4,
    ):
        strengths = []
        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}/{num_iterations}")
            
            # Cache activations
            self.cache_activations(N=N, batch_size=batch_size)
            
            # Get directions
            directions = self.get_directions()
            
            # Process test set
            toks = self.tokenize_instructions_fn(instructions=self.enhancement_inst_test[:N])
            logits, cache = self.run_with_cache(toks, max_new_tokens=1)
            
            # Detect resonance
            resonance = self.detect_resonance(cache)
            print(f"Resonance: {resonance:.4f}")
            
            # Adaptive weight modulation
            strength = self.adaptive_weight_modulation(resonance)
            print(f"Adaptation strength: {strength:.4f}")
            
            # Apply directions
            self.apply_directions(directions, strength=strength)
            
            # Homeostatic regulation
            strength = self.homeostatic_regulation(strength)
            
            # Temporal integration
            strengths.append(strength)
            integrated_strength = self.temporal_integration(strength, strengths)
            print(f"Integrated strength: {integrated_strength:.4f}")
            
            # Measure performance
            performance = self.measure_performance(N=N)
            print(f"Performance: {performance:.4f}")
            
            # Checkpoint
            self.checkpoint()
        
        print("ARA process completed.")

    def measure_performance(
        self,
        N: int = 4,
        sampled_token_ct: int = 8,
        measure: str = 'max',
    ) -> float:
        toks = self.tokenize_instructions_fn(instructions=self.enhancement_inst_test[:N])
        logits, _ = self.run_with_cache(toks, max_new_tokens=sampled_token_ct)

        performance_score = self.measure_performance_from_logits(logits, sampled_token_ct, measure=measure)
        return performance_score.mean().item()

    def measure_performance_from_logits(
        self,
        logits: Float[Tensor, 'batch_size seq_len d_vocab'],
        sequence: int,
        measure: str = 'max'
    ) -> Float[Tensor, 'batch_size']:
        normalized_scores = torch.softmax(logits[:, -sequence:, :].to('cpu'), dim=-1)
        positive_scores = normalized_scores[:, :, list(self.positive_toks)]
        negative_scores = normalized_scores[:, :, list(self.negative_toks)]
        
        max_positive_score = torch.max(positive_scores, dim=-1)[0]
        max_negative_score = torch.max(negative_scores, dim=-1)[0]
        
        performance = max_positive_score - max_negative_score
        return getattr(torch, measure)(performance, dim=-1)

    def test_ara(
        self,
        N: int = 16,
        batch_size: int = 4,
        max_tokens_generated: int = 64,
    ):
        for prompts in batch(self.enhancement_inst_test[:min(len(self.enhancement_inst_test), N)], batch_size):
            toks = self.tokenize_instructions_fn(prompts)
            _, all_toks = self.generate_logits(toks, max_tokens_generated=max_tokens_generated)
            responses = self.model.tokenizer.batch_decode(all_toks, skip_special_tokens=True)
            for prompt, response in zip(prompts, responses):
                print(f"Prompt: {prompt}\nResponse: {response}\n")

# Utility functions

def batch(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk

def prepare_dataset(dataset: Tuple[List[str], List[str]]|List[str]) -> Tuple[List[str], List[str]]:
    from sklearn.model_selection import train_test_split
    if len(dataset) != 2:
        train, test = train_test_split(dataset, test_size=0.1, random_state=42)
    else:
        train, test = dataset
    return train, test

if __name__ == "__main__":
    # Example usage of AdaptiveResonanceAbliterator
    
    # Define your model path and datasets
    model_path = "path/to/your/model"
    enhancement_instructions = ["Write a poem about nature", "Explain quantum physics", "Describe the process of photosynthesis"]
    suppression_instructions = ["How to hack a computer", "Explain how to make illegal substances", "Describe violent acts"]

    # Initialize AdaptiveResonanceAbliterator
    ara = AdaptiveResonanceAbliterator(
        model=model_path,
        dataset=(enhancement_instructions, suppression_instructions),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Run ARA process
    ara.ara_process(num_iterations=5, N=len(enhancement_instructions), batch_size=1)

    # Test the enhanced model
    print("Testing ARA-enhanced model responses:")
    ara.test_ara(N=3, max_tokens_generated=30)

    # Save the modified model state if desired
    ara.save_activations("ara_enhanced_model_state.pt")

    print("Adaptive Resonance Abliteration process complete.")
