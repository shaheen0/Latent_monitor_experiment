import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from typing import Tuple
import numpy as np
from config import DEVICE ,model_id

def load_model(
    model_id: str,
    atten: str = "sdpa",
    torch_dtype: torch.dtype = torch.float16,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:

    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation=atten,
        torch_dtype=torch_dtype,
    ).to(DEVICE)

    model.eval()
    model.requires_grad_(False)

    return model, tokenizer

# ==============================================================================================

def get_layer_index(
    model: PreTrainedModel,
    ratio: float = 0.75,
) -> int:

    num_layers = model.config.num_hidden_layers

    return int(num_layers * ratio)

# ==============================================================================================

def prepare_text(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    response: str,
) -> tuple[str, str]:

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )

    if not isinstance(prompt_text, str):
        raise TypeError("Expected prompt_text to be a string.")

    full_text = prompt_text + response

    return prompt_text, full_text


# ----------------------------------------------------------------------------------------------
def mean_pooling(answer_activations: np.ndarray) -> np.ndarray:
    """Optional: [T, D] -> [D]. Never used in token-level extraction."""
    if answer_activations.ndim != 2 or answer_activations.shape[0] == 0:
        raise ValueError("Expected nonempty token activations [T, D].")
    return answer_activations.astype(np.float32).mean(axis=0)


@torch.no_grad()
# Token-level Activations (extract_activations function)
def extract_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layer_idx: int,
    sentences: list[str],
    question: str = "Say something.",
    batch_size: int = 2,
) -> list[dict]:
    
    device = next(model.parameters()).device
    results: list[dict] = []
    captured: dict[str, torch.Tensor] = {}

    def hook(_model, _inputs, output):
        captured["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    
    try:
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start:start + batch_size]
            
            prepared_text_pairs = [prepare_text(tokenizer, question, response) for response in batch] # [(prompt_text-1,full_text-1),(prompt_text-2,full_text-2)]
            
            # print(f"prepared_text_pairs {prepared_text_pairs }")
            prompt_texts = [prompt for prompt, _ in prepared_text_pairs ]  
   
            # print(f"prompt_texts {prompt_texts}")
            
            full_texts = [full for _, full in prepared_text_pairs ]
            
            # print(f"full_text  {full_texts}")
            tokenized_batch = tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=False,  # Never silently drop response tokens
                add_special_tokens=False,
            ).to(device)

            max_positions = getattr(model.config, "max_position_embeddings", None)
            if max_positions is not None and tokenized_batch.input_ids.shape[1] > max_positions:
                raise ValueError("Batch exceeds model context limit.")

            prompt_lengths = []
            for j, prompt_text in enumerate(prompt_texts):
                
                prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
                
                n_total = int(tokenized_batch.attention_mask[j].sum().item())
                
                full_ids_list = tokenized_batch.input_ids[j, :n_total].tolist()
                if full_ids_list[:len(prompt_ids)] != prompt_ids:
                    raise ValueError(f"Token alignment mismatch at sample {start + j}.")
                if n_total <= len(prompt_ids):
                    raise ValueError(f"Empty response at sample {start + j}.")
                prompt_lengths.append(len(prompt_ids))

            captured.clear()
            model(
                input_ids=tokenized_batch.input_ids,
                attention_mask=tokenized_batch.attention_mask,
                use_cache=False
                )
            if "h" not in captured:
                raise RuntimeError("Layer hook did not run.")
            hidden = captured["h"]  # [B, padded_T, D]

            for j, n_prompt in enumerate(prompt_lengths):
                
                n_total = int(tokenized_batch.attention_mask[j].sum().item())
                
                response_hidden = hidden[j, n_prompt:n_total, :]
                
                if not torch.isfinite(response_hidden).all().item():
                    raise ValueError(f"NaN/Inf at sample {start + j}.")
                response_ids = tokenized_batch.input_ids[j, n_prompt:n_total].cpu().tolist()
                activations = response_hidden.float().cpu().numpy()
                assert len(response_ids) == activations.shape[0]
                results.append({"token_ids": response_ids, "activations": activations})

            print(f"Processed: {min(start + batch_size, len(sentences))}/{len(sentences)}")
    finally:
        handle.remove()
    return results


# ==============================================================================================
# Sentence-level / Pooled Activations (get_activations_batch function)
def get_activations_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layer_idx: int,
    sentences: list[str],
    question: str = "Say something.",
    batch_size: int = 2,
) -> np.ndarray:
    """
    Extract mean-pooled activations for multiple sentences.

    Returns:
        NumPy array [n_sentences, hidden_dim].
    """

    # Step 1: Sab sentences ki token-level activations extract karo
    results = extract_activations(
        model=model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        sentences=sentences,
        question=question,
        batch_size=batch_size,
    )
    print(f"------------results {results}")
    # Step 2: Har sentence ke token vectors ko mean-pool karo
    activations = []

    for result in results:

        # Shape: [T, 576]
        answer_activations = result["activations"]

        # Shape: [576]
        vector = mean_pooling(answer_activations)

        activations.append(vector)

    # Step 3: Sab sentence vectors ko ek matrix mein combine karo
    return np.stack(activations).astype(np.float32)



# if __name__ == "__main__":
#     llm, tokenizer = load_model(model_id)
#     LAYER_INDEX = get_layer_index(llm)

#     answer = (
#         "Newton's first law states that an object remains "
#         "at rest or moves at constant velocity unless acted "
#         "upon by an external force."
#     )

#     results = extract_activations(
#         model=llm,
#         tokenizer=tokenizer,
#         layer_idx=LAYER_INDEX,
#         sentences=[answer],
#         batch_size=1,
#     )
#     print("Token IDs:", results[0]["token_ids"])
#     print("Activation shape:", results[0]["activations"].shape)
