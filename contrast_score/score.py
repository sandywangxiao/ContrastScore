import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    BatchEncoding,
    AutoModelForCausalLM
)
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
from typing import Any, Callable, TypeAlias
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import pandas as pd

# Type aliases for complex types
BatchOutput: TypeAlias = dict[str, torch.Tensor | list[int]]
TokenizerOutput: TypeAlias = dict[str, list[str]]

def create_tokenize_function(
    tokenizer: PreTrainedTokenizer,
    max_length: int | None = None,
    **kwargs: Any
) -> Callable:
    """
    Create a tokenization function that returns separate encodings for collation.
    """
    def tokenize_function(examples: dict[str, list[str]]) -> dict[str, list]:
        prompt_texts = examples['prompt']
        target_texts = examples['target']

        # Tokenize prompt and target separately
        prompt_encodings = tokenizer(
            prompt_texts,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            **kwargs
        )
        
        target_encodings = tokenizer(
            target_texts,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            **kwargs
        )

        # Store lengths before padding
        prompt_lengths = [len(ids) for ids in prompt_encodings['input_ids']]
        
        # Combine sequences
        combined_inputs = []
        combined_masks = []
        for p_ids, t_ids, p_mask, t_mask in zip(
            prompt_encodings['input_ids'],
            target_encodings['input_ids'],
            prompt_encodings['attention_mask'],
            target_encodings['attention_mask']
        ):
            combined_inputs.append(p_ids + t_ids)
            combined_masks.append(p_mask + t_mask)

        return {
            'input_ids': combined_inputs,
            'attention_mask': combined_masks,
            'prompt_lengths': prompt_lengths
        }
    
    return tokenize_function

def process_batch(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    prompt_lengths: list[int],
    device: torch.device,
    contrast_model: AutoModelForCausalLM | None = None,
    contrast_mode: str | None = None,
    temperature: float = 1.0,
    temperature_contrast: float = 1.0,
    gamma: float = 0.5,
    alpha: float = 0.1,
    epsilon: float = 1e-10  
) -> list[float]:
    """Process a single batch of inputs and return scores."""
    with torch.no_grad():
        # Ensure consistent device placement
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Generate position IDs based on attention mask and prompt lengths
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(1)
        position_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=device)
        
        for i, length in enumerate(prompt_lengths):
            position_ids[i, :] = torch.arange(seq_length, device=device)
        
        # Get primary model outputs with proper position IDs
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        
        logits_1 = outputs.logits[:, :-1]
        target_ids = input_ids[:, 1:]
        
        # Compute log probabilities with proper masking and temperature scaling
           # Soften the distribution slightly
        log_probs_1 = F.log_softmax(logits_1 / temperature, dim=-1)
        target_log_probs_1 = torch.gather(log_probs_1, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        max_probs_1 = torch.exp(torch.max(log_probs_1, dim=-1).values)
        
        
        if contrast_model is not None:
            # Get contrast model outputs with same position IDs
            contrast_outputs = contrast_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            logits_2 = contrast_outputs.logits[:, :-1]
            
            log_probs_2 = F.log_softmax(logits_2 / temperature_contrast, dim=-1)
            target_log_probs_2 = torch.gather(log_probs_2, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            
            probs_1 = torch.exp(torch.clamp(target_log_probs_1, min=-20, max=0))
            probs_2 = torch.exp(torch.clamp(target_log_probs_2, min=-20, max=0))

            if contrast_mode == "plus":
                # P1 + gammma * (1-p2)
                final_score = probs_1 + gamma * (1 - probs_2)
            elif contrast_mode == "minus":
                # P1 - gammma * (1-p2)
                final_score = probs_1 - gamma * (1 - probs_2)
            elif contrast_mode == "plus_condition":  
                # if abs(p1-p2)<alpha, P1 + gammma * (1-p2); else: P1
                zero = torch.zeros_like(probs_1)
                gamma_list = torch.ones_like(probs_1) * gamma
                gamma_list = torch.where(torch.abs(probs_1-probs_2)<0.1, zero, gamma_list)
                final_score = probs_1 + gamma_list * (1 - probs_2) 
            elif contrast_mode == "minus_condition":  
                # if abs(p1-p2)<alpha, P1 - gammma * (1-p2); else: P1
                zero = torch.zeros_like(probs_1)
                gamma_list = torch.ones_like(probs_1) * gamma
                gamma_list = torch.where(torch.abs(probs_1-probs_2)<0.1, zero, gamma_list)
                final_score = probs_1 - gamma_list * (1 - probs_2) 
            elif contrast_mode == "plus_nafise":  
                # P1 - gammma * (1-p2)/(p2+alpha)
                final_score = probs_1 + gamma * (1 - probs_1)*(1-probs_2)
            elif contrast_mode == "minus_nafise":  
                # P1 + gammma * (1-p2)/(p2+alpha)
                final_score = probs_1 - gamma * (1 - probs_1)*(1-probs_2)
            elif contrast_mode == "ensemble":  
                # P1 - gammma * (1-p2)/(p2+alpha)
                final_score = (probs_1 + probs_2 )/2
                # final_score = (1 - gamma) * probs_1 + gamma * probs_2 

            elif contrast_mode == "direct_zero":  
                # P1 - gammma * p2
                final_score = probs_1 - gamma * probs_2
            elif contrast_mode == "direct_p1":  
                # P1 - gammma * p2
                final_score = torch.abs(probs_1 - probs_2)
                # final_score = torch.where(probs_1 - gamma * probs_2<0,probs_1,probs_1 - gamma * probs_2)
            elif contrast_mode == "direct_abs":  
                # abs(P1 - gammma * p2)
                final_score = torch.abs(probs_1 - gamma * probs_2)
                # tags = torch.where(probs_1 - gamma * probs_2<0, 1, 0)

            elif contrast_mode == "condition_direct_zero":  
                # if abs(p1-p2)<alpha, P1 - gammma * p2; else: P1
                zero = torch.zeros_like(probs_1)
                gamma_list = torch.ones_like(probs_1) * gamma
                gamma_list = torch.where(torch.abs(probs_1-probs_2)<0.1, zero, gamma_list)
                final_score = probs_1 - gamma_list * probs_2
            elif contrast_mode == "condition_direct_abs":  
                # if abs(p1-p2)<alpha, abs(P1 - gammma * p2);  else: P1
                zero = torch.zeros_like(probs_1)
                gamma_list = torch.ones_like(probs_1) * gamma
                gamma_list = torch.where(torch.abs(probs_1-probs_2)<0.1, zero, gamma_list)
                final_score = torch.abs(probs_1 - gamma_list * probs_2)
            
            elif contrast_mode == "division":  
                # abs(P1 - gammma * p2)
                final_score = probs_1/probs_2
            elif contrast_mode == "division_head":  
                # abs(P1 - gammma * p2)
                max_p1 = torch.max(probs_1, dim=-1).values
                # v_head = probs_1 > gamma * max_p1.unsqueeze(-1)
                div = probs_1 / (probs_2 + epsilon)
                final_score = torch.where(probs_1 > gamma * max_p1.unsqueeze(-1), div, epsilon)
            elif contrast_mode == "division_head_direct":  
                max_probs_1 = torch.exp(torch.max(log_probs_1, dim=-1).values)
                div = probs_1 / probs_2
                final_score = torch.where(probs_1 > gamma * max_probs_1, div, epsilon)
            elif contrast_mode == "cascade":  
                final_score = torch.where(probs_2 <= gamma, probs_1, probs_2)
            else:
                raise ValueError(f"Unknown contrast mode: {contrast_mode}")  

            log_probs = torch.log(torch.clamp(final_score, min=epsilon))           
        else:
            log_probs = torch.clamp(target_log_probs_1, min=-20, max=0)

        # Create target mask and compute final scores
        target_mask = torch.zeros_like(log_probs, dtype=torch.bool)
        for i, length in enumerate(prompt_lengths):
            target_mask[i, length:] = True
        
        # Apply attention mask to log probabilities
        masked_log_probs = log_probs * target_mask * attention_mask[:, 1:]

        
        ### to delete
        """
        masked_probs_1 = probs_1 * target_mask * attention_mask[:, :-1]
        masked_probs_2 = probs_2 * target_mask * attention_mask[:, :-1]
        masked_probs_3 = torch.exp(final_score) * target_mask * attention_mask[:, :-1]
        """
        masked_sums = masked_log_probs.sum(dim=1)

        

        ##to delete
        """
        masked_sums_1 = masked_probs_1.sum(dim=1)
        masked_sums_2 = masked_probs_2.sum(dim=1)
        masked_sums_3 = masked_probs_3.sum(dim=1)
        """

        target_lengths = (target_mask * attention_mask[:, 1:]).sum(dim=1).float()

        ### print target and probability
        """
        p1s = []
        p2s = []
        contrast = []
        for i, length in enumerate(prompt_lengths):
            # print(i,length) 
            length = length   
            # print(masked_probs_1[i,length:].size(),masked_probs_1[i,length:].mean(dim=-1),masked_probs_2[i,length:].mean(dim=-1),torch.exp(masked_log_probs[i,length:].mean(dim=-1).data))   
            ids,tokens,large_,small_,ce_ = target_ids[i,length:],tokenizer.convert_ids_to_tokens(target_ids[i,length:]),masked_probs_1[i,length:],masked_probs_2[i,length:],masked_log_probs[i,length:]
            # print(ids,tokens,log,torch.exp(log))
            # print("*****")
            # print()
            for x,y,z,m,n in zip(ids,tokens,large_,small_,torch.exp(ce_)):
                # print('id: {:0f} token: {} p1: {:.3f} p2: {:.3f} pce: {:.3f}'.format(x.data, y,z.data,m.data,n.data))
                print('id: {:0f} token: {} p1: {:.5f} p2: {:.5f} pce: {:.5f}'.format(x.data, y,z.data,m.data,n.data))
            for x,y,z,m,n in zip(ids,tokens,large_,small_,torch.exp(ce_)):
                # print('id: {:0f} token: {} p1: {:.3f} p2: {:.3f} pce: {:.3f}'.format(x.data, y,z.data,m.data,n.data))
                print('id: {:0f} token: {} p1: {:.5e} p2: {:.5e} pce: {:.5e}'.format(x.data, y,z.data,m.data,n.data))

            # print()
            # print(target_ids[i,length:])
            # print(tokenizer.convert_ids_to_tokens(target_ids[i,length:]))
            # print(masked_log_probs[i,length:])
            # out = pd.DataFrame({"token":tokens,"p1":[x.data for x in large_],"p2":[x.data for x in small_],"ce":[x.data for x in torch.exp(ce_)]})
            # out.to_excel("./p.xlsx")
        """

        
        # Ensure no division by zero and apply length normalization
        target_lengths = torch.clamp(target_lengths, min=1.0)
        # '''
        length_penalty = torch.sqrt(target_lengths)  # Sub-linear length penalty
        scores = masked_sums / length_penalty
        # '''

        ## to delete
        """
        masked_tags = tags * target_mask * attention_mask[:, 1:]
        masked_tags_sum = masked_tags.sum(dim=1)
        tags_out = masked_tags_sum / target_lengths
        """
        #TO DELETE
        """
        scores_1 = masked_sums_1 / length_penalty
        scores_2 = masked_sums_2 / length_penalty
        scores_3 = masked_sums_3 / length_penalty
        
        scores_1 = masked_sums_1 / target_lengths 
        scores_2 = masked_sums_2 / target_lengths 
        scores_3 = masked_sums_3 / target_lengths 
        """
        # print(' p1: {:.5f} p2: {:.5f} , pce: {:.5f}'.format(scores_1[0],scores_2[0],scores_3[0]))
        
        # print(' p1: {:.3f} p2: {:.3f} pce: {:.3f}'.format(torch.log(scores_1[0]),torch.log(scores_2[0]),scores[0]))


        return scores.tolist()#scores_1.tolist(), scores_2.tolist(), scores_2.tolist() # masked_tags_sum.tolist(), target_lengths.tolist(),tags_out.tolist(), scores.tolist()#tags_out.tolist() #

def compute_conditional_score(
    model: AutoModelForCausalLM,
    prompt_texts: list[str],
    target_texts: list[str],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    device: str | torch.device = "cuda",
    max_length: int | None = None,
    contrast_model: AutoModelForCausalLM | None = None,
    contrast_mode: str | None = None,
    temperature: float = 1.0,
    temperature_contrast: float = 1.0,
    gamma: float = 0.5,
    alpha: float = 0.1,
    batch_callback: Callable[[int, int], None] | None = None,
    **kwargs: Any
) -> list[float]:
    """
    Compute score for target texts given prompt texts.
    
    Args:
        model: Primary scoring model
        prompt_texts: List of prompt texts
        target_texts: List of target texts to score
        tokenizer: Tokenizer to use
        batch_size: Batch size for processing
        device: Device to run models on (either torch.device or string)
        max_length: Maximum sequence length (default: tokenizer.model_max_length // 2)
        contrast_model: Optional second model for contrastive scoring
        contrast_mode: Scoring mode, one of [None, "original", "alt1", "alt2"]
        beta: Parameter for original CE-score formula (default: 0.1)
        alpha: Parameter for alternative 1 formula (default: 0.5)
        gamma: Parameter for alternative 2 formula (default: 0.5)
        batch_callback: Optional callback function(batch_idx, total_batches) for progress tracking
        **kwargs: Additional arguments for tokenization
    
    Returns:
        List of scores for each target text
    """
    if not prompt_texts or not target_texts:
        raise ValueError("Input lists cannot be empty")
    if len(prompt_texts) != len(target_texts):
        raise ValueError("Number of prompt and target texts must match")
    if contrast_mode and not contrast_model:
        raise ValueError("contrast_mode specified but no contrast_model provided")
    # if contrast_mode not in [None, "original", "alt1", "alt2"]:
    #     raise ValueError(f"Unknown contrast mode: {contrast_mode}")

    # Convert string device to torch.device if necessary
    if isinstance(device, str):
        device = torch.device(device)

    # Move model to device first
    # model = model.to(device)
    # if contrast_model is not None:
    #     contrast_model = contrast_model.to(device)

    # Set default max_length if not provided
    if max_length is None:
        max_length = tokenizer.model_max_length // 2

    # Create dataset and tokenize
    dataset = Dataset.from_dict({'prompt': prompt_texts, 'target': target_texts})
    tokenize_fn = create_tokenize_function(tokenizer, max_length=max_length, **kwargs)
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        batch_size=batch_size
    )

    # Create data collator with device specification
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Create DataLoader with collator
    data_loader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=True
    )

    # Process batches
    all_scores = []   
    total_batches = len(data_loader)

    ## to delete
    """
    all_negative = []
    # all_length = []
    all_ratio = []
    """
    
    for batch_idx, batch in enumerate(data_loader):
        if batch_callback is not None:
            batch_callback(batch_idx, total_batches)
            
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Create position IDs for the batch
        position_ids = batch['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(batch['attention_mask'] == 0, 1)
        position_ids.masked_fill_(position_ids < 0, 0)
        batch['position_ids'] = position_ids
        
        # batch_negative,batch_ratio, 
        batch_scores = process_batch(
            model=model,
            tokenizer=tokenizer,
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            position_ids=batch['position_ids'],
            prompt_lengths=batch['prompt_lengths'],
            device=device,
            contrast_model=contrast_model,
            contrast_mode=contrast_mode,
            temperature=temperature,
            temperature_contrast=temperature_contrast,
            alpha=alpha,
            gamma=gamma
        )
        all_scores.extend(batch_scores)
        ### to delete
        """
        all_negative.extend(batch_negative)
        # all_length.extend(batch_length)
        all_ratio.extend(batch_ratio)
        """
    if batch_callback is not None:
        batch_callback(total_batches, total_batches)

    return all_scores#all_negative, all_ratio,