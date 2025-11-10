from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
# import sacrebleu
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
import json
from contra_score.score import compute_conditional_score

def process_data():
    """
    read parquet-format data and get the useful information
    """
    data = pd.read_parquet("~/data/wmt23/data/train-00000-of-00001.parquet") 
    data["prompt"] = ["Translate the following sentence to English:\n" + data["src"][i] if data["lp"][i].split("-")[-1]=="en"\
                        else "Translate the following sentence to German:\n" + data["src"][i] for i in range(len(data))]

    data["output"] = data["mt"]
    data.to_csv("~/tasks/contrastcore/wmt23.csv", index=False)
    

def read_online_model(model_family, model_name):
    model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype="auto",
            device_map="auto"
    ) 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_family == 'llama':
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    return model,tokenizer

def calculate_contrast_mode(model_family,dataset,contrast_mode):
    """
    calculate different contrast mode score
    """
    data = pd.read_csv(f'~/tasks/contrastcore/{dataset}.csv')
    if dataset == "wmt23":
        max_len = 512
    else:
        max_len = 1024

    if model_family == "llama":
        model_large_names = [ "meta-llama/Llama-3.2-3B-Instruct","meta-llama/Llama-3.1-8B-Instruct","meta-llama/Llama-3.1-8B-Instruct"]
        model_small_names = [ "meta-llama/Llama-3.2-1B-Instruct","meta-llama/Llama-3.2-1B-Instruct","meta-llama/Llama-3.2-3B-Instruct"]
    else:
        model_large_names = [ "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"] 
        model_small_names = [ "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"] 
    for model_large_name, model_small_name in zip(model_large_names,model_small_names):
        large, small = model_large_name.split("-")[-2], model_small_name.split("-")[-2]
        model_large, tokenzier = read_online_model(model_family, model_large_name)
        model_small, _ = read_online_model(model_family, model_small_name)  
        print(model_large_name, model_small_name, contrast_mode)            
        data[f"{large}_{small}_{contrast_mode}"] = compute_conditional_score(
                                            model = model_large,
                                            prompt_texts = list(data['prompt']),
                                            target_texts =list(data["output"]), 
                                            tokenizer = tokenzier,
                                            batch_size = 16,
                                            device = "cuda",
                                            max_length = max_len, 
                                            contrast_model = model_small,
                                            contrast_mode = contrast_mode  
                                            )
    data.to_csv(f'~/tasks/contrastcore/{dataset}_{model_family}_{contrast_mode}.csv',index=False) 


def cal_single_model(model_family, dataset):   
    """
    single model score
    """
    data = pd.read_csv(f'~/tasks/contrastcore/{dataset}.csv')

    if model_family == 'llama':
        model_names = [ "meta-llama/Llama-3.2-1B-Instruct","meta-llama/Llama-3.2-3B-Instruct","meta-llama/Llama-3.1-8B-Instruct"]
    else:
        model_names = [ "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"] 
    
    if dataset == "wmt23":
        max_len = 512
    else:
        max_len = 1024

    for model_name in model_names:
        print("model:", model_name)
        model, tokenizer = read_online_model(model_family, model_name)                      
        single_scores = compute_conditional_score(
                    model = model,
                    prompt_texts = list(data['prompt']),
                    target_texts =list(data['output']),
                    tokenizer = tokenizer,
                    batch_size = 16,
                    device = "cuda",
                    max_length = max_len,
                    temperature = 1.0
                    )

        data[f"{model_name}"] = single_scores
    data.to_csv(f'~/tasks/contrastcore/{dataset}_single_{model_family}.csv',index=False)   

def single_model_corr(dataset):
    """
    single model correlation  
    """
    out = pd.DataFrame()
    models = []
    scores_en_de = []
    scores_zh_en = []
    scores_he_en = []
    scores_avg = []
    scores_coherence = []
    scores_consistency = []
    scores_relevance = []
    scores_fluency = []

    for model_family in ["llama", "qw"]:
        if model_family == 'llama':
            model_names = ["llama32_1B_instruct", "llama32_3B_instruct", "llama31_8B_instruct"]
        else:
            model_names = ["qw25_0.5B","qw25_3B", "qw25_7B", ]

        data = pd.read_csv(f'~/tasks/contrastcore/{dataset}_single_{model_family}.csv')
        for corr in ["pearson"]:
            for model in model_names:
                models.append(f"{model}_{corr}")
                if dataset == "wmt23":
                    score_en_de = data.groupby("lp")[f"{model}"].corr(data["mqm-score"],method=corr)["en-de"]
                    score_zh_en = data.groupby("lp")[f"{model}"].corr(data["mqm-score"],method=corr)["zh-en"]
                    score_he_en = data.groupby("lp")[f"{model}"].corr(data["mqm-score"],method=corr)["he-en"]
                    score_avg = (score_en_de + score_zh_en + score_he_en)/3.0
                    scores_en_de.append(score_en_de)
                    scores_zh_en.append(score_zh_en)
                    scores_he_en.append(score_he_en)
                    scores_avg.append(score_avg)
                else:
                    score_coherence = data[f"{model}"].corr(data[("coherence")],method=corr)
                    score_consistency = data[f"{model}"].corr(data[("consistency")],method=corr)
                    score_fluency = data[f"{model}"].corr(data[("fluency")],method=corr)
                    score_relevance = data[f"{model}"].corr(data[("relevance")],method=corr)
                    score_avg = (score_coherence + score_consistency + score_fluency + score_relevance)/4.0
                    scores_coherence.append(score_coherence)
                    scores_consistency.append(score_consistency)
                    scores_fluency.append(score_fluency)
                    scores_relevance.append(score_relevance)
                    scores_avg.append(score_avg)

    if dataset == "wmt23":
        out = pd.DataFrame({"model":models, "en_de":scores_en_de, "zh_en":scores_zh_en, "he_en":scores_he_en, "avg":scores_avg})
    else:
        out = pd.DataFrame({"model":models, "coherence":scores_coherence,"consistency":scores_consistency,"fluency":scores_fluency,"relevance":scores_relevance, "avg":scores_avg})

    out.to_csv(f"./{dataset}_single_model_corr_{model_family}.csv", index=False)


def contrast_mode_corr(dataset,contrast_mode):
    """
    contrast score, correlation
    """
    out = pd.DataFrame()
    models = []
    scores_en_de = []
    scores_zh_en = []
    scores_he_en = []
    scores_avg = [] 
    scores_coherence = []
    scores_consistency = []
    scores_fluency = []
    scores_relevance = []

    for model_family in ["llama" ,"qw"]: #,"qw"
        if model_family == "llama":
            model_large_names = ["llama31_8B_instruct", "llama31_8B_instruct","llama32_3B_instruct"]
            model_small_names = ["llama32_3B_instruct","llama32_1B_instruct","llama32_1B_instruct" ]
        else:
            model_large_names = ["qw25_7B", "qw25_7B", "qw25_3B",] #
            model_small_names = ["qw25_3B", "qw25_0.5B", "qw25_0.5B",] #

        data = pd.read_csv(f'~/tasks/contrastcore/{dataset}_{model_family}_{contrast_mode}.csv')

        for corr in ["pearson"]:
            for model_large_name, model_small_name in zip(model_large_names,model_small_names):
                large, small = model_large_name.split("_")[1], model_small_name.split("_")[1]                    
                models.append(f"{large}_{small}_{contrast_mode}_{corr}")
                if dataset == "wmt23":
                    score_en_de = data.groupby("lp")[f"{large}_{small}_{contrast_mode}"].corr(data["mqm-score"],method=corr)["en-de"]
                    score_zh_en = data.groupby("lp")[f"{large}_{small}_{contrast_mode}"].corr(data["mqm-score"],method=corr)["zh-en"]
                    score_he_en = data.groupby("lp")[f"{large}_{small}_{contrast_mode}"].corr(data["mqm-score"],method=corr)["he-en"]
                    score_avg = (score_en_de + score_zh_en + score_he_en)/3.0
                    scores_en_de.append(score_en_de)
                    scores_zh_en.append(score_zh_en)
                    scores_he_en.append(score_he_en)
                    scores_avg.append(score_avg)
                else:
                    score_coherence = data[f"{large}_{small}_{contrast_mode}"].corr(data[("coherence")],method=corr)
                    score_consistency = data[f"{large}_{small}_{contrast_mode}"].corr(data[("consistency")],method=corr)
                    score_fluency = data[f"{large}_{small}_{contrast_mode}"].corr(data[("fluency")],method=corr)
                    score_relevance = data[f"{large}_{small}_{contrast_mode}"].corr(data[("relevance")],method=corr)
                    score_avg = (score_coherence + score_consistency + score_fluency + score_relevance)/4.0
                    scores_coherence.append(score_coherence)
                    scores_consistency.append(score_consistency)
                    scores_fluency.append(score_fluency)
                    scores_relevance.append(score_relevance)
                    scores_avg.append(score_avg)

    if dataset == "wmt23":
        out = pd.DataFrame({"model":models, "en_de":scores_en_de, "zh_en":scores_zh_en, "he_en":scores_he_en, "avg":scores_avg})
    else:
        out = pd.DataFrame({"model":models, "coherence":scores_coherence,"consistency":scores_consistency,"fluency":scores_fluency,"relevance":scores_relevance, "avg":scores_avg})
    
    out.to_csv(f"{dataset}_contrast_mode_corr_{contrast_mode}.csv", index=False)

if __name__ == "__main__":
    # 1.read data
    process_data()
      
    # 2.calculate single model score
    for dataset in ["wmt23","summeval"]:        
        for model_family in ["llama", "qw"]:
            cal_single_model(model_family, dataset)
    
    # 3.calculate contrast mode score
    for dataset in ["wmt23","summeval"]:        
        for model_family in ["llama", "qw"]:
            calculate_contrast_mode(model_family, dataset, "ensemble")
            calculate_contrast_mode(model_family, dataset, "contrast")

    # 4. calualte correlation 
    for dataset in ["wmt23","summeval"]:        
        single_model_corr(dataset)
        contrast_mode_corr(dataset, "ensemble")
        contrast_mode_corr(dataset, "contrast")
