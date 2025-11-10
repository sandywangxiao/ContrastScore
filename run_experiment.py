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
from sacrebleu.metrics import BLEU, CHRF, TER

def process_data():
    """
    read parquet-format data and get the useful information
    """
    # data = pd.read_parquet("~/data/wmt23/data/train-00000-of-00001.parquet") 
    # data["prompt"] = ["Translate the following sentence to English:\n" + data["src"][i] if data["lp"][i].split("-")[-1]=="en"\
    #                     else "Translate the following sentence to German:\n" + data["src"][i] for i in range(len(data))]

    # data["word_len"] = [len(x.split(" ")) for x in data["mt"]]
    # data["output"] = data["mt"]
    # print(data["lp"].value_counts())
    data = pd.read_excel("~/tasks/contrastive-evaluation/wmt23.xlsx")
    # data["output"] = data["mt"]
    data["prompt_ref"] = [x + "\n"+ "Rephrase the sentence above:" for x in data["ref"]]
    data.to_excel('~/tasks/contrastive-evaluation/wmt23_.xlsx',index=False)


def evaluate_llm(model_family,model_name):
    """
    use llm to evaluate the translation result
    """
    model_name_ = f"../../models/{model_family}/{model_name}"
    model = AutoModelForCausalLM.from_pretrained(
        model_name_,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    data = pd.read_excel("~/tasks/contrastive-evaluation/wmt23_200.xlsx")
    translate_sens = []
    # 
    for i in tqdm(range(len(data))):
        # You are Qwen, created by Alibaba Cloud. 
        prompt = "Please evaluate the quality of the translation text of the source text, based on the accuracy,fluency and coherence, and rate the translation on a scale of 1 to 10, where 1 = poor and 10 = excellent. Only return the score without extra text.  \n source text:" + data["src"][i] +"\ntranslation text:" + data["mt"][i] +"\n"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=256
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        translate_sens.append(response)
    data[f"llm_{model_name}"] = translate_sens
    data.to_excel(f'~/tasks/contrastive-evaluation/wmt200_prompt_{model_name}.xlsx',index=False)

def cal_BARTScore():
    bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')# 
    bart_scorer.load(path='../../tools/bartscore/BARTScore/model/bart_score.pth')
    data = pd.read_excel('~/tasks/contrastive-evaluation/wmt23_200.xlsx')
    bart_scores = []
    for i in tqdm(range(0,len(data),4)):
        bartscores = bart_scorer.score(list(data["mt"])[i:i+4],list(data["ref"])[i:i+4], batch_size=4)
        print(bartscores)
        bart_scores.extend(bartscores)
    data["llm_bartscore"] = bart_scores
    data.to_excel('~/tasks/contrastive-evaluation/wmt200_bartscore.xlsx',index=False)

def cal_bartscore_model(model_family, dataset):   
    """
    single model
    """
    data = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}.xlsx')
    data['prompt_ref_new'] = [x + "\n"+ "Such as, " for x in data["ref"]]#that is to say

    if model_family == 'llama':
        model_names = ["llama32_1B_instruct", "llama32_3B_instruct", "llama31_8B_instruct"]
    else:
        model_names = ["qw25_0.5B",  "qw25_3B", "qw25_7B"]
    
    if dataset == "wmt23":
        max_len = 512
    else:
        max_len = 1024

    for model_name in model_names:
        print("model:", model_name)
        model, tokenizer = read_model(model_family, model_name)                      
        single_scores = compute_conditional_score(
                    model = model,
                    prompt_texts = list(data['prompt_ref_new']),
                    target_texts =list(data['output']),
                    tokenizer = tokenizer,
                    batch_size = 16,
                    device = "cuda",
                    max_length = max_len,
                    temperature = 1.5
                    )

        data[f"{model_name}"] = single_scores

    # data.to_excel(f'~/tasks/contrastive-evaluation/{dataset}_all_bartscore_{model_family}_refs_3_1_4.xlsx',index=False)   
    data.to_excel(f'~/tasks/contrastive-evaluation/{dataset}_all_bartscore_{model_family}_tem_15_3_27.xlsx',index=False)   


def cal_ce_score_gamma(model_large, model_small,tokenizer, gamma_,dataset):   
    data = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}.xlsx')

    ce_scores_al2 = compute_conditional_score(model = model_large,
                                                prompt_texts = list(data['prompt']),
                                                target_texts =list(data['output']), # todo  mt
                                                tokenizer = tokenizer,
                                                batch_size = 16,
                                                device = "cuda",
                                                max_length = 512, 
                                                contrast_model = model_small,
                                                contrast_mode = "direct_abs", 
                                                temperature = 1.5,
                                                temperature_contrast= 0.5,
                                                gamma = gamma_   
                                                )
    return ce_scores_al2


def test_gamma(model_family, dataset):
    data = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}.xlsx')
    if model_family == "llama":
        model_large_names = [ "llama31_8B_instruct","llama32_3B_instruct"] #"llama31_8B_instruct",
        model_small_names = ["llama32_1B_instruct","llama32_1B_instruct" ]#"llama32_3B_instruct",
    else:
        model_large_names = ["qw25_7B", "qw25_7B", "qw25_3B",] #
        model_small_names = ["qw25_3B", "qw25_0.5B", "qw25_0.5B",] #

    for model_large_name, model_small_name in zip(model_large_names,model_small_names):
        large, small = model_large_name.split("_")[1], model_small_name.split("_")[1]
        model_large, tokenzier = read_model(model_family, model_large_name)
        model_small, _ = read_model(model_family, model_small_name)
        for gamma in tqdm([0.03,0.05,0.08,]):  # 0.7,0.8,0.9      
            print(model_large_name, model_small_name, gamma)            
            data[f"{large}_{small}_{gamma}"] = cal_ce_score_gamma(model_large, model_small,tokenzier, gamma,dataset)
    # data.to_excel(f'~/tasks/contrastive-evaluation/wmt23_ce_{model_family}_gamma_0.5_all_nifise.xlsx',index=False)
    data.to_excel(f'~/tasks/contrastive-evaluation/{dataset}_{model_family}_gamma_2_26.xlsx',index=False)


def test_formula(model_family,dataset,formulas):
    """
    calculate different formulas
    """
    data = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}.xlsx')
    # data['prompt_ref_new'] = [x + "\n"+ "Such as, " for x in data["ref"]]#that is to say; To put it another way:  To rephrase it:
    if dataset == "wmt23":
        # output_name = 'mt' 
        max_len = 512
    else:
        # output_name = 'output' 
        max_len = 1024

    if model_family == "llama":
        model_large_names = [ "llama32_3B_instruct","llama31_8B_instruct", "llama31_8B_instruct",]#
        model_small_names = ["llama32_1B_instruct" ,"llama32_3B_instruct","llama32_1B_instruct",]#
    else:
        model_large_names = [   "qw25_3B","qw25_7B","qw25_7B",]
        model_small_names = [   "qw25_0.5B","qw25_3B", "qw25_0.5B",]
    for model_large_name, model_small_name in zip(model_large_names,model_small_names):
        large, small = model_large_name.split("_")[1], model_small_name.split("_")[1]
        model_large, tokenzier = read_model(model_family, model_large_name)
        model_small, _ = read_model(model_family, model_small_name)
        for formula in tqdm(formulas):  #  
            print(model_large_name, model_small_name, formula)
            # data[f"{large}_{small}_{formula}_negative"],data[f"{large}_{small}_{formula}_length"],data[f"{large}_{small}_{formula}_ratio"], data[f"{large}_{small}_{formula}_score"]
            data[f"{large}"],data[f"{small}"],data[f"{large}_{small}_{formula}"] = compute_conditional_score(
                                                model = model_large,
                                                prompt_texts = list(data['prompt']),#_ref_new
                                                # target_texts =list(data['output']), 
                                                target_texts =["\n" + x for x in data['output']], 
                                                tokenizer = tokenzier,
                                                batch_size = 16,
                                                device = "cuda",
                                                max_length = max_len, 
                                                contrast_model = model_small,
                                                contrast_mode = formula, 
                                                temperature = 1.5,
                                                temperature_contrast= 0.5,
                                                gamma = 0.1 
            )  
            data.to_excel(f'~/tasks/contrastive-evaluation/{dataset}_{model_family}_rank_{large}_{small}.xlsx',index=False)  #formulas_refs_based_3_1_4

    # data.to_excel(f'~/tasks/contrastive-evaluation/{dataset}_{model_family}_formulas_v_head_3_28.xlsx',index=False)  #formulas_refs_based_3_1_4

def test_temperature(model_family, dataset):
    data = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}.xlsx')
    if dataset == "wmt23":
        output_name = 'mt' 
        max_len = 512
    else:
        output_name = 'output' 
        max_len = 1024
    if model_family == "llama":
        model_large_names = [ "llama31_8B_instruct","llama32_3B_instruct"]#"llama31_8B_instruct",
        model_small_names = ["llama32_1B_instruct","llama32_1B_instruct" ]#"llama32_3B_instruct",
    else:
        model_large_names = [ "qw25_7B", "qw25_3B",] #"qw25_7B",
        model_small_names = [ "qw25_0.5B", "qw25_0.5B"] #"qw25_3B",
    for model_large_name, model_small_name in zip(model_large_names,model_small_names):
        large, small = model_large_name.split("_")[1], model_small_name.split("_")[1]
        model_large, tokenzier = read_model(model_family, model_large_name)
        model_small, _ = read_model(model_family, model_small_name)
        for expert_T in [1.5,1.2,1.0,0.8,0.5]:
            for amateur_T in [0.5,0.8,1.0,1.2,1.5]:
                print(model_large_name, model_small_name, expert_T, amateur_T)            
                data[f"{large}_{small}_{expert_T}_{amateur_T}"] = compute_conditional_score(
                                                    model = model_large,
                                                    prompt_texts = list(data['prompt']),
                                                    target_texts =list(data[output_name]),
                                                    tokenizer = tokenzier,
                                                    batch_size = 16,
                                                    device = "cuda",
                                                    max_length = max_len, 
                                                    contrast_model = model_small,
                                                    contrast_mode = "plus_condition", 
                                                    temperature = expert_T,
                                                    temperature_contrast = amateur_T,
                                                    gamma = 0.1    
                                                    )
    data.to_excel(f'~/tasks/contrastive-evaluation/{dataset}_{model_family}_temperature_2_5.xlsx',index=False)


def read_model(model_family, model_name):
    model = AutoModelForCausalLM.from_pretrained(
            f"../../models/{model_family}/{model_name}", #../
            torch_dtype="auto",
            device_map="auto"
    ) 
    tokenizer = AutoTokenizer.from_pretrained(f"../../models/{model_family}/{model_name}")
    if model_family == 'llama':
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    return model,tokenizer

def cal_pearson_gamma(model_family,dataset):
    data = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}_{model_family}_gamma_2_26.xlsx')#{dataset}_{model_family}_gamma_baseline_2_8.xlsx
    if model_family == "llama":
        model_large_names = [ "llama31_8B_instruct","llama32_3B_instruct"]#"llama31_8B_instruct",
        model_small_names = ["llama32_1B_instruct","llama32_1B_instruct" ] #"llama32_3B_instruct",
    else:
        model_large_names = ["qw25_7B", "qw25_7B", "qw25_3B",] #
        model_small_names = ["qw25_3B", "qw25_0.5B", "qw25_0.5B"] #   

    names = []
    scores_en_de = []
    scores_zh_en = []
    scores_he_en = []
    scores_avg = []
    scores_avg_all = []
    scores_coherence = []
    scores_consistency = []
    scores_relevance = []
    scores_fluency = []

    for model_large_name, model_small_name in zip(model_large_names,model_small_names):
        for gamma in [0.03,0.05,0.08]: #[0.1,0.2,0.3,0.4,0.5,0.6, 0.7,0.8,0.9]:
            large, small = model_large_name.split("_")[1], model_small_name.split("_")[1]
            names.append(f"{large}_{small}_{gamma}")
            if dataset == "wmt23":
                score_en_de = data.groupby("lp")[f"{large}_{small}_{gamma}"].corr(data["mqm-score"],method='pearson')["en-de"]
                score_zh_en = data.groupby("lp")[f"{large}_{small}_{gamma}"].corr(data["mqm-score"],method='pearson')["zh-en"]
                score_he_en = data.groupby("lp")[f"{large}_{small}_{gamma}"].corr(data["mqm-score"],method='pearson')["he-en"]
                score_avg = (score_en_de + score_zh_en + score_he_en)/3.0
                scores_en_de.append(score_en_de)
                scores_zh_en.append(score_zh_en)
                scores_he_en.append(score_he_en)
                scores_avg.append(score_avg)
            else:
                score_coherence = data[f"{large}_{small}_{gamma}"].corr(data[("coherence")],method='pearson')
                score_consistency = data[f"{large}_{small}_{gamma}"].corr(data[("consistency")],method='pearson')
                score_fluency = data[f"{large}_{small}_{gamma}"].corr(data[("fluency")],method='pearson')
                score_relevance = data[f"{large}_{small}_{gamma}"].corr(data[("relevance")],method='pearson')
                score_avg = (score_coherence + score_consistency + score_fluency + score_relevance)/4.0
                scores_coherence.append(score_coherence)
                scores_consistency.append(score_consistency)
                scores_fluency.append(score_fluency)
                scores_relevance.append(score_relevance)
                scores_avg.append(score_avg)

    if dataset == "wmt23":
        out = pd.DataFrame({"gamma":names, "en_de":scores_en_de, "zh_en":scores_zh_en, "he_en":scores_he_en, "avg":scores_avg})
    else:
        out = pd.DataFrame({"gamma":names, "coherence":scores_coherence,"consistency":scores_consistency,"fluency":scores_fluency,"relevance":scores_relevance, "avg":scores_avg})
    
    out.to_excel(f'~/tasks/contrastcore/{dataset}_{model_family}_corr_2_26.xlsx')





def cal_corr_temperature_by_lang(model_family):
    data = pd.read_excel(f'~/tasks/contrastive-evaluation/wmt23_ce_{model_family}_gamma_temperature_2.xlsx')
    if model_family == "llama":
        model_large_names = [ "llama31_8B_instruct","llama32_3B_instruct"]#"llama31_8B_instruct",
        model_small_names = ["llama32_1B_instruct","llama32_1B_instruct" ]#"llama32_3B_instruct",
    else:
        model_large_names = [ "qw25_7B", "qw25_3B",] #"qw25_7B",
        model_small_names = [ "qw25_0.5B", "qw25_0.5B"] #"qw25_3B",

    names = []
    scores_pearson = []
    scores_kendall = []
    scores_spearman = []
    scores_en_de = []
    scores_zh_de = []
    scores_he_en = []

    for model_large_name, model_small_name in zip(model_large_names,model_small_names):
        large, small = model_large_name.split("_")[1], model_small_name.split("_")[1]
        for expert_T in [1.5,1.2,1.0,0.8,0.5]:
            for amateur_T in [0.5,0.8,1.0,1.2,1.5]:
                print(model_large_name, model_small_name, expert_T, amateur_T)                
                names.append(f"{large}_{small}_{expert_T}_{amateur_T}")
                scores_pearson.append(data[f"{large}_{small}_{expert_T}_{amateur_T}"].corr(data["mqm-score"],method="pearson"))
                scores_kendall.append(data[f"{large}_{small}_{expert_T}_{amateur_T}"].corr(data["mqm-score"],method="kendall"))
                scores_spearman.append(data[f"{large}_{small}_{expert_T}_{amateur_T}"].corr(data["mqm-score"],method="spearman"))
                scores_en_de.append(data.groupby("lp")[f"{large}_{small}_{expert_T}_{amateur_T}"].corr(data["mqm-score"],method="pearson")["en-de"])
                scores_zh_de.append(data.groupby("lp")[f"{large}_{small}_{expert_T}_{amateur_T}"].corr(data["mqm-score"],method="pearson")["zh-en"])
                scores_he_en.append(data.groupby("lp")[f"{large}_{small}_{expert_T}_{amateur_T}"].corr(data["mqm-score"],method="pearson")["he-en"])

    out = pd.DataFrame({"gamma":names, "en-de":scores_en_de,"zh-de":scores_zh_de, "he-en":scores_he_en,"all_pearson":scores_pearson,"all_kendall":scores_kendall,"all_spearman":scores_spearman})
    out.to_excel(f'~/tasks/contrastcore/wmt23_all_{model_family}_gamma_temperature_lang_2_new.xlsx')



def error_analysis(model_family, model_large_name, model_small_name): 
    """
    error analysis
    """
    data = pd.read_excel('~/tasks/contrastive-evaluation/wmt23.xlsx')
    model_large = AutoModelForCausalLM.from_pretrained(
        f"../../models/{model_family}/{model_large_name}" ,
        torch_dtype="auto",
        device_map="auto"
    )
    model_small = AutoModelForCausalLM.from_pretrained(
        f"../../models/{model_family}/{model_small_name}" ,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer_common = AutoTokenizer.from_pretrained(f"../../models/{model_family}/{model_large_name}")
    if model_family == 'llama':
        tokenizer_common.add_special_tokens({'pad_token': '[PAD]'})
        model_large.resize_token_embeddings(len(tokenizer_common))
        model_small.resize_token_embeddings(len(tokenizer_common))


    #[7467,16883,15706,13352,14529,9821,18060,19237,6290]
    indexes = [14756,12402,15933] #[19393,9977,13508] #[14763,10055,6524]#[20554,15846,12315]#[10901,13255,15609,19140] 
    # print(data['prompt'][6164])
    # print(data['prompt'][646])
    for index in indexes:
        print(index)
        ce_scores_al2 = compute_conditional_score(
        model = model_large,
        prompt_texts = list([data['prompt'][index]]),
        target_texts =list(["\n"+data['mt'][index]]),#"\n"+
        tokenizer = tokenizer_common,
        batch_size = 8,
        device = "cuda",
        max_length = 512, 
        contrast_model = model_small,
        contrast_mode = "direct_abs",
        temperature = 1.5,
        temperature_contrast = 0.5,
        gamma=0.1
    )
        print("*************")


    





def calucluate_single_model_corr(dataset):
    """
    To keep ******
    bartscore-style score,by lang  
    """
    out = pd.DataFrame()
    models = []
    scores_en_de = []
    scores_zh_en = []
    scores_he_en = []
    scores_avg = []
    scores_avg_all = [] 
    scores_coherence = []
    scores_consistency = []
    scores_relevance = []
    scores_fluency = []

    for model_family in ["llama", "qw"]:
        if model_family == 'llama':
            model_names = ["llama32_1B_instruct", "llama32_3B_instruct", "llama31_8B_instruct"]
        else:
            model_names = ["qw25_0.5B","qw25_3B", "qw25_7B", ]

        data = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}_all_bartscore_{model_family}_refs_3_1_4.xlsx')#dataset}_all_bartscore_{model_family}_2_13
        #all_bartscore_{model_family}_noT.xlsx')  wmt
        for corr in ["pearson","kendall","spearman"]:
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
                    scores_avg_all.append(data[f"{model}"].corr(data["mqm-score"],method=corr))
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
        out = pd.DataFrame({"model":models, "en_de":scores_en_de, "zh_en":scores_zh_en, "he_en":scores_he_en, "avg":scores_avg,"avg_all":scores_avg_all})
    else:
        out = pd.DataFrame({"model":models, "coherence":scores_coherence,"consistency":scores_consistency,"fluency":scores_fluency,"relevance":scores_relevance, "avg":scores_avg})

    out.to_excel(f"./{dataset}_bartscore_corr_ref_3_1_4.xlsx")#2_13


def formulas_CE_corr(dataset,formulas):
    """
    To keep  *******
    ce score, baseline  by lang
    """
    out = pd.DataFrame()
    models = []
    scores_en_de = []
    scores_zh_en = []
    scores_he_en = []
    scores_avg = []
    scores_avg_all = [] 
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

        # data = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}_{model_family}_formulas_direct_p1_3_24.xlsx')   ##### todo {dataset}_{model_family}_formulas_division_2_26
        data = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}_{model_family}_formulas_v_head_3_28.xlsx')   ##### todo {dataset}_{model_family}_formulas_division_2_26

        for corr in ["pearson","kendall","spearman"]:
            for formula in formulas:
                for model_large_name, model_small_name in zip(model_large_names,model_small_names):
                    large, small = model_large_name.split("_")[1], model_small_name.split("_")[1]                    
                    models.append(f"{large}_{small}_{formula}_{corr}")
                    if dataset == "wmt23":
                        score_en_de = data.groupby("lp")[f"{large}_{small}_{formula}"].corr(data["mqm-score"],method=corr)["en-de"]
                        score_zh_en = data.groupby("lp")[f"{large}_{small}_{formula}"].corr(data["mqm-score"],method=corr)["zh-en"]
                        score_he_en = data.groupby("lp")[f"{large}_{small}_{formula}"].corr(data["mqm-score"],method=corr)["he-en"]
                        score_avg = (score_en_de + score_zh_en + score_he_en)/3.0
                        scores_en_de.append(score_en_de)
                        scores_zh_en.append(score_zh_en)
                        scores_he_en.append(score_he_en)
                        scores_avg.append(score_avg)
                        scores_avg_all.append(data[f"{large}_{small}_{formula}"].corr(data["mqm-score"],method=corr))
                    else:
                        score_coherence = data[f"{large}_{small}_{formula}"].corr(data[("coherence")],method=corr)
                        score_consistency = data[f"{large}_{small}_{formula}"].corr(data[("consistency")],method=corr)
                        score_fluency = data[f"{large}_{small}_{formula}"].corr(data[("fluency")],method=corr)
                        score_relevance = data[f"{large}_{small}_{formula}"].corr(data[("relevance")],method=corr)
                        score_avg = (score_coherence + score_consistency + score_fluency + score_relevance)/4.0
                        scores_coherence.append(score_coherence)
                        scores_consistency.append(score_consistency)
                        scores_fluency.append(score_fluency)
                        scores_relevance.append(score_relevance)
                        scores_avg.append(score_avg)

    if dataset == "wmt23":
        out = pd.DataFrame({"model":models, "en_de":scores_en_de, "zh_en":scores_zh_en, "he_en":scores_he_en, "avg":scores_avg,"avg_all":scores_avg_all})
    else:
        out = pd.DataFrame({"model":models, "coherence":scores_coherence,"consistency":scores_consistency,"fluency":scores_fluency,"relevance":scores_relevance, "avg":scores_avg})

    # out.to_excel(f"./{dataset}_gamma_baseline_with_T.xlsx")  division_corr_2_26
    out.to_excel(f"{dataset}_formulas_direct_head_3_28.xlsx", index=False)
    


# def cal_pearson_gamma_by_lang(model_family):
#     data = pd.read_excel(f'~/tasks/contrastive-evaluation/wmt23_ce_{model_family}_gamma_formula_baseline.xlsx')
#     if model_family == "llama":
#         model_large_names = ["llama31_8B_instruct", "llama31_8B_instruct",]#"llama32_3B_instruct"]
#         model_small_names = ["llama32_3B_instruct","llama32_1B_instruct",]#"llama32_1B_instruct" ]
#     else:
#         model_large_names = ["qw25_7B", "qw25_7B", "qw25_3B","qw25_3B",] #
#         model_small_names = ["qw25_3B", "qw25_1.5B", "qw25_1.5B","qw25_0.5B"] #
#         # model_large_names = ["qw25_7B", "qw25_7B", "qw25_3B",] #
#         # model_small_names = ["qw25_3B", "qw25_1.5B", "qw25_1.5B"] #   

#     names = []
#     scores_pearson = []
#     scores_kendall = []
#     scores_spearman = []
#     scores_en_de = []
#     scores_zh_de = []
#     scores_he_en = []

#     for model_large_name, model_small_name in zip(model_large_names,model_small_names):
#         for gamma in ["plus_condition", "minus_condition", "plus_baseline", "minus_baseline",]:#[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]: #
#             large, small = model_large_name.split("_")[1], model_small_name.split("_")[1]
#             names.append(f"{large}_{small}_{gamma}")
#             scores_pearson.append(data[f"{large}_{small}_{gamma}"].corr(data["mqm-score"],method="pearson"))
#             scores_kendall.append(data[f"{large}_{small}_{gamma}"].corr(data["mqm-score"],method="kendall"))
#             scores_spearman.append(data[f"{large}_{small}_{gamma}"].corr(data["mqm-score"],method="spearman"))
#             scores_en_de.append(data.groupby("lp")[f"{large}_{small}_{gamma}"].corr(data["mqm-score"],method="pearson")["en-de"])
#             scores_zh_de.append(data.groupby("lp")[f"{large}_{small}_{gamma}"].corr(data["mqm-score"],method="pearson")["zh-en"])
#             scores_he_en.append(data.groupby("lp")[f"{large}_{small}_{gamma}"].corr(data["mqm-score"],method="pearson")["he-en"])

#     out = pd.DataFrame({"gamma":names, "en-de":scores_en_de,"zh-de":scores_zh_de, "he-en":scores_he_en,"all_pearson":scores_pearson,"all_kendall":scores_kendall,"all_spearman":scores_spearman})
#     out.to_excel(f'~/tasks/contrastive-evaluation/wmt23_all_{model_family}_gamma_temperature_lang.xlsx')


def process_summeval_data():
    # with open('../../data/summeval/model_annotations.aligned.paired.jsonl', 'r') as f:
    #     lines = f.readlines()
    # output = []
    # story = []
    # score_coherence = []
    # score_consistency = []
    # score_fluency = []
    # score_relevance = []
    # reference = []
    # references = []
    # for line in lines:
    #     text = json.loads(line.strip())
    #     output.append(text.get("decoded"))
    #     story.append(text.get("text"))
    #     reference.append(text.get("references")[0])
    #     references.append(text.get("references"))
    #     score_coherence.append(sum([x.get("coherence") for x in text.get("expert_annotations")])/3.0)
    #     score_consistency.append(sum([x.get("consistency") for x in text.get("expert_annotations")])/3.0)
    #     score_fluency.append(sum([x.get("fluency") for x in text.get("expert_annotations")])/3.0)
    #     score_relevance.append(sum([x.get("relevance") for x in text.get("expert_annotations")])/3.0)
    
    # out  = pd.DataFrame({"output":output, "story":story, "reference":reference,"references":references, "coherence":score_coherence,"consistency":score_consistency,"fluency":score_fluency,"relevance":score_relevance})
    # out["prompt"] = ["Write an accurate, relevant,  and coherent summary of the following texts:\n" + text + "\nSummary:\n" for text in out["story"]]
    # out["expert_avg"] = (out["coherence"] + out["consistency"] + out["fluency"] + out["relevance"])/4.0
    data = pd.read_excel("~/tasks/contrastive-evaluation/summeval.xlsx")
    # data["output"] = data["mt"]
    data["prompt_ref"] = [x + "\n"+ "Rephrase the sentence above:" for x in data["reference"]]
    # data.to_excel('~/tasks/contrastive-evaluation/wmt23_.xlsx',index=False)
    data.to_excel("./summeval_.xlsx",index=False)

def rescale_score(score_list):
    max_score = max(score_list)
    min_score = min(score_list)
    score_list = np.array(score_list)
    return ((score_list - min_score) / (max_score - min_score)).tolist()


def combine_data(dataset, model_family):
    if model_family == "llama":
        single_c = ['mt', 'system', 'mqm-score', 'src', 'ref', 'doc_id', 'lp', 'word_len',\
        'prompt', 'llama32_1B_instruct', 'llama32_3B_instruct', 'llama31_8B_instruct']
        ce_c = ['8B_3B_plus_baseline', '8B_3B_direct_abs', '8B_1B_plus_baseline', '8B_1B_direct_abs', '3B_1B_plus_baseline', '3B_1B_direct_abs']
    else:
        single_c = ['mt', 'system', 'mqm-score', 'src', 'ref', 'doc_id', 'lp', 'word_len',\
       'prompt', 'qw25_0.5B',  'qw25_3B', 'qw25_7B']
        ce_c = ['7B_3B_plus_baseline', '7B_3B_direct_abs',  '7B_0.5B_plus_baseline', '7B_0.5B_direct_abs', '3B_0.5B_plus_baseline', '3B_0.5B_direct_abs']

    
    data_single = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}_all_bartscore_{model_family}.xlsx')[single_c]
    data_ce = pd.read_excel(f'{dataset}_{model_family}_formulas_final_1.5_0.5_0.1.xlsx')[ce_c]
    # print(data_single.columns)
    # print(data_ce.columns)
    data = pd.concat([data_single,data_ce],axis =1)


    if model_family == "llama":
        data_c = ['llama32_1B_instruct', 'llama32_3B_instruct', 'llama31_8B_instruct','8B_3B_plus_baseline', '8B_3B_direct_abs', '8B_1B_plus_baseline', '8B_1B_direct_abs', '3B_1B_plus_baseline', '3B_1B_direct_abs','mqm-score']
    else:
        data_c = ['qw25_0.5B',  'qw25_3B', 'qw25_7B','7B_3B_plus_baseline', '7B_3B_direct_abs',  '7B_0.5B_plus_baseline', '7B_0.5B_direct_abs', '3B_0.5B_plus_baseline', '3B_0.5B_direct_abs','mqm-score']
    
    for column in data_c:

        print(column)
        # print(type(data[column]))
        # data[f"{column}_new"] = data[column].apply(rescale_score)
        data[f"{column}_new"] = rescale_score(data[column])

    data.to_excel(f"./{dataset}_{model_family}_merge.xlsx", index=False)

def test_formula_multi(model_family,dataset,formulas):
    """
    calculate different formulas, multi
    """
    data = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}.xlsx')

    with open('../../data/summeval/model_annotations.aligned.paired.jsonl', 'r') as f:
        lines = f.readlines()
    for i in range(11):
        data[f'prompt_{i}'] = [json.loads(line.strip()).get("references")[i] for line in lines]
        data[f'prompt_{i}_ref'] = [x + "\n"+ "Rephrase the sentence above:\n" for x in data[f'prompt_{i}']]

    if dataset == "wmt23":
        output_name = 'mt' 
        max_len = 512
    else:
        output_name = 'output' 
        max_len = 1024

    if model_family == "llama":
        model_large_names = ["llama31_8B_instruct", "llama31_8B_instruct","llama32_3B_instruct"]#
        model_small_names = ["llama32_3B_instruct","llama32_1B_instruct","llama32_1B_instruct" ]#
    else:
        model_large_names = [ "qw25_7B", "qw25_7B", "qw25_3B",] #
        model_small_names = [ "qw25_3B", "qw25_0.5B", "qw25_0.5B"] #
    
    for model_large_name, model_small_name in zip(model_large_names,model_small_names):
        large, small = model_large_name.split("_")[1], model_small_name.split("_")[1]
        model_large, tokenzier = read_model(model_family, model_large_name)
        model_small, _ = read_model(model_family, model_small_name)
        for formula in tqdm(formulas):  #  
            print(model_large_name, model_small_name, formula)  
            for i in range(11):          
                data[f"{large}_{small}_{formula}_{i}"] = compute_conditional_score(
                                                    model = model_large,
                                                    prompt_texts = list(data[f'prompt_{i}_ref']),
                                                    target_texts =list(data[output_name]), 
                                                    tokenizer = tokenzier,
                                                    batch_size = 16,
                                                    device = "cuda",
                                                    max_length = max_len, 
                                                    contrast_model = model_small,
                                                    contrast_mode = formula, 
                                                    temperature = 1.5,
                                                    temperature_contrast= 0.5,
                                                    gamma = 0.1   
                                                    )
    data.to_excel(f'~/tasks/contrastive-evaluation/{dataset}_{model_family}_formulas_refs_based_3_1.xlsx',index=False)

def formulas_CE_corr_multi(dataset,formulas):
    """
    To keep  *******
    ce score, baseline  by lang
    """
    out = pd.DataFrame()
    models = []
    scores_en_de = []
    scores_zh_en = []
    scores_he_en = []
    scores_avg = []
    scores_avg_all = [] 
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

        data = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}_{model_family}_formulas_refs_based_3_1.xlsx')   ##### todo {dataset}_{model_family}_formulas_division_2_26
        
        for corr in ["pearson","kendall","spearman"]:
            for formula in formulas:
                for model_large_name, model_small_name in zip(model_large_names,model_small_names):
                    large, small = model_large_name.split("_")[1], model_small_name.split("_")[1]                    
                    models.append(f"{large}_{small}_{formula}_{corr}")
                    if dataset == "wmt23":
                        score_en_de = data.groupby("lp")[f"{large}_{small}_{formula}"].corr(data["mqm-score"],method=corr)["en-de"]
                        score_zh_en = data.groupby("lp")[f"{large}_{small}_{formula}"].corr(data["mqm-score"],method=corr)["zh-en"]
                        score_he_en = data.groupby("lp")[f"{large}_{small}_{formula}"].corr(data["mqm-score"],method=corr)["he-en"]
                        score_avg = (score_en_de + score_zh_en + score_he_en)/3.0
                        scores_en_de.append(score_en_de)
                        scores_zh_en.append(score_zh_en)
                        scores_he_en.append(score_he_en)
                        scores_avg.append(score_avg)
                        scores_avg_all.append(data[f"{large}_{small}_{formula}"].corr(data["mqm-score"],method=corr))
                    else:                        
                        prompt_list = []
                        for i in range(11):          
                            prompt_list.append(f"{large}_{small}_{formula}_{i}")
                        
                        temp = data[prompt_list]
                        data[f"{large}_{small}_{formula}"] = temp.mean(axis=1) ###

                        score_coherence = data[f"{large}_{small}_{formula}"].corr(data[("coherence")],method=corr)
                        score_consistency = data[f"{large}_{small}_{formula}"].corr(data[("consistency")],method=corr)
                        score_fluency = data[f"{large}_{small}_{formula}"].corr(data[("fluency")],method=corr)
                        score_relevance = data[f"{large}_{small}_{formula}"].corr(data[("relevance")],method=corr)
                        score_avg = (score_coherence + score_consistency + score_fluency + score_relevance)/4.0
                        scores_coherence.append(score_coherence)
                        scores_consistency.append(score_consistency)
                        scores_fluency.append(score_fluency)
                        scores_relevance.append(score_relevance)
                        scores_avg.append(score_avg)

    if dataset == "wmt23":
        out = pd.DataFrame({"model":models, "en_de":scores_en_de, "zh_en":scores_zh_en, "he_en":scores_he_en, "avg":scores_avg,"avg_all":scores_avg_all})
    else:
        out = pd.DataFrame({"model":models, "coherence":scores_coherence,"consistency":scores_consistency,"fluency":scores_fluency,"relevance":scores_relevance, "avg":scores_avg})

    out.to_excel(f"{dataset}_formulas_refs_based_3_1_avg.xlsx", index=False)



def formulas_CE_corr_multi_wmt(dataset,formulas):
    """
    To keep  *******
    test multi prompts
    """
    out = pd.DataFrame()
    models = []
    scores_en_de = []
    scores_zh_en = []
    scores_he_en = []
    scores_avg = []
    scores_avg_all = [] 

    # data1 = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}_{model_family}_formulas_refs_based.xlsx')
    # data2 = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}_{model_family}_formulas_refs_based_lang_2_28.xlsx')   ##### todo {dataset}_{model_family}_formulas_division_2_26
    


    for model_family in ["llama" ,"qw"]: #,"qw"
        if model_family == "llama":
            model_large_names = ["llama32_3B_instruct","llama31_8B_instruct", "llama31_8B_instruct",]
            model_small_names = ["llama32_1B_instruct","llama32_1B_instruct","llama32_3B_instruct", ]
        else:
            model_large_names = ["qw25_3B","qw25_7B", "qw25_7B", ] #
            model_small_names = ["qw25_0.5B", "qw25_0.5B","qw25_3B",] #

        data1 = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}_{model_family}_formulas_refs_based_3_1.xlsx')   ##### that is to say
        data2 = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}_{model_family}_formulas_refs_based_3_1_2.xlsx') ## to put another way
        data3 = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}_{model_family}_formulas_refs_based_3_1_3.xlsx')  #to rephrase it
        data4 = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}_{model_family}_formulas_refs_based_3_1_4.xlsx') #such as
        data5 = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}_{model_family}_formulas_ref_based.xlsx') #rephrase the sentences

        data_all = [data1, data2, data3, data4,data5]   

        for corr in ["pearson"]:#,"kendall","spearman"
            for index in [[0],[1],[2],[3],[4],[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[2,3],[2,4],[0,1,2],[1,2,3],[0,1,3],[0,2,3],[0,2,4],[0,1,2,3],[0,1,2,3,4]]:
                for formula in formulas:
                # for index in [[0],[1],[2],[3],[4][0,1],[0,2],[0,3],[1,2],[1,3],[2,3],[0,1,2],[1,2,3],[0,1,3],[0,2,3],[0,1,2,3]]:
                    for model_large_name, model_small_name in zip(model_large_names,model_small_names):
                        large, small = model_large_name.split("_")[1], model_small_name.split("_")[1]  
                    
                        index_tag= "_".join([str(x) for x in index])                 
                        models.append(f"{large}_{small}_{formula}_{index_tag}")
                        # out = data1["mqm-score"]
                        data = pd.DataFrame()
                        for i in index:
                            data = pd.concat([data,data_all[i][f"{large}_{small}_{formula}"]], axis=1)
                        temp = data.mean(axis=1).rename(f"{large}_{small}_{formula}")
                        data = pd.concat([temp, data1[["mqm-score","lp"]]], axis=1)
                        print(data.columns)
                        if dataset == "wmt23":
                            score_en_de = data.groupby("lp")[f"{large}_{small}_{formula}"].corr(data["mqm-score"],method=corr)["en-de"]
                            score_zh_en = data.groupby("lp")[f"{large}_{small}_{formula}"].corr(data["mqm-score"],method=corr)["zh-en"]
                            score_he_en = data.groupby("lp")[f"{large}_{small}_{formula}"].corr(data["mqm-score"],method=corr)["he-en"]
                            score_avg = (score_en_de + score_zh_en + score_he_en)/3.0
                            scores_en_de.append(score_en_de)
                            scores_zh_en.append(score_zh_en)
                            scores_he_en.append(score_he_en)
                            scores_avg.append(score_avg)
                            scores_avg_all.append(data[f"{large}_{small}_{formula}"].corr(data["mqm-score"],method=corr))

    if dataset == "wmt23":
        out = pd.DataFrame({"model":models, "en_de":scores_en_de, "zh_en":scores_zh_en, "he_en":scores_he_en, "avg":scores_avg,"avg_all":scores_avg_all})
    out.to_excel(f"{dataset}_formulas_refs_based_3_4_avg.xlsx", index=False)

def calucluate_single_model_corr_multi(dataset):
    """
    To keep ******
    bartscore-style score,by lang  
    """
    out = pd.DataFrame()
    models = []
    scores_en_de = []
    scores_zh_en = []
    scores_he_en = []
    scores_avg = []
    scores_avg_all = [] 

    for model_family in ["llama", "qw"]:
        if model_family == 'llama':
            model_names = ["llama32_1B_instruct", "llama32_3B_instruct", "llama31_8B_instruct"]
        else:
            model_names = ["qw25_0.5B","qw25_3B", "qw25_7B", ]


        
        data1 = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}_all_bartscore_{model_family}_refs_3_1.xlsx')#dataset}_all_bartscore_{model_family}_2_13
        data2 = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}_all_bartscore_{model_family}_refs_3_1_2.xlsx')#dataset}_all_bartscore_{model_family}_2_13
        data3 = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}_all_bartscore_{model_family}_refs_3_1_3.xlsx')#dataset}_all_bartscore_{model_family}_2_13
        data4 = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}_all_bartscore_{model_family}_refs_3_1_4.xlsx')#dataset}_all_bartscore_{model_family}_2_13
        data4 = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}_all_bartscore_{model_family}_refs_3_1_4.xlsx')#dataset}_all_bartscore_{model_family}_2_13
        data5 = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}_all_bartscore_{model_family}_ref_2_26.xlsx')#dataset}_all_bartscore_{model_family}_2_13
        data_all = [data1, data2, data3, data4,data5]   
        #all_bartscore_{model_family}_noT.xlsx')  wmt
        for corr in ["pearson",]:#"kendall","spearman"
            for index in [[0],[1],[2],[3],[4],[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[2,3],[2,4],[0,1,2],[1,2,3],[0,1,3],[0,2,3],[0,2,4],[0,1,2,3],[0,1,2,3,4]]:
                        #                                
                        # models.append(f"{large}_{small}_{formula}_{index_tag}")
                        # # out = data1["mqm-score"]
                        # data = pd.DataFrame()
                        # for i in index:
                        #     data = pd.concat([data,data_all[i][f"{large}_{small}_{formula}"]], axis=1)
                        # temp = data.mean(axis=1).rename(f"{large}_{small}_{formula}")
                        # data = pd.concat([temp, data1[["mqm-score","lp"]]], axis=1)

                for model in model_names:
                    index_tag= "_".join([str(x) for x in index]) 
                    models.append(f"{model}_{index_tag}")
                    data = pd.DataFrame()
                    for i in index:
                        data = pd.concat([data,data_all[i][f"{model}"]], axis=1)
                    temp = data.mean(axis=1).rename(f"{model}")
                    data = pd.concat([temp, data1[["mqm-score","lp"]]], axis=1)

                    if dataset == "wmt23":
                        score_en_de = data.groupby("lp")[f"{model}"].corr(data["mqm-score"],method=corr)["en-de"]
                        score_zh_en = data.groupby("lp")[f"{model}"].corr(data["mqm-score"],method=corr)["zh-en"]
                        score_he_en = data.groupby("lp")[f"{model}"].corr(data["mqm-score"],method=corr)["he-en"]
                        score_avg = (score_en_de + score_zh_en + score_he_en)/3.0
                        scores_en_de.append(score_en_de)
                        scores_zh_en.append(score_zh_en)
                        scores_he_en.append(score_he_en)
                        scores_avg.append(score_avg)
                        scores_avg_all.append(data[f"{model}"].corr(data["mqm-score"],method=corr))
    if dataset == "wmt23":
        out = pd.DataFrame({"model":models, "en_de":scores_en_de, "zh_en":scores_zh_en, "he_en":scores_he_en, "avg":scores_avg,"avg_all":scores_avg_all})
    # else:
    #     out = pd.DataFrame({"model":models, "coherence":scores_coherence,"consistency":scores_consistency,"fluency":scores_fluency,"relevance":scores_relevance, "avg":scores_avg})

    out.to_excel(f"./{dataset}_bartscore_corr_ref_3_4_avg.xlsx")#2_13



def print_var():
    data = pd.read_excel("~/tasks/contrastive-evaluation/wmt23.xlsx")
    data_selected = data[(data["lp"]=="zh-en")&(data["word_len"]<20)]
    print(data_selected.groupby("src")["mqm-score"].std().sort_values(ascending=False)[:20])


def negative_print():
    for dataset in [  "summeval", "wmt23", ]:# ,   "summeval" "wmt23",        
        for model_family in ["llama", "qw"]:#
            # data = pd.read_excel(f"~/tasks/contrastive-evaluation/{dataset}.xlsx")
            # data["word_len"] = [len(x.split(" "))for x in data["output"]]
            # data = pd.read_excel(f"~/tasks/contrastive-evaluation/{dataset}_{model_family}_formulas_negative_ratios_3_20.xlsx")

            data = pd.read_excel(f"~/tasks/contrastive-evaluation/{dataset}_{model_family}_formulas_direct_p1_3_25.xlsx")

            # print(data["word_len"].describe())
            # data.describe().to_excel(f"~/tasks/contrastive-evaluation/{dataset}_{model_family}_formulas_negative_ratios_3_20_statics.xlsx")
            print(data.describe())


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

def online_test_formula(model_family,dataset,formulas):
    """
    calculate different formulas
    """
    data = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}.xlsx')
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
        for formula in tqdm(formulas):   
            print(model_large_name, model_small_name, formula)            
            data[f"{large}_{small}_{formula}"] = compute_conditional_score(
                                                model = model_large,
                                                prompt_texts = list(data['prompt']),#_ref_new
                                                target_texts =list(data["output"]), 
                                                tokenizer = tokenzier,
                                                batch_size = 16,
                                                device = "cuda",
                                                max_length = max_len, 
                                                contrast_model = model_small,
                                                contrast_mode = formula, 
                                                temperature = 1.5,
                                                temperature_contrast= 0.5,
                                                gamma = 0.1   
                                                )
    data.to_excel(f'~/tasks/contrastive-evaluation/{dataset}_{model_family}_formulas_online_3_24.xlsx',index=False)  #formulas_refs_based_3_1_4


def cal_online_bartscore_model(model_family, dataset):   
    """
    single model
    """
    data = pd.read_excel(f'~/tasks/contrastcore/{dataset}.xlsx')

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

    data.to_excel(f'~/tasks/contrastive-evaluation/{dataset}_all_bartscore_{model_family}_online_3_18.xlsx',index=False)   



def find_instances(dataset, model_family):

    if model_family == "llama":
        single_c = [ 'llama32_1B_instruct', 'llama32_3B_instruct', 'llama31_8B_instruct']
    else:
        single_c = ['qw25_0.5B',  'qw25_3B', 'qw25_7B']
    
    data_single = pd.read_excel(f'~/tasks/contrastive-evaluation/{dataset}_all_bartscore_{model_family}_tem_15_3_27.xlsx')[single_c]
    

    data_single.columns = [f"{col}_temp" for col in data_single.columns]

    data = pd.read_excel(f"./{dataset}_{model_family}_merge.xlsx")

    if model_family == "llama":
        select_c = ['mt', 'system', 'mqm-score', 'src', 'ref', 'doc_id', 'lp', 'word_len',\
        'prompt', 'llama32_1B_instruct', 'llama32_3B_instruct', 'llama31_8B_instruct','8B_3B_direct_abs', '8B_1B_direct_abs',  '3B_1B_direct_abs']
    else:
        select_c = ['mt', 'system', 'mqm-score', 'src', 'ref', 'doc_id', 'lp', 'word_len',\
       'prompt', 'qw25_0.5B',  'qw25_3B', 'qw25_7B', '7B_3B_direct_abs',  '7B_0.5B_direct_abs',  '3B_0.5B_direct_abs']

    data = data[select_c]
    data = pd.concat([data,data_single], axis=1)

    if model_family == "llama":
        data_c = ['llama32_1B_instruct', 'llama32_3B_instruct', 'llama31_8B_instruct','llama32_1B_instruct_temp', 'llama32_3B_instruct_temp', 'llama31_8B_instruct_temp', '8B_3B_direct_abs','8B_1B_direct_abs', '3B_1B_direct_abs']
        model_large_names = ["llama31_8B_instruct", "llama31_8B_instruct","llama32_3B_instruct"]
        model_small_names = ["llama32_3B_instruct","llama32_1B_instruct","llama32_1B_instruct" ]
    else:
        data_c = ['qw25_0.5B',  'qw25_3B', 'qw25_7B', 'qw25_0.5B_temp',  'qw25_3B_temp', 'qw25_7B_temp','7B_3B_direct_abs', '7B_0.5B_direct_abs', '3B_0.5B_direct_abs']
        model_large_names = ["qw25_7B", "qw25_7B", "qw25_3B",] #
        model_small_names = ["qw25_3B", "qw25_0.5B", "qw25_0.5B",] #
    
    data["human_rank"] = data.groupby('src')["mqm-score"].rank()#method='average'
    for column in data_c:

        print(column)

        data[f"{column}_rank"] = data.groupby('src')[column].rank()
        data[f"{column}_rank_minus_human"] = data[f"{column}_rank"] - data["human_rank"] 
    # data_selected = data[data[""]]

    
    for model_large_name, model_small_name in zip(model_large_names,model_small_names):
        large, small = model_large_name.split("_")[1], model_small_name.split("_")[1]                    
        data_select = data[abs(data[f"{large}_{small}_direct_abs_rank_minus_human"])<abs(data[f"{model_large_name}_temp_rank_minus_human"])]
        print(model_large_name, model_small_name ,data_select.shape)
        # print(data_select)

    # data['rank_in_group'] = data.groupby('src')['value'].rank(method='dense')
    data.to_excel(f"./{dataset}_{model_family}_temp_merge_rank_default.xlsx", index=False)

def find_instances_(dataset,model_family):

    

    if model_family == "llama":
        data_c = ['llama32_1B_instruct', 'llama32_3B_instruct', 'llama31_8B_instruct','llama32_1B_instruct_temp', 'llama32_3B_instruct_temp', 'llama31_8B_instruct_temp', '8B_3B_direct_abs','8B_1B_direct_abs', '3B_1B_direct_abs']
        model_large_names = ["llama31_8B_instruct", "llama31_8B_instruct","llama32_3B_instruct"]
        model_small_names = ["llama32_3B_instruct","llama32_1B_instruct","llama32_1B_instruct" ]
    else:
        data_c = ['qw25_0.5B',  'qw25_3B', 'qw25_7B', 'qw25_0.5B_temp',  'qw25_3B_temp', 'qw25_7B_temp','7B_3B_direct_abs', '7B_0.5B_direct_abs', '3B_0.5B_direct_abs']
        model_large_names = ["qw25_7B", "qw25_7B", "qw25_3B",] #
        model_small_names = ["qw25_3B", "qw25_0.5B", "qw25_0.5B",] #
    
    
    for model_large_name, model_small_name in zip(model_large_names,model_small_names):
        large, small = model_large_name.split("_")[1], model_small_name.split("_")[1]     
        data = pd.read_excel(f'{dataset}_{model_family}_rank_{large}_{small}.xlsx')
        data["human_rank"] = data.groupby('src')["mqm-score"].rank()#method='average'
        for column in [f"{large}",f"{small}",f"{large}_{small}_direct_abs"]:

            print(column)

            data[f"{column}_rank"] = data.groupby('src')[column].rank()
            data[f"{column}_rank_minus_human"] = data[f"{column}_rank"] - data["human_rank"] 
        data_select = data[abs(data[f"{large}_{small}_direct_abs_rank_minus_human"])<abs(data[f"{large}_rank_minus_human"])]
        print(model_large_name, model_small_name ,data_select.shape)
        data.to_excel(f"./{dataset}_{model_family}_rank_3_30.xlsx", index=False)
        # print(data_select)

    # data['rank_in_group'] = data.groupby('src')['value'].rank(method='dense')
    data.to_excel(f"./{dataset}_{model_family}_temp_merge_rank_default.xlsx", index=False)


if __name__ == "__main__":
    # 1.read data
    # process_data()  
    # 2. evaluate using llm

    # for model_name in ["llama32_1B_instruct", "llama32_3B_instruct", "llama31_8B_instruct"]:
    #     print(model_name, "llm evaluate***********************")
    #     evaluate_llm(model_family, model_name)

    # 3.calculate BARTScore
    # cal_BARTScore()
    # for model_name in ["llama32_1B_instruct", "llama32_3B_instruct", "llama31_8B_instruct"]:
    #     print(model_name, "llama bart!*******************")
    #     cal_bartscore_model(model_family, model_name)
    

    # 4. calualte correlation 
    # dataset = "wmt23"
    # for model_family in ["llama", "qw"]:
    # #     combine_data(dataset, model_family)
    model_family = "qw"
    model_large_name, model_small_name =  'qw25_3B', 'qw25_0.5B'

    # model_family = "llama"
    # model_large_name, model_small_name = "llama32_3B_instruct", "llama32_1B_instruct"
    error_analysis(model_family, model_large_name, model_small_name)

    """

    for model_family in ["llama" ,"qw"]: #,"qw"
        if model_family == "llama":
            model_large_names = ["llama32_3B_instruct","llama31_8B_instruct", "llama31_8B_instruct",]
            model_small_names = ["llama32_1B_instruct","llama32_1B_instruct","llama32_3B_instruct", ]
        else:
            model_large_names = ["qw25_3B","qw25_7B", "qw25_7B", ] #
            model_small_names = ["qw25_0.5B", "qw25_0.5B","qw25_3B",] #
        # model_family = "llama"
        # model_large_name, model_small_name = "llama32_3B_instruct", "llama32_1B_instruct"
        for model_large_name, model_small_name in zip(model_large_names,model_small_names):
            print(model_large_name, model_small_name)
            error_analysis(model_family, model_large_name, model_small_name)
    """
    # bleu_chrf()
    
    # process_summeval_data()

    # rank 
    # for dataset in [ "wmt23"]:
    #     for model_family in ["llama","qw"]:
    #         find_instances_(dataset, model_family)


    #         test_gamma(model_family,dataset)
    #         cal_pearson_gamma(model_family,dataset)

    # negative_print()
    # for CE score 
    """
    for dataset in [ "wmt23",]:# ,   "summeval" "wmt23",,"summeval"
        formulas = ["direct_abs"]#,"plus_condition","division" "ensemble", "division_head_direct"
        
        for model_family in ["llama", "qw"]:#"llama",
            test_formula(model_family,dataset, formulas)
            # online_test_formula(model_family,dataset, formulas)
        # formulas_CE_corr_multi_wmt(dataset,formulas)
            # test_formula_multi(model_family,dataset,formulas)
        # formulas_CE_corr_multi(dataset,formulas)
        # formulas_CE_corr(dataset,formulas)
    """
    # print_var() 
    '''
    for dataset in ["wmt23"]:# ,   "summeval" "wmt23","summeval"
        
        for model_family in ["llama", "qw"]:#
            cal_bartscore_model(model_family, dataset)
            # cal_online_bartscore_model(model_family, dataset)
        # calucluate_single_model_corr(dataset)
        # calucluate_single_model_corr_multi(dataset)
    '''
    # for dataset in [ "wmt23","summeval"]:# ,   "summeval"   "summeval"
    #     # formulas = ["direct_zero","direct_abs","condition_direct_zero","condition_direct_abs"]
    #     # formulas = ["plus", "plus_nafise","plus_condition","minus", "minus_nafise","minus_condition"]
    #     # formulas = ["plus","plus_condition","direct_abs","condition_direct_abs"]
    #     # formulas = ["plus_baseline"]
    #     formulas = ["plus_baseline","direct_abs","plus_condition",]
    #     for model_family in ["llama","qw"]:

    #         # test_gamma(model_family,dataset)
    #         # cal_pearson_gamma(model_family,dataset)

    #         test_formula(model_family,dataset, formulas)

    #     formulas_CE_corr(dataset,formulas)
            # cal_pearson_gamma_by_lang(model_family)
            # cal_bartscore_model(model_family,dataset)
            # test_temperature(model_family,dataset)
            # cal_corr_temperature_by_lang(model_family)

            # cal_pearson_gamma_by_lang(model_family)
            # test_temperature(model_family)
            # cal_corr_temperature_by_lang(model_family)
            # get_token_len()tra    
            # model_large_name, model_small_name = "llama31_8B_instruct", "llama32_1B_instruct"
            # error_analysis(model_family, model_large_name, model_small_name)
            # recalucluate_corr()
            # cal_bartscore_model_complete(model_family)
            # process_summeval_data()
            # formulas_CE_corr_by_lang(dataset,formulas)
            # recalucluate_single_model_corr(dataset)

            
