# ContrastScore

This is the Source Code of Paper: [ContrastScore: Towards Higher Quality, Less Biased, More Efficient Evaluation Metrics with Contrastive Evaluation](https://arxiv.org/abs/2504.02106).


## Setting up the Environment

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd contrast-score
   ```

2. Create and activate the conda environment using the provided YAML file:
   ```bash
   conda env create -f conda-env.yaml
   conda activate contrastscore
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Project Structure

- `contrast_score/`: Main package directory containing the scoring implementation
- `conda-env.yaml`: Conda environment specification
- `setup.py`: Package installation configuration

## Usage
Set up the parameters.

- `model`: the expert model.
- `contrast_model`: the amateur model.
- `contrast_mode`: `contrast` for ContrastScore, `ensemble` for ensemble methods.
- `prompt_texts`: prompts including task description or/and context or input.
- `target_texts`: texts to be evaluated. 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from contrast_score.score import compute_conditional_score

# Load model and tokenizer
expert_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
amateur_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
# the tokenizer of same model family is the same
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Example inputs
prompt_texts = [
    "Translate the following sentence to English:\n今天天气不错\n",
    "Write an accurate, relevant, and coherent summary of the following texts:\nManchester City are keen to sign Anderlecht teenager Evangelos Patoulidis. The 14-year-old playmaker is regarded as one of the best talents to emerge from Anderlecht's youth set-up and has also attracted attention from Arsenal and Barcelona. The Belgian starlet rejected a move to Barcelona's La Masia academy when he was 12 as his family wanted him to continue his studies. He has continued to impress and City have held discussions with Anderlecht chairman Roger Vanden Stock in the hope of agreeing a compensation package. Manuel Pellegrini is looked to build for the future by snapping up hot property Evangelos Patoulidis.\nSummary:\n",
]
target_texts = [
    "The weather is nice today",
    "evangelos patoulidis is regarded as one of the best players to emerge from anderlecht youth. he has also attracted attention from arsenal and barcelona. the belgian starlet rejected a move to barcelona's la masia academy . the 14-year-old has attracted interest from barcelona to barcelona.",
    ]

# Compute scores
scores = compute_conditional_score(
    model=model,
    contrast_model=amateur_model,
    contrast_mode="contrast",
    prompt_texts=prompt_texts,
    target_texts=target_texts,
    tokenizer=tokenizer,
    batch_size=2,
    device="cuda"  # Use "cpu" if CUDA is not available
)
```

## Additional Features

- Handles empty strings in both prompts and targets
- Automatically processes long sequences (truncated to max_length)
- Supports both CPU and CUDA devices
- Configurable batch sizes for efficient processing

## Bib
Please cite our work if you find it useful.
```
@article{wang2025contrastscore,
  title={ContrastScore: Towards Higher Quality, Less Biased, More Efficient Evaluation Metrics with Contrastive Evaluation},
  author={Wang, Xiao and Larionov, Daniil and Wu, Siwei and Liu, Yiqi and Eger, Steffen and Moosavi, Nafise Sadat and Lin, Chenghua},
  journal={arXiv preprint arXiv:2504.02106},
  year={2025}
}
```
