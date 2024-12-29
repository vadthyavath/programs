import torch
import os
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
import composer
from composer.utils import dist
from composer.devices import DeviceGPU
from datasets import load_dataset
from composer import Trainer
from composer.models import HuggingFaceModel
from composer.optim import DecoupledAdamW, LinearWithWarmupScheduler
from torch.utils.data import DataLoader
from huggingface_hub import login
from datetime import datetime, timedelta
import time
from torch.utils.data.distributed import DistributedSampler
from composer.utils import dist

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant."
MAX_LENGTH = 4000
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01



def create_dataloaders(dataset, device_train_batch_size, device_eval_batch_size):
    # Create samplers for training and evaluation
    train_sampler = dist.get_sampler(
        dataset['latest'],
        shuffle=True,
        drop_last=True
    )
    
    eval_sampler = dist.get_sampler(
        dataset['latest'],
        shuffle=False,
        drop_last=False
    )

    # Create dataloaders with distributed samplers
    train_dataloader = DataLoader(
        dataset['latest'],
        batch_size=device_train_batch_size,
        sampler=train_sampler,
        pin_memory=True
    )
    
    eval_dataloader = DataLoader(
        dataset['latest'],
        batch_size=device_eval_batch_size,
        sampler=eval_sampler,
        pin_memory=True
    )
    
    return train_dataloader, eval_dataloader

# Model setup functions
def setup_tokenizer(model_name, auth_token):
    login(auth_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

def setup_model(model_name, tokenizer):
    torch_dtype = torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto', #f"cuda:{gpu_id}",
        quantization_config=quantization_config,
        # torch_dtype=torch_dtype
    )
    current_size = model.get_input_embeddings().weight.shape[0]
    if current_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    return model

# Data processing functions
def format_chat_template(row, tokenizer):
    row_json = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["input_question"]},
        {"role": "assistant", "content": row["output_parsed_answer"]}
    ]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

# Model class
class LlamaHFModel(HuggingFaceModel):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, batch):
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['input_ids']
        )
        return outputs

    def eval_forward(self, batch, outputs=None):
        return self.forward(batch)

    def loss(self, outputs, batch):
        return outputs.loss

def main():
     # Set GPU
    # gpu_id = 0  # Change this to use different GPU
    # torch.cuda.set_device(gpu_id)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=5400))
    from composer.utils import dist

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    gpu_id = 0
    device = torch.device(f'cuda:{gpu_id}')
    # torch.cuda.set_device(gpu_id)
    t = torch.cuda.get_device_properties(0).total_memory
        
    print("device_count: ", torch.cuda.device_count(), os.environ["CUDA_VISIBLE_DEVICES"], dist.get_world_size(), t)
    # region download dataset
    # composer.utils.dist.initialize_dist(DeviceGPU(), timeout=1000)
    # Initialize model and tokenizer
    tokenizer = setup_tokenizer(MODEL_NAME, os.environ.get('HF_TOKEN'))
    model = setup_model(MODEL_NAME, tokenizer)

    # Load and process dataset
    dataset = load_dataset('meta-llama/Llama-3.1-8B-Instruct-evals', 'Llama-3.1-8B-Instruct-evals__nexus__details')
    formatted_dataset = dataset.map(lambda x: format_chat_template(x, tokenizer))
    
    # Tokenize dataset
    tokenized_dataset = formatted_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=[
            'task_type', 'task_name', 'subtask_name', 'input_question', 
            'input_choice_list', 'input_final_prompts', 'input_correct_responses', 
            'output_prediction_text', 'output_parsed_answer', 'output_choice_completions', 
            'output_choice_negative_log_likelihoods', 'output_metrics', 'is_correct', 
            'input_question_hash', 'input_final_prompts_hash', 'benchmark_label', 
            'eval_config', 'text'
        ]
    )
    tokenized_dataset.set_format(type='torch')

    # Setup training
    composer_model = LlamaHFModel(model, tokenizer)
    optimizer = torch.optim.AdamW(
        composer_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    lr_scheduler = LinearWithWarmupScheduler(t_warmup='0.06dur', alpha_f=0.02)
    # Initialize trainer
    # gpu_id=2
    # device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'#torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    global_train_batch_size = 8
    global_test_batch_size = 8
    global_eval_batch_size = 8
    device_train_batch_size = global_train_batch_size // dist.get_world_size()
    device_test_batch_size = global_test_batch_size // dist.get_world_size()
    device_eval_batch_size = global_eval_batch_size // dist.get_world_size()
    train_dataloader, eval_dataloader = create_dataloaders(tokenized_dataset, device_train_batch_size, device_eval_batch_size)

    trainer = Trainer(
        model=composer_model,
        train_dataloader=train_dataloader, #DataLoader(tokenized_dataset['latest'], batch_size=1, shuffle=True),
        eval_dataloader=eval_dataloader, #DataLoader(tokenized_dataset['latest'], batch_size=1, shuffle=True),
        max_duration="2ba",
        load_weights_only=True,
        load_strict_model_weights=False,
        optimizers=optimizer,
        schedulers=[lr_scheduler],
        device='gpu' if torch.cuda.is_available() else 'cpu',
        precision='fp32',
        seed=17,
        save_interval='2ba',
        # device_train_microbatch_size='8'
        
    )

    # Train model
    trn_st = time.time()
    trainer.fit()
    trn_nd = time.time()
    training_time = str(timedelta(seconds=trn_nd - trn_st))
    print(f"Training and validation done. {training_time}")

if __name__ == "__main__":
    main()
