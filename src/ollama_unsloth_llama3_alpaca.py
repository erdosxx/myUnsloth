from datasets import load_dataset
from transformers import TextStreamer, TrainingArguments
from trl import SFTTrainer
from unsloth import (
    FastLanguageModel,
    apply_chat_template,
    is_bfloat16_supported,
    standardize_sharegpt,
    to_sharegpt,
)
import torch

# Choose any! We auto support RoPE Scaling internally!
max_seq_length = 2048

# None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
dtype = None

# Use 4bit quantization to reduce memory usage. Can be False.
load_in_4bit = True

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
# More models at https://huggingface.co/unsloth
fourbit_models = [
    # New Mistral v3 2x faster!
    "unsloth/mistral-7b-v0.3-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-bnb-4bit",
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    # Phi-3 2x faster!
    "unsloth/Phi-3-mini-4k-instruct",
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    # Gemma 2.2x faster!
    "unsloth/gemma-7b-bnb-4bit",
]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # use one if using gated models like meta-llama/Llama-2-7b-hf
    # token = "hf_...",
)

# We now add LoRA adapters so we only need to
# update 1 to 10% of all parameters!
model = FastLanguageModel.get_peft_model(
    model,
    # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    # Supports any, but = 0 is optimized
    lora_dropout=0,
    # Supports any, but = "none" is optimized
    bias="none",
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    # True or "unsloth" for very long context
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    # We support rank stabilized LoRA
    use_rslora=False,
    # And LoftQ
    loftq_config=None,
)

# Data Prep
dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")
print(dataset.column_names)

dataset = to_sharegpt(
    dataset,
    merged_prompt="{instruction}[[\nYour input is:\n{input}]]",
    output_column_name="output",
    conversation_extension=3,  # Select more to handle longer conversations
)

dataset = standardize_sharegpt(dataset)

alpaca_prompt = """
Below is an instruction that describes a task, paired with
an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

chat_template = """{SYSTEM}
USER: {INPUT}
ASSISTANT: {OUTPUT}"""

# Llama-3 prompt format
chat_template = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|>"""

# ChatML format
chat_template = """
<|im_start|>system
{SYSTEM}<|im_end|>
<|im_start|>user
{INPUT}<|im_end|>
<|im_start|>assistant
{OUTPUT}<|im_end|>"""

chat_template = """
Below are some instructions that describe some tasks.
Write responses that appropriately complete each request.

### Instruction:
{INPUT}

### Response:
{OUTPUT}"""

dataset = apply_chat_template(
    dataset,
    tokenizer=tokenizer,
    chat_template=chat_template,
    # default_system_message = "You are a helpful assistant", << [OPTIONAL]
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    # Can make training 5x faster for short sequences.
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        # num_train_epochs = 1, # For longer training runs!
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(
    torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} "
    + "minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(
    f"Peak reserved memory for training % of max memory = {lora_percentage} %."
)

# Enable native 2x faster inference
FastLanguageModel.for_inference(model)
messages = [  # Change below!
    {
        "role": "user",
        "content": "Continue the fibonacci sequence! Your input is "
        + "1, 1, 2, 3, 5, 8,",
    },
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    input_ids,
    streamer=text_streamer,
    max_new_tokens=128,
    pad_token_id=tokenizer.eos_token_id,
)

# Enable native 2x faster inference
FastLanguageModel.for_inference(model)
messages = [  # Change below!
    {
        "role": "user",
        "content": "Continue the fibonacci sequence! Your input is "
        + "1, 1, 2, 3, 5, 8",
    },
    {
        "role": "assistant",
        "content": "The fibonacci sequence continues as "
        + "13, 21, 34, 55 and 89.",
    },
    {"role": "user", "content": "What is France's tallest tower called?"},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    input_ids,
    streamer=text_streamer,
    max_new_tokens=128,
    pad_token_id=tokenizer.eos_token_id,
)

model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving

if False:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="lora_model",  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
pass

messages = [  # Change below!
    {
        "role": "user",
        "content": "Describe anything special about a sequence. "
        + "Your input is 1, 1, 2, 3, 5, 8,",
    },
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")


text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    input_ids,
    streamer=text_streamer,
    max_new_tokens=128,
    pad_token_id=tokenizer.eos_token_id,
)

if False:
    # I highly do NOT suggest - use Unsloth if possible
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoPeftModelForCausalLM.from_pretrained(
        "lora_model",  # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit=load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained("lora_model")

# Save to 8bit Q8_0
if True:
    model.save_pretrained_gguf(
        "model",
        tokenizer,
    )
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False:
    model.push_to_hub_gguf("hf/model", tokenizer, token="")

# Save to 16bit GGUF
if False:
    model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")
if False:
    model.push_to_hub_gguf(
        "hf/model", tokenizer, quantization_method="f16", token=""
    )

# Save to q4_k_m GGUF
if False:
    model.save_pretrained_gguf(
        "model", tokenizer, quantization_method="q4_k_m"
    )
if False:
    model.push_to_hub_gguf(
        "hf/model", tokenizer, quantization_method="q4_k_m", token=""
    )

# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        "hf/model",  # Change hf to your username!
        tokenizer,
        quantization_method=[
            "q4_k_m",
            "q8_0",
            "q5_k_m",
        ],
        token="",  # Get a token at https://huggingface.co/settings/tokens
    )
