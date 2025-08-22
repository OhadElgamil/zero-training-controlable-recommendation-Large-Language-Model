from transformers import AutoTokenizer
from data.data_interface import DInterface

#this script generates prompts using the same prompt generation LLaRA used for training
# we will use this prompts in out research

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

#this sets the amount of iteration we will do, higher values gives more prompts
Target_amount = 40

#fix an issue, dont know why but stackoverflow said to do so and it worked...
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#this should create the data module
data_module = DInterface(
    llm_tokenizer= tokenizer,
    dataset="movielens_data",
    batch_size = 8,
    max_epochs=1,
    num_workers=1,
    prompt_path="/home/ohad.elgamil/LLaRA/prompt/movie.txt"
)


train_loader = data_module.train_dataloader()

with open("/home/ohad.elgamil/LLaRA/prompt/train_prompts.txt", "w", encoding="utf-8") as f:
    for i, batch in enumerate(train_loader):
        texts = tokenizer.batch_decode(batch["tokens"]["input_ids"], skip_special_tokens=True)
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")
        if i >= Target_amount: #stop after 20 iterations, can change to whatever
            break 