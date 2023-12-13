from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
model_name = "/itrex/neural-chat-7b-v1-1"     # Hugging Face model_id or local model
prompt = "Once upon a time, there existed a little girl,"

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

print(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, load_in_8bit=True, use_llm_runtime=False)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=30)
