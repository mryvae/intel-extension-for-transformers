from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
import cProfile
model_name = "Intel/neural-chat-7b-v1-1"     # Hugging Face model_id or local model
prompt = "Once upon a time, there existed a little girl,"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
# profiler = cProfile.Profile()
# profiler.enable()
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
# profiler.disable()
# profiler.dump_stats("profile_data_int8_runtime.prof")
