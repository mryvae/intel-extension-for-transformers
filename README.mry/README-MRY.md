# 1 doc

线下推理文档

https://huggingface.co/docs/transformers/installation#offline-mode

transformer infer加速：

https://huggingface.co/docs/transformers/perf_infer_cpu

Intel® Extension for PyTorch

https://intel.github.io/intel-extension-for-pytorch/cpu/latest/index.html

# 2 trace

## 2.1 float

```python
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
model_name = "/itrex/neural-chat-7b-v1-1"     # Hugging Face model_id or local model
prompt = "Once upon a time, there existed a little girl,"

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

print(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name,local_files_only=True)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=30)
```

`python -m pdb neural_chat_float_inference.py`

modeling_auto.py: from_pretrained

​	if isinstance(quantization_config, MixedPrecisionConfig):

​		model = cls.ORIG_MODEL.from_pretrained

## 2.2 INT8

```python
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
```

modeling_auto.py: from_pretrained

​	use_llm_runtime = kwargs.pop("use_llm_runtime", False)

​	if isinstance(quantization_config, WeightOnlyQuantConfig):

​		model = cls.ORIG_MODEL.from_pretrained

成功！！！！

# 3 environment

## 3.1 docker

sudo docker run -itd --name chat_box_v1 --cap-add CAP_SYS_ADMIN --privileged -v /home/ruoyan/llm/itrex:/itrex intel/ai-tools:itrex-chatbot

sudo docker exec -it <container_id> /bin/bash

## 3.2 perf

安装

apt-get update

apt-get upgrade

apt-get install linux-perf

apt install linux-tools-generic

ln -sf /usr/lib/linux-tools/5.15.0-89-generic/perf /usr/bin/perf

使用 perf生成火焰图

**1，首先使用 perf record 命令记录进程的 CPU 使用情况**
命令：sudo perf record -a -g python neural_chat_int8_inference.py

python -X perf neural_chat_int8_inference.py

**2. 使用 perf script 工具对 perf.data 进行解析**
命令：sudo perf script -i perf.data &> perf.unfold
**3. 使用 Flame Graph 工具将 perf.unfold 中的符号折叠** //生成脚本文件
命令：sudo FlameGraph/stackcollapse-perf.pl perf.unfold &> perf.folded
**4. 使用 Flame Graph 工具将 perf.folded 生成 svg 火焰图**
命令：sudo FlameGraph/flamegraph.pl perf.folded > perf.svg 

## 3.3 cProfile

```python
import cProfile

def your_function():
    # 你的函数代码...

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    your_function()
    profiler.disable()
    profiler.dump_stats("profile_data.prof")
```

python cProfile_neural_chat_int8_inference.py

flameprof profile_data.prof > profile_data.svg  

OR snakeviz profile_data.prof

py-spy record -o profile.svg python neural_chat_int4_inference_origin.py

查看火焰图：https://www.modb.pro/db/144385

# other

模型镜像下载

https://blog.csdn.net/a61022706/article/details/134887159

https://zhuanlan.zhihu.com/p/646907543

https://hf-mirror.com/

```
git clone https://hf-mirror.com/EleutherAI/gpt-j-6b
```
