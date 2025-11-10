import transformers
print(transformers.__file__)
print(transformers.__version__)
# filepath: c:\Users\hazim\Desktop\Market-Sentiment-Analysis-PSM\test_args.py
import inspect
from transformers import TrainingArguments
#print(inspect.signature(TrainingArguments.__init__))
#print(dir(transformers))
from transformers import TrainingArguments
args = TrainingArguments(
    output_dir="test",
    eval_strategy="epoch"
)
print(args)

import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")