"""Export the final checkpoint to ONNX so the model can run in a web browser.

uv run --with onnx --with onnxruntime --with onnxscript export_onnx.py results_v1/model_step60000.pt web_model/

Writes model.onnx (fp32, the "Full" model of the website), model_int8.onnx (the "Light" model) and tokenizer.json.
Both models are hosted on the Hugging Face Hub (volodyaDuckDuck/mini-llada), the website downloads them from there.
"""
import json
import os
import sys

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from sample import load_model

ckpt, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
model = load_model(ckpt)
x = torch.full((1, 40), model.cfg.mask_id, dtype=torch.long)
torch.onnx.export(model, (x,), f"{out}/model.onnx", input_names=["ids"], output_names=["logits"],
                  dynamic_axes={"ids": {1: "T"}, "logits": {1: "T"}}, opset_version=17, dynamo=False)
quantize_dynamic(f"{out}/model.onnx", f"{out}/model_int8.onnx", weight_type=QuantType.QInt8)

t = json.load(open("data/tokenizer.json"))
json.dump({"vocab": t["model"]["vocab"], "merges": [" ".join(m) for m in t["model"]["merges"]],
           "mask_id": model.cfg.mask_id, "eos_id": t["model"]["vocab"]["[EOS]"], "seq_len": model.cfg.seq_len},
          open(f"{out}/tokenizer.json", "w"))
print({f: os.path.getsize(f"{out}/{f}") // 1e6 for f in os.listdir(out)})
