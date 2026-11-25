"""Merge the verified local SFT adapter into its original BF16 multimodal base."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor
from peft import PeftModel
from eval_model import attention_for


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--adapter',required=True)
    p.add_argument('--output',default='checkpoints/sft_gemma4_e4b_merged')
    args=p.parse_args()
    adapter=Path(args.adapter); out=Path(args.output)
    assert not out.exists(), 'Refuse overwriting merged output'
    meta=json.loads((adapter/'adapter_config.json').read_text())
    base=meta['base_model_name_or_path']
    assert base=='google/gemma-4-E4B-it',base
    revision='4ecc2ba63bcac23ba4a767fc87025b1450f84c9c'
    cfg=AutoConfig.from_pretrained(base,revision=revision,local_files_only=True)
    model=AutoModelForMultimodalLM.from_pretrained(base,revision=revision,config=cfg,dtype=torch.bfloat16,device_map='cpu',attn_implementation=attention_for(cfg),local_files_only=True)
    model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
    model=model.merge_and_unload(safe_merge=True)
    assert all(p.dtype==torch.bfloat16 for p in model.parameters() if p.is_floating_point()),'Non-BF16 parameter'
    model.save_pretrained(out,safe_serialization=True,max_shard_size='4GB')
    processor=AutoProcessor.from_pretrained(base,revision=revision,local_files_only=True)
    # Prefer the actual training processor/template, retaining base fallback for complete multimodal metadata.
    if (adapter/'processor_config.json').exists():
        processor=AutoProcessor.from_pretrained(str(adapter),local_files_only=True)
    processor.save_pretrained(out)
    assert (out/'config.json').exists() and (out/'chat_template.jinja').exists()
    print('MERGE_COMPLETE:',out,flush=True)


if __name__=='__main__': main()
