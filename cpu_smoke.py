"""CPU-only sandbox smoke; no model load or reserved data access."""
import ast,importlib,inspect,json,os
from pathlib import Path
from transformers import AutoTokenizer
root=Path(os.environ['PROJECT_DIR']);root.mkdir(exist_ok=True)
from generate_data import generate_split,TRAIN_POOL,SYSTEM_PROMPT
from grader import grade,aggregate
import random
rows=generate_split('cpu_smoke',4,TRAIN_POOL,random.Random(7))
scores=[grade(json.dumps(r['ground_truth']),r) for r in rows]
a=aggregate(scores)
assert a['json_valid_rate']==1 and a['found_exact_f1']==1 and a['hallucination_rate']==0
for name in ['generate_stress_data','generate_sft_v2_data','generate_prospective_data','eval_model','merge_adapter','probe_missing_brace','score_prospective_slices','verify_stress_data','verify_sft_v2_data','verify_prospective_data']:
    # merge imports torch/PEFT but does not load any model on import.
    mod=importlib.import_module(name)
    if hasattr(mod,'ROOT'):assert mod.ROOT==root
proc=AutoTokenizer.from_pretrained('google/gemma-4-E4B-it',local_files_only=True)
from eval_model import render_prompts
prompts=render_prompts(proc,rows)
lengths=[len(proc(p,truncation=False)['input_ids']) for p in prompts]
assert all(x>0 for x in lengths) and all('<|channel>thought' not in p for p in prompts)
# Exercise actual staged original generator export into sandbox only.
from generate_data import export,generate
# A full historical corpus is CPU/stdlb, but not needed for import/render smoke.
(root/'smoke_rows.json').write_text(json.dumps(rows))
print(json.dumps({'small_generator_records':4,'grader_selftest':a,'all12script_imports':True,'PROJECT_DIR_override':True,'offline_tokenizer_render_lengths':lengths,'model_loaded':False,'GPU_used':False,'full_regeneration_tested':False,'full_training_tested':False}))
