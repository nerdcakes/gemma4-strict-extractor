"""Greedy baseline evaluator; no training, no external generation/grade APIs."""
import argparse
import json
import time
from pathlib import Path
from grader import grade, aggregate


def render_prompts(processor, rows):
    prompts=[processor.apply_chat_template(r['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
    assert all('<|channel>thought' not in p for p in prompts), 'Thinking token unexpectedly emitted; stop'
    return prompts


def attention_for(cfg):
    if 'text_config' in getattr(cfg, 'sub_configs', {}) or hasattr(cfg, 'text_config'):
        mapping = {'text_config': 'sdpa'}
        for name, backend in [('vision_config', 'sdpa'), ('audio_config', 'eager')]:
            if getattr(cfg, name, None) is not None:
                mapping[name] = backend
        return mapping
    return 'sdpa'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_path',default='google/gemma-4-E4B-it')
    parser.add_argument('--eval_file',default='data/eval_test.jsonl')
    parser.add_argument('--run_name',default='base_gemma4_e4b')
    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--limit',type=int,default=None)
    args=parser.parse_args()
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor, AutoModelForMultimodalLM
    assert args.batch_size>0
    rows=[json.loads(l) for l in Path(args.eval_file).read_text().splitlines()]
    if args.limit: rows=rows[:args.limit]
    assert rows
    resultdir=Path('results'); resultdir.mkdir(exist_ok=True)
    output=resultdir/(args.run_name+'_preds.jsonl')
    summary_path=resultdir/(args.run_name+'_summary.json')
    assert not output.exists() and not summary_path.exists(), 'Refuse overwriting a prior benchmark'
    start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    processor=AutoProcessor.from_pretrained(args.model_path)
    tokenizer=processor.tokenizer
    tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    prompts=render_prompts(processor,rows)
    cfg=AutoConfig.from_pretrained(args.model_path)
    attn_map=attention_for(cfg)
    model=AutoModelForMultimodalLM.from_pretrained(args.model_path,config=cfg,torch_dtype=torch.bfloat16,attn_implementation=attn_map,device_map='cuda')
    resolved_attention={name:getattr(getattr(model.config,name,None),'_attn_implementation',None) for name in ['text_config','vision_config','audio_config'] if getattr(model.config,name,None) is not None}
    print('RESOLVED_ATTENTION:',json.dumps(resolved_attention),flush=True)
    if 'text_config' in resolved_attention:
        assert resolved_attention['text_config']=='sdpa', 'Text SDPA not active; stop'
    model.eval()
    torch.cuda.synchronize()
    loaded=time.perf_counter()
    scores=[]; generated_tokens=0
    with output.open('x',buffering=1) as stream:
        for begin in range(0,len(rows),args.batch_size):
            batch=rows[begin:begin+args.batch_size]
            inputs=tokenizer(prompts[begin:begin+args.batch_size],return_tensors='pt',padding=True).to('cuda')
            with torch.inference_mode():
                seq=model.generate(**inputs,do_sample=False,max_new_tokens=768,pad_token_id=tokenizer.pad_token_id)
            continuation=seq[:,inputs['input_ids'].shape[1]:]
            raw_outputs=tokenizer.batch_decode(continuation,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            generated_tokens+=int((continuation!=tokenizer.pad_token_id).sum())
            for row,raw in zip(batch,raw_outputs):
                score=grade(raw,row); scores.append(score)
                record={k:row[k] for k in ['id','source_text','ground_truth']}
                record.update({'raw_output':raw,'parsed_output':score['parsed_output'],'field_scores':score['field_scores'],'error_types':score['error_types']})
                stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            torch.cuda.synchronize()
            print(json.dumps({'completed':len(scores),'total':len(rows),'inference_seconds':round(time.perf_counter()-loaded,2),'peak_allocated_GiB':round(torch.cuda.max_memory_allocated()/2**30,3)}),flush=True)
    torch.cuda.synchronize()
    end=time.perf_counter()
    summary=aggregate(scores)
    summary.update({'model_path':args.model_path,'model_revision':getattr(model.config,'_commit_hash',None),'transformers_version':transformers.__version__,'torch_version':torch.__version__,'batch_size':args.batch_size,'max_new_tokens':768,'do_sample':False,'enable_thinking':False,'attention':resolved_attention,'dtype':'bfloat16','device':'cuda','load_seconds':loaded-start,'inference_seconds':end-loaded,'wall_clock_seconds':end-start,'peak_vram_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_vram_reserved_bytes':torch.cuda.max_memory_reserved(),'generated_tokens_nonpadding':generated_tokens})
    summary_path.write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
