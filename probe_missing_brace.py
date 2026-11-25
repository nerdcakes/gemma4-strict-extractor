"""Bounded suffix census and eight generation traces; stdout JSON."""
import json

def self_test():
    assert suffix('2029.\"]}}')=='period_quote'
    assert suffix('2029\"]}}')=='nonperiod_quote'
    assert suffix('[]}}')=='empty_evidence'
    assert suffix('bad')=='other'

import argparse
import collections
import hashlib
from pathlib import Path
ROOT=Path(__import__("os").environ.get("PROJECT_DIR",str(Path(__file__).resolve().parent))).resolve()

def suffix(text):
    if text.endswith('.\"]}}'): return 'period_quote'
    if text.endswith('\"]}}'): return 'nonperiod_quote'
    if text.endswith('[]}}'): return 'empty_evidence'
    return 'other'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8388608),b''): h.update(chunk)
    s=path.stat()
    return {'sha256':h.hexdigest(),'bytes':s.st_size,'uid':s.st_uid,'gid':s.st_gid,'mode':s.st_mode}

def load_inputs():
    files=['grader.py','eval_model.py','generate_data.py','generate_stress_data.py','data/train_sft.jsonl','data/val.jsonl','data/eval_test.jsonl','data/eval_stress.jsonl']
    files += ['results/'+m+'_gemma4_e4b_stress_'+s for m in ('base','sft') for s in ('preds.jsonl','summary.json')]
    files += [str(p.relative_to(ROOT)) for p in Path(__import__('os').environ.get('TOKENIZER_PATH','google/gemma-4-E4B-it')).rglob('*') if p.is_file()]
    return {p:digest(ROOT/p) for p in files}

def token_info(tok,text):
    ids=tok(text,add_special_tokens=False)['input_ids']
    return {'text':text,'ids':ids,'pieces':tok.convert_ids_to_tokens(ids),'decoded_each':[tok.decode([i],skip_special_tokens=False) for i in ids]}

def census(tok):
    from grader import parse
    sets={k:[json.loads(l) for l in (ROOT/'data'/f).read_text().splitlines()] for k,f in [('train','train_sft.jsonl'),('eval','eval_test.jsonl'),('stress','eval_stress.jsonl')]}
    pred={m:{p['id']:p for p in [json.loads(l) for l in (ROOT/'results'/(m+'_gemma4_e4b_stress_preds.jsonl')).read_text().splitlines()]} for m in ('base','sft')}
    report={}; per_row={}
    for split,rows in sets.items():
        counter=collections.Counter(); detail=[]
        for row in rows:
            canonical=json.dumps(row['ground_truth'],ensure_ascii=False)
            if split=='train':
                assert row['prompt'][-1]['role']=='assistant'
                target=row['prompt'][-1]['content']; assert target==canonical,'training serialization mismatch'
            else: target=canonical
            g=row['ground_truth']['effective_date']; period=bool(g['evidence'] and g['evidence'][-1].endswith('.'))
            counter[(suffix(target),g['status'],period,target[-6:])]+=1
            detail.append({'id':row['id'],'suffix_category':suffix(target),'effective_date_status':g['status'],'final_evidence_period':period,'last6':target[-6:]})
        report[split]={'records':len(rows),'cross':[{'suffix_category':k[0],'effective_date_status':k[1],'final_evidence_period':k[2],'last6':k[3],'count':v} for k,v in sorted(counter.items())],'category_counts':dict(collections.Counter(d['suffix_category'] for d in detail)),'assistant_target_serialization_exact':True if split=='train' else 'canonical json.dumps(ensure_ascii=False)'}
        assert sum(counter.values())==len(rows); per_row[split]=detail
    cross={}
    flags={}
    for model in ('base','sft'):
        c=collections.Counter(); flags[model]={}
        for row,d in zip(sets['stress'],per_row['stress']):
            text=pred[model][row['id']]['raw_output']; valid=parse(text) is not None; one=not valid and parse(text+'}') is not None
            state='valid' if valid else 'one_missing_outer_brace' if one else 'other_invalid'
            flags[model][row['id']]=state
            c[(d['suffix_category'],d['effective_date_status'],state,text[-6:])]+=1
        cross[model]=[{'GT_suffix':k[0],'GT_effective_date_status':k[1],'output_state':k[2],'predicted_last6':k[3],'count':v} for k,v in sorted(c.items())]
        assert sum(c.values())==200
    stress=sets['stress']; chosen=['stress-0000','stress-0001']
    chosen.append(next(r['id'] for r in stress if flags['base'][r['id']]=='valid' and flags['sft'][r['id']]=='one_missing_outer_brace'))
    chosen.append(next(r['id'] for r in stress if flags['base'][r['id']]=='valid' and flags['sft'][r['id']]=='valid'))
    assert len(set(chosen))==4,('selected IDs not distinct',chosen)
    templates=[]
    reps=[next(r for r in sets['train'] if r['ground_truth']['effective_date']['status']!='MISSING'),next(r for r in sets['train'] if r['ground_truth']['effective_date']['status']=='MISSING')]+stress[:2]
    for r in reps:
        target=r['prompt'][-1]['content'] if 'prompt' in r else json.dumps(r['ground_truth'],ensure_ascii=False)
        msgs=r['messages']+[{'role':'assistant','content':target}]
        full=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=False,enable_thinking=False)
        ids=tok(full,add_special_tokens=False)['input_ids']
        templates.append({'id':r['id'],'effective_date_status':r['ground_truth']['effective_date']['status'],'target_last6':target[-6:],'templated_last10_ids':ids[-10:],'templated_last10_pieces':tok.convert_ids_to_tokens(ids[-10:]),'templated_last10_decoded':tok.decode(ids[-10:],skip_special_tokens=False)})
    probes=['2029.\"]}}','2029\"]}}','[]}}','2029.\"]}','2029\"]}','[]}']
    return {'census':report,'stress_crosstabs':cross,'literal_tokenizations':[token_info(tok,x) for x in probes],'templated_completion_tails':templates,'selected_ids':chosen,'selected_original_batch4_context':{rid:[r['id'] for r in stress[(i//4)*4:(i//4)*4+4]] for i,r in enumerate(stress) for rid in [r['id']] if rid in chosen},'limits':'Census canonical targets preserves json.dumps default spaces and insertion order; train assistant target verified byte-identical. Suffix categories exclusive. New selected4 batch differs from original contiguous batch4 contexts; compare decoded outputs explicitly.'}

def gpu_trace(cpu):
    import torch
    from transformers import AutoProcessor,AutoConfig,AutoModelForMultimodalLM
    from eval_model import render_prompts,attention_for
    rows={r['id']:r for r in [json.loads(l) for l in (ROOT/'data/eval_stress.jsonl').read_text().splitlines()]}
    selected=[rows[k] for k in cpu['selected_ids']]
    result={}
    for label,path in [('base','google/gemma-4-E4B-it'),('sft',__import__('os').environ.get('TOKENIZER_PATH','google/gemma-4-E4B-it'))]:
        proc=AutoProcessor.from_pretrained(path,local_files_only=True); tok=proc.tokenizer; tok.padding_side='left'
        if tok.pad_token_id is None: tok.pad_token=tok.eos_token
        cfg=AutoConfig.from_pretrained(path,local_files_only=True)
        model=AutoModelForMultimodalLM.from_pretrained(path,config=cfg,torch_dtype=torch.bfloat16,attn_implementation=attention_for(cfg),device_map='cuda',local_files_only=True)
        model.eval()
        prompts=render_prompts(proc,selected); inputs=tok(prompts,return_tensors='pt',padding=True).to('cuda')
        with torch.inference_mode(): out=model.generate(**inputs,do_sample=False,max_new_tokens=768,pad_token_id=tok.pad_token_id,return_dict_in_generate=True,output_scores=True)
        continuation=out.sequences[:,inputs['input_ids'].shape[1]:]
        eos=model.generation_config.eos_token_id; stop_ids=[eos] if isinstance(eos,int) else list(eos or [])
        old={p['id']:p['raw_output'] for p in [json.loads(l) for l in (ROOT/'results'/(label+'_gemma4_e4b_stress_preds.jsonl')).read_text().splitlines()]}
        candidate_strings=['}','}}','\"]}','\"]}}','[]}','[]}}','<turn|>','<eos>','<|think|>']
        candidates=[token_info(tok,x) for x in candidate_strings]
        traces=[]
        for i,row in enumerate(selected):
            ids=continuation[i].tolist(); stop_index=next((j for j,x in enumerate(ids) if x in stop_ids),None)
            actual=ids[:stop_index+1] if stop_index is not None else ids
            trace={'id':row['id'],'all_actual_ids_through_first_stop':actual,'generated_length_through_first_stop':len(actual),'later_batch_padding_excluded':len(ids)-len(actual),'first_stop_index':stop_index,'stop_id':actual[-1] if stop_index is not None else None,'last10_ids':actual[-10:],'last10_pieces':tok.convert_ids_to_tokens(actual[-10:]),'last10_decode_with_specials':tok.decode(actual[-10:],skip_special_tokens=False),'steps':[]}
            decoded=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            trace['decoded_output']=decoded; trace['saved_output_exact_match']=decoded==old[row['id']]; trace['saved_output']=old[row['id']]
            for step in sorted(set(j for j in [len(actual)-2,len(actual)-1] if j>=0)):
                probs=torch.softmax(out.scores[step][i].float(),dim=-1); values,indices=torch.topk(probs,5)
                candidate_token_ids=sorted(set([x['ids'][0] for x in candidates if x['ids']]+stop_ids))
                trace['steps'].append({'generation_step_zero_based':step,'selected_id':actual[step],'top5':[{'id':t,'piece':tok.convert_ids_to_tokens(t),'decoded':tok.decode([t],skip_special_tokens=False),'probability':p} for t,p in zip(indices.tolist(),values.tolist())],'candidate_first_token_probabilities':[{'id':t,'piece':tok.convert_ids_to_tokens(t),'probability':probs[t].item()} for t in candidate_token_ids],'stop_id_probability_mass':sum(probs[t].item() for t in set(stop_ids)),'single_token_closing_brace_candidate_mass':sum(probs[t].item() for t in set(x['ids'][0] for x in candidates[:6] if len(x['ids'])==1))})
            traces.append(trace)
        result[label]={'model_path':path,'resolved_attention':{n:getattr(getattr(model.config,n,None),'_attn_implementation',None) for n in ['text_config','vision_config','audio_config'] if getattr(model.config,n,None) is not None},'eos_stop_ids':stop_ids,'pad_token_id':tok.pad_token_id,'candidate_tokenizations':candidates,'probability_semantics':'Scores are processed generation next-token scores, softmax FP32. Candidate probabilities are FIRST TOKEN ONLY; multi-token suffix sequence probabilities are NOT computed. Single-token closing-brace candidate mass deduplicates token IDs and excludes multi-token strings. Stop mass deduplicates configured EOS IDs. No extra forward/generation for candidate sequences.','batch_size':4,'max_new_tokens':768,'do_sample':False,'enable_thinking':False,'traces':traces}
        del out,continuation,inputs,model,proc; import gc; gc.collect(); torch.cuda.empty_cache()
    assert sum(len(v['traces']) for v in result.values())==8
    return result

if __name__=='__main__':
    self_test()
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',choices=['cpu','gpu'],required=True); args=parser.parse_args()
    before=load_inputs()
    if args.phase=='cpu':
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained('google/gemma-4-E4B-it',local_files_only=True)
        report=census(tok); report['input_hashes']=before
        report['evaluator_behavior']='eval_model.py:10 add_generation_prompt=True,enable_thinking=False;65 left-padded tokenizer;67 greedy,max_new_tokens768,no explicit EOS override;68 continuation slice;69 skip_special_tokens=True. Probe inherits model generation_config EOS; first stop IDs captured, later batch padding excluded. No raw-ID evidence available from original saved text.'
    else:
        cpu=json.loads((ROOT/'brace_probe_cpu.json').read_text()); report={'cpu':cpu,'gpu':gpu_trace(cpu),'input_hashes_before':before}
    after=load_inputs(); assert after==before,'frozen input hash/metadata changed'
    report['input_hashes_unchanged']=True
    print(json.dumps(report,ensure_ascii=False,indent=2))
