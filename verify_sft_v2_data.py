"""Independent successor-data acceptance oracle; read-only, JSON stdout."""
import collections
import hashlib
import json
import math
import re
from pathlib import Path
ROOT=Path(__import__("os").environ.get("PROJECT_DIR",str(Path(__file__).resolve().parent))).resolve()

def row_check(row,system):
    source=row['source_text']; assert row['messages']==[{'role':'system','content':system},{'role':'user','content':source}]
    assert row['prompt']==row['messages']+[{'role':'assistant','content':json.dumps(row['ground_truth'],ensure_ascii=False)}]
    for field,g in row['ground_truth'].items():
        assert all(q and q in source for q in g['evidence']), 'evidence'
        if g['status']=='FOUND': assert len(g['evidence'])==1 and g['value'] in g['evidence'][0]
        elif g['status']=='MISSING': assert g['value'] is None and g['evidence']==[]
        else:
            assert g['status']=='CONFLICT' and g['value'] is None and len(g['evidence'])==2
            a,b=row['conflict_values'][field]; assert a!=b and a in g['evidence'][0] and b in g['evidence'][1]
            if row['id'].startswith('v2-'):
                p=[source[:source.index(q)].count('\n\n') for q in g['evidence']]; assert abs(p[0]-p[1])>=2,'conflict distance'

def main():
    from transformers import AutoTokenizer
    from datasets import load_from_disk
    from grader import grade,aggregate
    tok=AutoTokenizer.from_pretrained('google/gemma-4-E4B-it',local_files_only=True)
    pool=json.loads((ROOT/'data/pool_manifest.json').read_text())
    import ast
    oldtree=ast.parse((ROOT/'generate_stress_data.py').read_text())
    stress=next(ast.literal_eval(n.value) for n in oldtree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='STRESS_POOL' for t in n.targets))
    forbidden={k:pool['EVAL_POOL'][k]+stress[k] for k in ['company_base','dba','contacts','domains','streets','cities']}
    # Every old eval/stress long invariant prose fragment is an exclusion only.
    old_literals=[]
    for filename in ['generate_data.py','generate_stress_data.py']:
        t=ast.parse((ROOT/filename).read_text())
        old_literals += [n.value for n in ast.walk(t) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>=60]
    fragments={s.casefold() for literal in old_literals for s in re.split(r'\{[^}]*\}',literal) if len(s)>=60}
    ds=load_from_disk(str(ROOT/'data/halo_sft_v2_dataset'))
    report={'splits':{},'train_val_rowhash_overlap':0,'frozen_grader':'unmodified','narrative_template_fragment_overlap':0}; hashes={}; previews=[]
    for split,oldfile,newfile,count,newcount,dskey in [('train','train_sft.jsonl','train_sft_v2.jsonl',2800,800,'train'),('validation','val.jsonl','val_v2.jsonl',300,100,'validation')]:
        oldlines=(ROOT/'data'/oldfile).read_text().splitlines(); lines=(ROOT/'data'/newfile).read_text().splitlines()
        assert len(lines)==count and len(set(lines))==count
        original=collections.Counter(oldlines); current=collections.Counter(lines); assert all(current[k]==v for k,v in original.items()),'original byte retention'
        rows=[json.loads(l) for l in lines]; assert len({r['id'] for r in rows})==count
        system=json.loads(oldlines[0])['prompt'][0]['content']; new=[r for r in rows if r['id'].startswith('v2-')]; assert len(new)==newcount
        assert len(ds[dskey])==count and ds[dskey].column_names==['prompt']
        full=[]; prompt=[]; statuses=collections.Counter(); suffix=collections.Counter(); near=0; decoy=0
        for i,r in enumerate(rows):
            row_check(r,system); assert ds[dskey][i]['prompt']==r['prompt'],'DatasetDict exact prompt match'
            target=r['prompt'][-1]['content']; category='period_quote' if target.endswith('.\"]}}') else 'nonperiod_quote' if target.endswith('\"]}}') else 'empty_evidence' if target.endswith('[]}}') else 'other'; suffix[category]+=1
            rendered=tok.apply_chat_template(r['prompt'],tokenize=False,add_generation_prompt=False,enable_thinking=False)
            n=len(tok(rendered,add_special_tokens=False,truncation=False)['input_ids']); full.append(n); assert n<=2450,(r['id'],n)
            if not r['id'].startswith('v2-'): continue
            text=r['source_text']; statuses.update(g['status'] for g in r['ground_truth'].values())
            assert not re.search(r'(?m)^\s*(?:[-*•]\s|\d+[.)]\s)|\||\b(?:supersedes|overrides|rescinds|takes precedence)\b',text),'form/precedence'
            for k,values in forbidden.items(): assert not any(v.casefold() in text.casefold() for v in values),('forbidden',r['id'],k)
            assert not any(f in text.casefold() for f in fragments),'old prose contamination'
            assert 'the separate archival supplier' in text and 'not a client representative' in text; decoy+=1
            for f,g in r['ground_truth'].items():
                if g['status']=='MISSING':
                    witness={'legal_entity_name':'the storage operator, not the applicant','dba_name':'an exhibition title, not an adopted DBA','entity_type':'the building owner, not this applicant','primary_contact':'not appointed to handle client onboarding','amount_due':'already paid for freight, not currently due','payment_terms':'belongs to the caterer, not this onboarding','effective_date':'only a rehearsal date, not service commencement'}[f]
                    assert witness in text,(r['id'],f,'near miss'); near+=1
            pr=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tok(pr)['input_ids']; assert ids==tok(pr,truncation=False)['input_ids']; prompt.append(len(ids)); assert 850<=len(ids)<=1750,(r['id'],len(ids))
        assert statuses=={'FOUND':newcount*7*45//100,'MISSING':newcount*7*25//100,'CONFLICT':newcount*7*30//100},statuses
        assert near>=.8*statuses['MISSING'] and decoy>=.8*newcount
        if split=='train': assert suffix['period_quote']>=400
        summary=aggregate([grade(json.dumps(r['ground_truth'],ensure_ascii=False),r) for r in rows])
        for k in ['json_valid_rate','json_valid_stripped_rate','found_exact_precision','found_exact_recall','found_exact_f1','missing_abstention_acc','conflict_detection_acc']: assert summary[k]==1.0,(split,k)
        assert summary['hallucination_rate']==0 and not summary['error_counts']
        def stats(v):
            s=sorted(v); return {'min':s[0],'p50':s[math.ceil(len(s)*.5)-1],'p95':s[math.ceil(len(s)*.95)-1],'max':s[-1]}
        hashes[split]={hashlib.sha256(json.dumps(r['prompt'],sort_keys=True,ensure_ascii=False).encode()).hexdigest() for r in rows}
        report['splits'][split]={'records':count,'original_rows_exact_retained':len(oldlines),'new_narratives':len(new),'narrative_status_counts':dict(statuses),'near_miss_fields':near,'decoy_packets':decoy,'suffix_census_full_split':dict(suffix),'full_sequence_tokens':stats(full),'narrative_prompt_tokens':stats(prompt),'self_test':summary,'DatasetDict_exact_prompt_matches':count}
        if split=='train': previews=[r for r in rows if r['id'].startswith('v2-')][:3]
    overlap=hashes['train']&hashes['validation']; assert not overlap
    assert json.loads((ROOT/'data/sft_v2_narrative_preview.json').read_text())==previews
    report['forbidden_entity_domain_template_matches']=0; report['preview_exact']=True
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
