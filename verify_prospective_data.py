"""Prospective corpus acceptance; lockbox rows are never printed."""
import collections
import hashlib
import json
import math
import re
from pathlib import Path
ROOT=Path(__import__("os").environ.get("PROJECT_DIR",str(Path(__file__).resolve().parent))).resolve()

def field_check(row):
    source=row['source_text']
    for f,g in row['ground_truth'].items():
        assert all(q and q in source for q in g['evidence']),'evidence'
        if g['status']=='FOUND': assert len(g['evidence'])==1 and g['value'] in g['evidence'][0]
        elif g['status']=='MISSING': assert g['value'] is None and not g['evidence']
        else:
            a,b=row['conflict_values'][f]; assert a!=b and len(g['evidence'])==2
            assert a in g['evidence'][0] and b in g['evidence'][1]
            ix=[source[:source.index(q)].count('\n\n') for q in g['evidence']]; assert abs(ix[0]-ix[1])>=2,'separation'
        tags=row['field_tags'][f]
        if 'attribution_trap' in tags:
            w=row['attribution_witnesses'][f]; assert w['quote'] in source and w['value'] in w['quote']
            assert g['status'] in ['FOUND','MISSING'] and w['value']!=g['value'],'attribution'
        assert ('separated_conflict' in tags)==(g['status']=='CONFLICT')

def format_check(row):
    text=row['source_text']
    blocks=re.findall(r'(?ms)^----- BEGIN SUPPORTING-PARTY RECORD -----\n(.*?)\n----- END SUPPORTING-PARTY RECORD -----$',text)
    if row['packet_format']=='pure':
        assert not blocks, 'pure packet carries hybrid block'
        return
    assert row['packet_format']=='hybrid' and len(blocks)==1,'hybrid structural block missing'
    block=blocks[0]
    for label in ['Document class:','Account holder role:','Subject:','Applicant account effect:','Settlement state:','Recorded particulars:','Prepared by:']:
        assert re.search(r'(?m)^'+re.escape(label)+r' .+',block), 'hybrid metadata line missing'
    assert 'Applicant account effect: None; supporting-party record only' in block
    assert re.search(r'(?m)^Record [1-7] / (landlord|broker|guarantor|attorney|prior_paid_vendor)\n.+',block),'typed record structure missing'
    for w in row['attribution_witnesses'].values():assert w['quote'] in block,'hybrid witness not in fragment'

def stats(values):
    s=sorted(values); return {'min':s[0],'p50':s[math.ceil(len(s)*.5)-1],'p95':s[math.ceil(len(s)*.95)-1],'max':s[-1]}

def main():
    import ast
    from transformers import AutoTokenizer
    from grader import grade,aggregate
    tok=AutoTokenizer.from_pretrained('google/gemma-4-E4B-it',local_files_only=True)
    oldfiles=['train_sft.jsonl','val.jsonl','eval_test.jsonl','eval_stress.jsonl','train_sft_v2.jsonl','val_v2.jsonl']
    old=[json.loads(l) for n in oldfiles for l in (ROOT/'data'/n).read_text().splitlines()]
    oldtext='\n'.join(r['source_text'] for r in old).casefold(); system=old[0]['messages'][0]['content']
    pools=json.loads((ROOT/'data/pool_manifest.json').read_text()); forbidden=[]
    for name in ['TRAIN_POOL','EVAL_POOL']:
        for k in ['company_base','dba','contacts','domains','streets','cities']: forbidden+=pools[name][k]
    for filename,var in [('generate_stress_data.py','STRESS_POOL'),('generate_sft_v2_data.py','POOL')]:
        tree=ast.parse((ROOT/filename).read_text()); p=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==var for t in n.targets))
        for k,v in p.items():
            if k!='templates': forbidden+=v
    tree=ast.parse((ROOT/'generate_prospective_data.py').read_text()); p=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROSPECTIVE_POOL' for t in n.targets))
    assert all(v.casefold() not in oldtext for vv in p.values() for v in vv),'prospective pool overlap'
    fragments=set()
    for filename in ['generate_data.py','generate_stress_data.py','generate_sft_v2_data.py']:
        for n in ast.walk(ast.parse((ROOT/filename).read_text())):
            if isinstance(n,ast.Constant) and isinstance(n.value,str): fragments.update(x.casefold() for x in re.split(r'\{[^}]*\}',n.value) if len(x)>=60)
    report={'partitions':{},'prior_split_counts':{n:len((ROOT/'data'/n).read_text().splitlines()) for n in oldfiles},'pool_identity_domain_overlap':0,'prior_template_fragment_overlap':0}; sets={}; confirm=[]
    for part,count in [('confirm',200),('lockbox',150)]:
        path=ROOT/'data'/('eval_prospective_'+part+'.jsonl'); rows=[json.loads(l) for l in path.read_text().splitlines()]; assert len(rows)==count and len({r['id'] for r in rows})==count
        statuses=collections.Counter(); formats=collections.Counter(); slices=collections.Counter(); suffix=collections.Counter(); near=0; prompt=[]; full=[]; completion=[]
        for r in rows:
            field_check(r); format_check(r); source=r['source_text']; assert r['messages']==[{'role':'system','content':system},{'role':'user','content':source}]
            assert not any(v.casefold() in source.casefold() for v in forbidden),'forbidden prior identity'
            assert not any(x in source.casefold() for x in fragments),'prior template fragment'
            assert source.casefold() not in oldtext
            formats[r['packet_format']]+=1
            for f,g in r['ground_truth'].items():
                statuses[g['status']]+=1; slices.update(r['field_tags'][f])
                if g['status']=='MISSING':
                    assert 'attribution_trap' in r['field_tags'][f] and r['attribution_witnesses'][f]['quote'] in source; near+=1
            target=json.dumps(r['ground_truth'],ensure_ascii=False); suffix['period_quote' if target.endswith('.\"]}}') else 'nonperiod_quote' if target.endswith('\"]}}') else 'empty_evidence' if target.endswith('[]}}') else 'other']+=1
            pr=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False); ids=tok(pr)['input_ids']; assert ids==tok(pr,truncation=False)['input_ids']; prompt.append(len(ids)); assert 1100<=len(ids)<=1800
            text=tok.apply_chat_template(r['messages']+[{'role':'assistant','content':target}],tokenize=False,add_generation_prompt=False,enable_thinking=False)
            full.append(len(tok(text,add_special_tokens=False,truncation=False)['input_ids'])); completion.append(len(tok(target,add_special_tokens=False)['input_ids'])); assert full[-1]<=2350 and completion[-1]<=650
        assert statuses['FOUND']/sum(statuses.values())>=.43 and statuses['FOUND']/sum(statuses.values())<=.47
        assert near>=statuses['MISSING']*.8
        if part=='confirm': assert formats=={'pure':100,'hybrid':100} and suffix['period_quote']>=100 and slices['attribution_trap']>=200 and slices['separated_conflict']>=200
        summary=aggregate([grade(json.dumps(r['ground_truth'],ensure_ascii=False),r) for r in rows])
        assert all(summary[k]==1 for k in ['json_valid_rate','json_valid_stripped_rate','found_exact_precision','found_exact_recall','found_exact_f1','missing_abstention_acc','conflict_detection_acc']) and summary['hallucination_rate']==0
        report['partitions'][part]={'records':count,'status_counts':dict(statuses),'format_counts':dict(formats),'field_slice_counts':dict(slices),'missing_near_miss_fields':near,'suffix_census':dict(suffix),'prompt_tokens':stats(prompt),'full_sequence_tokens':stats(full),'GT_completion_tokens':stats(completion),'self_test':summary,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        sets[part]={hashlib.sha256(r['source_text'].encode()).hexdigest() for r in rows}
        if part=='confirm': confirm=rows
    assert not sets['confirm']&sets['lockbox']; report['confirm_lockbox_rowhash_overlap']=0
    preview=json.loads((ROOT/'data/prospective_confirm_preview.json').read_text()); assert len(preview)==4 and all(r in confirm for r in preview)
    assert preview[0]['packet_format']=='pure' and any('attribution_trap' in v for v in preview[0]['field_tags'].values())
    assert preview[1]['packet_format']=='hybrid' and any('attribution_trap' in v for v in preview[1]['field_tags'].values())
    assert any('separated_conflict' in v for v in preview[2]['field_tags'].values()) and json.dumps(preview[2]['ground_truth'],ensure_ascii=False).endswith('.\"]}}')
    assert sum(g['status']=='MISSING' for g in preview[3]['ground_truth'].values())>=2
    report['confirm_preview_coverage_verified']=True; report['lockbox_rows_manually_exposed']=False
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
