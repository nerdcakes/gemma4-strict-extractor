"""Frozen preregistered confirm-only scoring. No lockbox content read."""
import collections
import hashlib
import json
from pathlib import Path
from grader import grade,aggregate,parse
ROOT=Path(__import__("os").environ.get("PROJECT_DIR",str(Path(__file__).resolve().parent))).resolve()

def fraction(correct,total):
    return {'correct':correct,'total':total,'accuracy':correct/total if total else None}

def paired(a,b):
    counts=[0,0,0,0]
    for x,y in zip(a,b,strict=True):counts[0 if x and y else 1 if y else 2 if x else 3]+=1
    return dict(zip(['both_correct','v2_only_correct','baseline_only_correct','both_incorrect'],counts))

def main():
    prereg=ROOT/'prospective_preregistration.json'
    assert hashlib.sha256(prereg.read_bytes()).hexdigest()=='1a5d2532086076b24f7360c356fb7eba0798587ecbae923460d5f99fb3fae2d3'
    spec=json.loads(prereg.read_text());path=ROOT/'data/eval_prospective_confirm.jsonl'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['confirm_sha256']
    rows=[json.loads(l) for l in path.read_text().splitlines()];assert len(rows)==200 and len({r['id'] for r in rows})==200
    fields=list(rows[0]['ground_truth']);members={s:[(i,f) for i,r in enumerate(rows) for f in fields if s in r['field_tags'][f]] for s in spec['primary_slices']}
    for s,pairs in members.items():assert len(pairs)==spec['primary_slices'][s]['denominator'] and len(pairs)>=150
    report={'preregistration_sha256':hashlib.sha256(prereg.read_bytes()).hexdigest(),'confirm_sha256':spec['confirm_sha256'],'models':{},'paired':{},'lockbox_content_read':False};correct={};rowcorrect={};rawpred={}
    for model,name in [('base','base_gemma4_e4b_confirm'),('v1','sft_v1_gemma4_e4b_confirm'),('v2','sft_v2_gemma4_e4b_confirm')]:
        preds=[json.loads(l) for l in (ROOT/'results'/(name+'_preds.jsonl')).read_text().splitlines()];saved=json.loads((ROOT/'results'/(name+'_summary.json')).read_text());assert len(preds)==200 and [p['id'] for p in preds]==[r['id'] for r in rows];assert len({p['id'] for p in preds})==200
        scores=[];correct[model]=[];leaks=[];missingbrace=[]
        for i,(r,p) in enumerate(zip(rows,preds)):
            assert all(p[k]==r[k] for k in ['source_text','ground_truth']);s=grade(p['raw_output'],r);assert all(p[k]==s[k] for k in ['parsed_output','field_scores','error_types']);scores.append(s)
            correct[model].append({f:bool(s['field_scores'][f][{'FOUND':'found_tp','MISSING':'missing_correct','CONFLICT':'conflict_correct'}[r['ground_truth'][f]['status']]]) for f in fields})
            if not s['json_valid'] and parse(p['raw_output']+'}') is not None:missingbrace.append(r['id'])
            for f,w in r['attribution_witnesses'].items():
                q=s['parsed_output'][f] if s['parsed_output'] else None
                if q and q['status']=='FOUND' and q['value']==w['value']:leaks.append({'id':r['id'],'field':f,'value':w['value'],'role':w['role']})
        summary=aggregate(scores);assert all(saved[k]==v for k,v in summary.items());rowcorrect[model]=[all(x.values()) for x in correct[model]];rawpred[model]=preds
        formats={}
        for fmt in ['pure','hybrid']:
            ix=[i for i,r in enumerate(rows) if r['packet_format']==fmt];assert len(ix)==100
            formats[fmt]={'row_accuracy':fraction(sum(rowcorrect[model][i] for i in ix),len(ix)),'field_accuracy':fraction(sum(sum(correct[model][i].values()) for i in ix),7*len(ix))}
        period=[i for i,r in enumerate(rows) if json.dumps(r['ground_truth'],ensure_ascii=False).endswith('.\"]}}')];assert len(period)==125
        report['models'][model]={'summary':saved,'unique_IDs':200,'independent_regrade_exact_match':True,'field_accuracy':fraction(sum(sum(x.values()) for x in correct[model]),1400),'row_accuracy':fraction(sum(rowcorrect[model]),200),'primary_slices':{s:fraction(sum(correct[model][i][f] for i,f in pairs),len(pairs)) for s,pairs in members.items()},'exact_thirdparty_value_leakage':{'count':len(leaks),'denominator':len(members['attribution_trap']),'matches':leaks},'packet_formats':formats,'period_ending':{'rows':125,'json_validity':fraction(sum(scores[i]['json_valid'] for i in period),125),'missing_outer_brace_count':sum(rows[i]['id'] in missingbrace for i in period)},'missing_outer_brace_count':len(missingbrace),'missing_outer_brace_ids':missingbrace,'per_row_correctness':[{'id':r['id'],'field_correct':correct[model][i],'row_correct':rowcorrect[model][i]} for i,r in enumerate(rows)]}
    for model in ['base','v1']:
        report['paired'][model+'_vs_v2']={'fields':paired([correct[model][i][f] for i in range(200) for f in fields],[correct['v2'][i][f] for i in range(200) for f in fields]),'rows':paired(rowcorrect[model],rowcorrect['v2']),'primary_slices':{s:paired([correct[model][i][f] for i,f in pairs],[correct['v2'][i][f] for i,f in pairs]) for s,pairs in members.items()}}
    v=report['models']['v2'];s=v['summary'];tests={'json_lt_98pct':s['json_valid_rate']<.98,'hallucinations_ge21':s['field_totals']['hallucination']>=21,'conflict_lt94pct':s['conflict_detection_acc']<.94,'missing_lt94pct':s['missing_abstention_acc']<.94,'attribution_lt92pct':v['primary_slices']['attribution_trap']['accuracy']<.92,'separated_lt92pct':v['primary_slices']['separated_conflict']['accuracy']<.92};report['decision']={'tests':tests,'verdict':'TRIGGER' if any(tests.values()) else 'SKIP','SMPO_authorized':False,'meaning':'Recommendation only; independent review and explicit owner approval required for any SMPO.'}
    failures=[]
    for i,p in enumerate(rawpred['v2']):
        if not rowcorrect['v2'][i]:failures.append({'id':p['id'],'source_text':p['source_text'],'ground_truth':p['ground_truth'],'raw_output':p['raw_output'],'parsed_output':p['parsed_output'],'error_types':p['error_types'],'incorrect_fields':[f for f in fields if not correct['v2'][i][f]]})
    report['v2_failure_examples']=failures[:3];report['v2_failure_rows']=len(failures)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
