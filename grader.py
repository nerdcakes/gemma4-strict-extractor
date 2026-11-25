"""Strict extraction grader. Python standard library only."""
import argparse
import collections
import json
import re
from pathlib import Path
from generate_data import FIELDS


def normalize(value):
    value=' '.join(value.split()).casefold()
    if re.fullmatch(r'\$[\d,]+(?:\.\d{2})?',value):
        value=value.replace(',','')
        if value.endswith('.00'): value=value[:-3]
    return value


def parse(raw):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out: raise ValueError('duplicate key')
            out[k]=v
        return out
    def reject(x): raise ValueError(x)
    try:
        data=json.loads(raw,object_pairs_hook=pairs,parse_constant=reject)
        if not isinstance(data,dict) or set(data)!=set(FIELDS): return None
        for v in data.values():
            if not isinstance(v,dict) or set(v)!={'status','value','evidence'}: return None
            if v['status'] not in ['FOUND','MISSING','CONFLICT']: return None
            if not isinstance(v['evidence'],list) or not all(isinstance(q,str) for q in v['evidence']): return None
            if v['status']=='FOUND':
                if not isinstance(v['value'],str): return None
            elif v['value'] is not None: return None
        return data
    except (ValueError,TypeError): return None


def grade(raw,row):
    pred=parse(raw)
    stripped=raw.strip()
    if stripped.startswith('```'):
        stripped=re.sub(r'^```(?:json)?\s*','',stripped,flags=re.I)
        stripped=re.sub(r'\s*```$','',stripped)
    errors=collections.Counter(); fields={}; totals=collections.Counter()
    gt=row['ground_truth']; source=row['source_text']
    if pred is None:
        errors['invalid_json_or_schema']+=1
        if raw.strip().startswith('```'): errors['markdown_fence']+=1
    for field in FIELDS:
        truth=gt[field]; p=pred[field] if pred is not None else None
        status=truth['status']; totals['found_gt']+=status=='FOUND'; totals['missing_gt']+=status=='MISSING'; totals['conflict_gt']+=status=='CONFLICT'
        score={'status_correct':bool(p and p['status']==status),'found_tp':False,'missing_correct':False,'conflict_correct':False,'hallucination':False}
        if p:
            evidence=p['evidence']; grounded=all(q and q in source for q in evidence)
            ungrounded=any(q not in source for q in evidence)
            if ungrounded: errors['ungrounded_quote']+=1
            hallucinated_missing=status=='MISSING' and (p['status']!='MISSING' or p['value'] is not None)
            score['hallucination']=ungrounded or hallucinated_missing
            if hallucinated_missing: errors['hallucinated_missing']+=1
            if p['status']=='FOUND':
                totals['found_pred']+=1
                valid_quote=bool(evidence) and grounded and all(p['value'] in q for q in evidence)
                score['found_tp']=status=='FOUND' and normalize(p['value'])==normalize(truth['value']) and valid_quote
                if status=='FOUND' and normalize(p['value'])!=normalize(truth['value']): errors['wrong_value']+=1
                if not valid_quote: errors['invalid_found_evidence']+=1
            score['missing_correct']=status=='MISSING' and p['status']=='MISSING' and p['value'] is None and evidence==[]
            if status=='CONFLICT' and p['status']=='CONFLICT' and p['value'] is None and len(evidence)>=2 and grounded:
                a,b=row['conflict_values'][field]
                score['conflict_correct']=any(i!=j and a in q1 and b in q2 for i,q1 in enumerate(evidence) for j,q2 in enumerate(evidence))
        if status=='CONFLICT' and not score['conflict_correct']: errors['missed_conflict']+=1
        for key in ['found_tp','missing_correct','conflict_correct','hallucination']: totals[key]+=score[key]
        fields[field]=score
    return {'json_valid':pred is not None,'json_valid_stripped':parse(stripped) is not None,'parsed_output':pred,'field_scores':fields,'error_types':dict(errors),'totals':dict(totals)}


def aggregate(results):
    n=len(results); sums=collections.Counter(); errors=collections.Counter()
    for r in results: sums.update(r['totals']); errors.update(r['error_types'])
    precision=sums['found_tp']/sums['found_pred'] if sums['found_pred'] else 0.0
    recall=sums['found_tp']/sums['found_gt'] if sums['found_gt'] else 0.0
    return {'records':n,'json_valid_rate':sum(r['json_valid'] for r in results)/n,'json_valid_stripped_rate':sum(r['json_valid_stripped'] for r in results)/n,
            'found_exact_precision':precision,'found_exact_recall':recall,'found_exact_f1':2*precision*recall/(precision+recall) if precision+recall else 0.0,
            'missing_abstention_acc':sums['missing_correct']/sums['missing_gt'] if sums['missing_gt'] else 1.0,
            'conflict_detection_acc':sums['conflict_correct']/sums['conflict_gt'] if sums['conflict_gt'] else 1.0,
            'hallucination_rate':sums['hallucination']/(n*7), 'hallucination_rate_valid_fields':sums['hallucination']/(sum(r['json_valid'] for r in results)*7) if any(r['json_valid'] for r in results) else None,
            'invalid_json_rows':sum(not r['json_valid'] for r in results),'error_counts':dict(errors),'field_totals':dict(sums),
            'denominator_note':'Primary hallucination_rate divides detected field hallucinations by all N*7 fields. Invalid rows are separately reported and receive no hallucination field classification; valid-fields-only rate also provided. Invalid rows receive no task TP credit.'}


def self_test(directory):
    from generate_data import TRAIN_POOL,EVAL_POOL
    assert all(set(TRAIN_POOL[k]).isdisjoint(EVAL_POOL[k]) for k in TRAIN_POOL)
    summaries={}
    for name,count in [('train_sft',2000),('eval_test',300)]:
        rows=[json.loads(l) for l in (directory/(name+'.jsonl')).read_text().splitlines()]
        assert len(rows)==count
        scores=[grade(json.dumps(r['ground_truth']),r) for r in rows]
        summary=aggregate(scores)
        for metric in ['json_valid_rate','found_exact_f1','missing_abstention_acc','conflict_detection_acc']: assert summary[metric]==1.0,(name,metric,summary)
        assert summary['hallucination_rate']==0.0
        if name=='train_sft':
            for row in rows: assert json.loads(row['prompt'][-1]['content'])==row['ground_truth']
        summaries[name]=summary
    print(json.dumps(summaries,indent=2))
    return summaries


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--self-test',action='store_true'); args=parser.parse_args()
    if args.self_test:
        result=self_test(Path(__import__('os').environ.get('PROJECT_DIR',str(Path(__file__).resolve().parent)))/'data')
        (Path(__file__).parent/'data/self_test_report.json').write_text(json.dumps(result,indent=2))
