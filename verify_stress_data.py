"""Independent Part1 verifier. No generation, model loading or file mutation."""
import ast
import collections
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__import__("os").environ.get("PROJECT_DIR",str(Path(__file__).resolve().parent))).resolve()
FIELDS = ('legal_entity_name','dba_name','entity_type','primary_contact','amount_due','payment_terms','effective_date')

def check(rows):
    assert len(rows) == 200, ('record_count', len(rows))
    assert len({r['id'] for r in rows}) == 200
    assert len({r['source_text'] for r in rows}) == 200
    original = json.loads((ROOT/'data/eval_test.jsonl').read_text().splitlines()[0])
    old_rows=[]; split_counts={}
    for file in ('train_sft.jsonl','val.jsonl','eval_test.jsonl'):
        split_rows=[json.loads(line) for line in (ROOT/'data'/file).read_text().splitlines()]
        split_counts[file]=len(split_rows)
        for r in split_rows:
            messages=r.get('messages',r.get('prompt',[]))
            source=r.get('source_text')
            if source is None:
                user=[m['content'] for m in messages if m['role']=='user']
                assert len(user)==1, 'cannot recover original user source'
                source=user[0]
            old_rows.append({'id':r.get('id'),'source_text':source})
    old_ids = {r['id'] for r in old_rows if r['id'] is not None}
    old_sources = {r['source_text'] for r in old_rows}
    pool = json.loads((ROOT/'data/pool_manifest.json').read_text())
    counts = collections.Counter()
    near = decoys = conflicts = 0
    # Derived from the records themselves, not generator tags or manifest counts.
    near_patterns = {
        'legal_entity_name': r'The courier receipt names (.+?), but that is the warehouse operator; the applicant has not sent its registered name\.',
        'dba_name': r'The booth sign said (.+?), which was the event theme rather than a trading name; nobody supplied a DBA for this applicant\.',
        'entity_type': r'The landlord is an? (.+?), but the applicant.s own organizational classification was left out of the conversation\.',
        'primary_contact': r'(.+?) offered to forward messages as the outside translator, without being appointed the client.s onboarding contact; that appointment has not arrived\.',
        'amount_due': r'The bank screen shows (.+?) already settled for the equipment deposit, while the current onboarding charge is still being calculated\.',
        'payment_terms': r'The printer.s own bill allowed (.+?), but nobody has stated what payment arrangement applies to this onboarding\.',
        'effective_date': r'(.+?) was the day the draft was emailed, not when services take effect; the commencement date remains undecided\.',
    }
    banned = re.compile(r'(?m)^\s*(?:[-*•]\s|\d+[.)]\s)|\||(?:Applicant legal entity|DBA / trading name|Entity classification|Designated client onboarding contact|Current onboarding balance due|Current payment terms|Effective / service start date)\s*[:=—]')
    for i, row in enumerate(rows):
        assert set(row) == {'id','source_text','messages','ground_truth','conflict_values','tags'}
        assert row['id'] == f'stress-{i:04d}' and row['id'] not in old_ids
        source = row['source_text']
        assert source not in old_sources
        assert row['messages'] == [{'role':'system','content':original['messages'][0]['content']},{'role':'user','content':source}]
        assert not banned.search(source), (row['id'],'form/table/bullet')
        for name in ('TRAIN_POOL','EVAL_POOL'):
            for key in ('company_base','dba','contacts','domains','streets','cities'):
                assert not any(v.casefold() in source.casefold() for v in pool[name][key]), (row['id'],name,key)
        decoy = re.search(r'(.+?) works for (.+?), the separate referral agency, and sent this from (.+?)\.', source)
        assert decoy, (row['id'],'third-party decoy absent')
        decoys += 1
        assert set(row['ground_truth']) == set(FIELDS)
        assert set(row['conflict_values']) == {f for f in FIELDS if row['ground_truth'][f]['status']=='CONFLICT'}
        for f in FIELDS:
            g = row['ground_truth'][f]; counts[g['status']] += 1
            assert set(g) == {'status','value','evidence'}
            if g['status'] == 'MISSING':
                assert g == {'status':'MISSING','value':None,'evidence':[]}
                assert re.search(near_patterns[f],source), (row['id'],f,'missing near-miss witness')
                near += 1
            else:
                assert all(q and source.count(q)==1 for q in g['evidence'])
                if g['status'] == 'FOUND':
                    assert len(g['evidence'])==1 and isinstance(g['value'],str) and g['value'] in g['evidence'][0]
                else:
                    assert g['status']=='CONFLICT' and g['value'] is None and len(g['evidence'])==2
                    a,b = row['conflict_values'][f]
                    assert a != b and a in g['evidence'][0] and b in g['evidence'][1]
                    # Contradictory evidence must be in paragraphs with another full paragraph between.
                    positions = [source[:source.index(q)].count('\n\n') for q in g['evidence']]
                    assert abs(positions[0]-positions[1]) >= 2, (row['id'],f,positions)
                    conflicts += 1
        used = {v for f,g in row['ground_truth'].items() for v in ([g['value']] if g['status']=='FOUND' else row['conflict_values'].get(f,[]))}
        assert decoy.group(1) not in used and decoy.group(2) not in used
        assert not re.search(r'(?i)\b(?:supersedes|overrides|replaces|rescinds|corrected version|takes precedence)\b',source)
    assert counts == {'FOUND':630,'MISSING':350,'CONFLICT':420}, counts
    assert near/counts['MISSING'] >= .8 and decoys/len(rows) >= .8
    from grader import grade, aggregate
    self_test = aggregate([grade(json.dumps(r['ground_truth']),r) for r in rows])
    for key in ('json_valid_rate','json_valid_stripped_rate','found_exact_precision','found_exact_recall','found_exact_f1','missing_abstention_acc','conflict_detection_acc'):
        assert self_test[key] == 1.0, (key,self_test)
    assert self_test['hallucination_rate']==0 and not self_test['error_counts']
    # Cross-check declared pools against actual original pools and all split text, including validation.
    tree = ast.parse((ROOT/'generate_stress_data.py').read_text())
    stress = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='STRESS_POOL' for t in n.targets))
    overlap = {}
    for name in ('TRAIN_POOL','EVAL_POOL'):
        for key in stress:
            overlap[name+'/'+key] = sorted(set(stress[key]) & set(pool[name].get(key,[])))
            assert not overlap[name+'/'+key]
    all_old = '\n'.join(r['source_text'] for r in old_rows).casefold()
    for key in ('company_base','dba','contacts','domains','streets','cities'):
        assert all(v.casefold() not in all_old for v in stress[key]), key
    assert all(t.casefold() not in all_old for t in stress['templates'])
    # Inspect every long prose literal in generator AST, covering CLAIMS, NEAR,
    # opening wrappers, conversational transitions, decoy prose and filler.
    prose_literals=sorted({n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>=60 and '\n' not in n.value})
    assert len(prose_literals)>=50, ('incomplete template coverage',len(prose_literals))
    old_templates=[v for name in ('TRAIN_POOL','EVAL_POOL') for key in ('templates','claim_forms','pending_forms','decoy_forms') for v in pool[name][key]]
    for literal in prose_literals:
        assert literal not in old_templates
        # Long invariant fragments prove that this prose family was not copied
        # verbatim into any original train/validation/eval packet.
        fragments=[x.casefold() for x in re.split(r'\{[^}]*\}',literal) if len(x)>=35]
        assert fragments and not any(x in all_old for x in fragments), ('prose overlap',literal)
    return {'template_prose_literals_checked':len(prose_literals),'template_exact_fragment_overlap':0,'records':len(rows),'fields':sum(counts.values()),'status_counts':dict(counts),'missing_near_misses':near,'missing_near_miss_rate':near/counts['MISSING'],'third_party_decoy_packets':decoys,'third_party_decoy_rate':decoys/len(rows),'separated_conflict_pairs':conflicts,'system_prompt_sha256':hashlib.sha256(original['messages'][0]['content'].encode()).hexdigest(),'old_split_counts':split_counts,'pool_overlap':overlap,'source_overlap':0,'form_table_bullet_matches':0,'self_test':self_test}

def main():
    path = ROOT/'data/eval_stress.jsonl'
    assert path.exists(), 'stress dataset is not yet generated'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = check(rows)
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(__import__('os').environ.get('TOKENIZER_PATH','google/gemma-4-E4B-it'),local_files_only=True)
    tok = proc.tokenizer
    assert tok.model_max_length >= 2048, ('input clipping',tok.model_max_length)
    lengths=[]
    for row in rows:
        prompt = proc.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        assert '<|channel>thought' not in prompt
        default = tok(prompt)['input_ids']; explicit=tok(prompt,truncation=False)['input_ids']
        assert default==explicit, 'implicit input clipping'
        lengths.append(len(default))
    vals=sorted(lengths)
    stats={'min':min(vals),'p50':vals[math.ceil(.5*len(vals))-1],'p95':vals[math.ceil(.95*len(vals))-1],'max':max(vals)}
    assert min(vals)>1004 and min(vals)>=1150 and stats['p50']>=1350 and max(vals)<=1850 and max(vals)<=1950, stats
    result.update({'prompt_tokens':stats,'token_lengths':dict(zip((r['id'] for r in rows),lengths)),'tokenizer_model_max_length':tok.model_max_length,'default_equals_untruncated':True,'enable_thinking':False,'preview_exact':json.loads((ROOT/'data/stress_sample_preview.json').read_text())==rows[:3]})
    assert result['preview_exact']
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
