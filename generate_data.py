"""Ground-truth-first onboarding packets. Stdlib only; seed 42."""
import collections
import hashlib
import json
import random
from pathlib import Path

FIELDS = ('legal_entity_name','dba_name','entity_type','primary_contact','amount_due','payment_terms','effective_date')
SYSTEM_PROMPT = '''Extract the current onboarding vendor/client's information from the packet. Return raw JSON only, without markdown or commentary, with exactly these seven keys: legal_entity_name, dba_name, entity_type, primary_contact, amount_due, payment_terms, effective_date. Each key must map to an object with exactly status, value, evidence. FOUND: an unambiguous stated string value and a list containing exactly one exact verbatim substring from the packet that contains that value. MISSING: absent, redacted, TBD, pending, or promised later; value null and evidence []. CONFLICT: two contradictory claims in different sections with no explicit resolution; value null and exactly two verbatim evidence substrings, one per claim. Do not resolve contradictions by assuming the later document wins. Preserve stated spelling, currency formatting and date formatting. primary_contact means the client's designated onboarding contact, not the sender, a broker, counsel or the vendor's employee. amount_due means the current onboarding invoice/retainer/balance due, not a prior paid invoice or illustrative amount. Ignore explicitly unrelated referring companies and old closed invoices. Missing or pending DBA does not imply the legal name is also the DBA.'''

# Intentional split-specific vocabularies and complete document phrasing families.
# Shared task labels and categorical values are the contract, not identities/templates.
TRAIN_POOL = {
 'company_base':['Alder Harbor','Cobalt Orchard','Juniper Bridge','Silver Meadow','Cedar Quay','Amber Ridge','Maple Circuit','Willow Foundry','Birch Summit','Coral Loom','Copper Vale','Elm Beacon'],
 'dba':['Bright Kettle','Moss Lantern','Paper Otter','Quiet Finch','Golden Apron','Small Compass','Velvet Fig','Blue Thimble','Marble Kite','Crisp Willow','Satin Sparrow','Lucky Acorn'],
 'contacts':['Arden Wells','Mira Soto','Jonah Kim','Esme Patel','Felix Reed','Lena Ortiz','Tomas Gray','Nadia Blake','Iris Chen','Owen Moss','Clara Pike','Ezra Stone'],
 'domains':['alder-mail.example','cedar-box.example','birch-post.example','coral-send.example'],
 'streets':['17 Alder Lane','82 Cedar Avenue','405 Birch Way','93 Elm Terrace'],
 'cities':['Alderford','Cedarhaven','Birchmere','Elmstead'],
 'templates':[
  '----- Forwarded email -----\nFrom: coordination@{domain}\nTo: intake@{domain}\nSent: Monday, 09:12\nSubject: Client intake\n{claims}\nMailing destination: {street}, {city}.\n{noise}',
  'PASTED INVOICE / RETAINER\nReference {ref}\n| Entry | Stated detail |\n| --- | --- |\n{claims}\nRouting office: {street}; {city}\n{noise}',
  'Slack export — #onboarding-desk\n09:41 coordinator: copying our intake snippets below\n{claims}\n09:44 assistant: address line {street}, {city}\n{noise}',
  'INTERNAL INTAKE NOTES\nCase {ref} / staff transcription\n{claims}\nDispatch locality: {street} — {city}\n{noise}'
 ],
 'claim_forms':['{label}: {value}','{label}  |  {value}','Recorded {label}: {value}','Intake entry / {label} = {value}'],
 'pending_forms':['{label}: still pending; supporting paperwork will follow.','{label}: [REDACTED].','{label}: TBD after review.','{label}: will be supplied Friday; leave this unresolved.'],
 'decoy_forms':['CC outside counsel: {person} (not the client onboarding contact). Referring partner: {company} (separate organization). Previous invoice: {amount}, PAID / CLOSED; unrelated to the current balance.'],
}
EVAL_POOL = {
 'company_base':['Saffron Estuary','Indigo Anvil','Magnolia Atlas','Onyx Prairie','Sequoia Loom','Topaz Delta','Banyan Signal','Garnet Cove','Acacia Crest','Opal Tundra','Laurel Nexus','Ivory Shoal'],
 'dba':['Crimson Teacup','Lunar Badger','Honey Telescope','Teal Fox','Dappled Pear','Woven Meteor','Rustic Poppy','Gentle Heron','Winding Pebble','Scarlet Quill','Dawn Cactus','Mint Comet'],
 'contacts':['Soren Voss','Amara Diaz','Leif Nwosu','Dahlia Park','Kian Bell','Zoya Marin','Rafi Cole','Anya Hale','Bram Liu','Selma Fox','Idris Page','Talia Snow'],
 'domains':['estuary-inbox.example','anvil-letter.example','atlas-routing.example','prairie-delivery.example'],
 'streets':['206 Saffron Road','71 Indigo Court','903 Magnolia Drive','38 Onyx Place'],
 'cities':['Saffronwick','Indigobrook','Magnoliaport','Onyxfield'],
 'templates':[
  'Begin forwarded correspondence\nFrom: liaison@{domain}\nTo: enrollment@{domain}\nSent: Tuesday at 14:26\nRe: account setup documents\n{claims}\nPostal record -> {street} / {city}\n{noise}',
  'ACCOUNTS RECEIVABLE EXTRACT\nDocument {ref}; OCR transcription\nDETAIL                         CONTENT\n{claims}\nDelivery record -> {street} ({city})\n{noise}',
  'SMS paste from enrollment coordinator\n[16:08] Here are the client details copied from the worksheet:\n{claims}\n[16:10] Correspondence goes to {street}, {city}\n{noise}',
  'BACK-OFFICE REGISTRATION MEMO\nWorksheet {ref}; unverified consolidation\n{claims}\nOffice locality on file -> {city}, {street}\n{noise}'
 ],
 'claim_forms':['{label} — {value}','For {label}, the document states «{value}».','Under {label} we have: {value}','{label} ........ {value}'],
 'pending_forms':['{label} — awaiting the client submission.','{label} — withheld from this extract.','{label} — not determined yet.','{label} — client promised to provide this tomorrow.'],
 'decoy_forms':['External adviser copied for reference only: {person}; this is not the designated onboarding contact. Introduction came from {company}, not the applicant. Historical settlement: {amount} [PAID, account closed]; do not carry this into this packet balance.'],
}
LABELS = {
 'legal_entity_name':'Applicant legal entity', 'dba_name':'DBA / trading name',
 'entity_type':'Entity classification','primary_contact':'Designated client onboarding contact',
 'amount_due':'Current onboarding balance due','payment_terms':'Current payment terms',
 'effective_date':'Effective / service start date',
}
TYPES = ['LLC','S-Corp','C-Corp','General Partnership','Sole Proprietorship','501(c)(3) Nonprofit','Revocable Trust']
TERMS = ['Net 30','Net 45','Due on Receipt','50% upfront / 50% on delivery','Net 15','Net 60']


def render_value(field, pool, rng):
    if field == 'legal_entity_name': return rng.choice(pool['company_base']) + ' Holdings ' + rng.choice(['LLC','Inc.','Ltd.'])
    if field == 'dba_name': return rng.choice(pool['dba'])
    if field == 'entity_type': return rng.choice(TYPES)
    if field == 'primary_contact': return rng.choice(pool['contacts'])
    if field == 'amount_due': return f'${rng.randrange(250,49000):,}.{rng.choice([0,25,50,75]):02d}'
    if field == 'payment_terms': return rng.choice(TERMS)
    month = rng.choice(['January','March','May','July','September','November'])
    return f'{month} {rng.randrange(1,29)}, {rng.choice([2027,2028])}'


def generate_split(split, count, pool, rng):
    total = count * len(FIELDS)
    statuses = ['FOUND'] * (total * 55 // 100) + ['MISSING'] * (total * 25 // 100)
    statuses += ['CONFLICT'] * (total - len(statuses))
    rng.shuffle(statuses)
    miss_total = statuses.count('MISSING')
    near_miss_flags = [True] * ((miss_total * 3 + 3)//4)
    near_miss_flags += [False] * (miss_total - len(near_miss_flags))
    rng.shuffle(near_miss_flags)
    decoys = [True] * (count//2) + [False] * (count-count//2)
    rng.shuffle(decoys)
    rows=[]
    for index in range(count):
        nchunks=rng.randrange(2,5)
        chunk_claims=[[] for _ in range(nchunks)]
        # A distinct wrapper type per chunk, in randomized order.
        wrapper_ids=rng.sample(range(4),nchunks)
        truth={}; conflicts={}; hidden={}; missing_subtypes={}; provenance={}; values={}
        for field in FIELDS:
            status=statuses.pop()
            value=render_value(field,pool,rng)
            hidden[field]=value
            if status=='MISSING':
                truth[field]={'status':status,'value':None,'evidence':[]}
                near=near_miss_flags.pop()
                missing_subtypes[field]='near_miss' if near else 'absent'
                if near:
                    chunk_claims[rng.randrange(nchunks)].append(rng.choice(pool['pending_forms']).format(label=LABELS[field]))
                continue
            # Business entity name suffix is generated independently: ensure its
            # spelling does not imply an entity classification that contradicts GT.
            # Legal entity names therefore use neutral business names only below.
            if field=='legal_entity_name': value=value.rsplit(' Holdings ',1)[0] + ' Holdings'
            claim=rng.choice(pool['claim_forms']).format(label=LABELS[field],value=value)
            first=rng.randrange(nchunks); chunk_claims[first].append(claim)
            values[field]=[value]; provenance[field]=[first]
            if status=='FOUND':
                truth[field]={'status':status,'value':value,'evidence':[claim]}
            else:
                other=value
                while other==value:
                    other=render_value(field,pool,rng)
                    if field=='legal_entity_name': other=other.rsplit(' Holdings ',1)[0] + ' Holdings'
                second=rng.choice([n for n in range(nchunks) if n!=first])
                claim2=rng.choice(pool['claim_forms']).format(label=LABELS[field],value=other)
                chunk_claims[second].append(claim2)
                truth[field]={'status':status,'value':None,'evidence':[claim,claim2]}
                conflicts[field]=[value,other]; values[field].append(other); provenance[field].append(second)
        chunks=[]
        for n in range(nchunks):
            rng.shuffle(chunk_claims[n])
            noise='Administrative copy only; no field takes precedence over another section.'
            if decoys[index] and n==0:
                # Disjoint named identities within a packet: don't reuse contact/legal entity as a decoy.
                used_names=set(v for vv in values.values() for v in vv)
                person=rng.choice([x for x in pool['contacts'] if x not in used_names and x!=hidden['primary_contact']])
                company=rng.choice([x for x in pool['company_base'] if x+' Holdings' not in used_names and x not in hidden['legal_entity_name']])+' Referral Partners'
                noise=rng.choice(pool['decoy_forms']).format(person=person,company=company,amount=f'${rng.randrange(50000,99000):,}.00')
            chunks.append(pool['templates'][wrapper_ids[n]].format(domain=rng.choice(pool['domains']),ref=f'{split.upper()}-{index:04d}-{n}',claims='\n'.join(chunk_claims[n]),street=rng.choice(pool['streets']),city=rng.choice(pool['cities']),noise=noise))
        source='\n\n========== attached fragment ==========\n\n'.join(chunks)
        tags=[]
        if conflicts: tags.append('has_conflict')
        if 'near_miss' in missing_subtypes.values(): tags.append('has_near_miss')
        if decoys[index]: tags.append('has_decoy')
        row={'id':f'{split}-{index:04d}','source_text':source,'messages':[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':source}], 'ground_truth':truth,'conflict_values':conflicts,'tags':tags,
             '_audit':{'hidden_candidates':hidden,'missing_subtypes':missing_subtypes,'claim_chunks':provenance,'claim_values':values,'chunk_count':nchunks,'template_ids':wrapper_ids}}
        rows.append(row)
    return rows


def generate():
    rng=random.Random(42)
    return {split:generate_split(split,count,EVAL_POOL if split=='eval' else TRAIN_POOL,rng) for split,count in [('train',2000),('val',200),('eval',300)]}


def validate_corpus(corpus):
    for key in TRAIN_POOL:
        assert set(TRAIN_POOL[key]).isdisjoint(EVAL_POOL[key]),key
    report={'pool_disjoint':True,'source_disjoint':True,'splits':{}}
    sources={}; ids=set()
    for split,rows in corpus.items():
        count=collections.Counter(); near=0; decoy=0; source_set=set()
        for row in rows:
            assert row['id'] not in ids; ids.add(row['id'])
            source=row['source_text']; assert source not in source_set; source_set.add(source)
            assert row['messages']==[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':source}]
            assert set(row['ground_truth'])==set(FIELDS)
            audit=row['_audit']; assert 2<=audit['chunk_count']<=4
            if 'has_decoy' in row['tags']: decoy+=1
            for f,g in row['ground_truth'].items():
                count[g['status']]+=1
                if g['status']=='MISSING':
                    assert g['value'] is None and g['evidence']==[]
                    assert audit['hidden_candidates'][f] not in source,(row['id'],f,'leak')
                    if audit['missing_subtypes'][f]=='near_miss': near+=1
                else:
                    assert all(q and q in source for q in g['evidence'])
                    for v,q in zip(audit['claim_values'][f],g['evidence']): assert v in q
                    if g['status']=='FOUND': assert len(g['evidence'])==1 and g['value'] in g['evidence'][0]
                    else:
                        assert g['value'] is None and len(set(row['conflict_values'][f]))==2
                        assert len(set(audit['claim_chunks'][f]))==2
        sources[split]=source_set
        report['splits'][split]={'records':len(rows),'status_counts':dict(count),'status_rates':{s:count[s]/(len(rows)*7) for s in ['FOUND','MISSING','CONFLICT']},'decoy_rate':decoy/len(rows),'missing_near_miss_rate':near/count['MISSING']}
    for a in sources:
        for b in sources:
            if a!=b: assert sources[a].isdisjoint(sources[b]),(a,b)
    return report


def export(corpus,directory):
    directory.mkdir(parents=True,exist_ok=True)
    for split,filename in [('train','train_sft.jsonl'),('val','val.jsonl'),('eval','eval_test.jsonl')]:
        with (directory/filename).open('x') as stream:
            for row in corpus[split]:
                exported=dict(row)
                if split!='eval':
                    exported['prompt']=row['messages']+[{'role':'assistant','content':json.dumps(row['ground_truth'],ensure_ascii=False)}]
                stream.write(json.dumps(exported,ensure_ascii=False)+'\n')
    (directory/'sample_preview.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='_audit'} for r in corpus['eval'][:3]],ensure_ascii=False,indent=2))
    (directory/'pool_manifest.json').write_text(json.dumps({'seed':42,'TRAIN_POOL':TRAIN_POOL,'EVAL_POOL':EVAL_POOL},indent=2))
    report=validate_corpus(corpus)
    report['sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.jsonl')}
    (directory/'generation_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    export(generate(),Path(__import__('os').environ.get('PROJECT_DIR',str(Path(__file__).resolve().parent)))/'data')
