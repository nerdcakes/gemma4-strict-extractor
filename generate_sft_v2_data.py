"""Fresh successor training prose, seed20261125. JSON bundle stdout only."""
import json
import random
from pathlib import Path
ROOT=Path(__import__("os").environ.get("PROJECT_DIR",str(Path(__file__).resolve().parent))).resolve()
SEED=20261125
FIELDS=('legal_entity_name','dba_name','entity_type','primary_contact','amount_due','payment_terms','effective_date')
POOL={
'company':['Rowan Workshop','Marigold Transit','Flint Sanctuary','Hazel Observatory','Paprika Atelier','Mallow Junction','Dunlin Station','Sorrel Workshop','Nettle Parlor','Tansy Arcade','Yarrow Assembly','Fennel Pavilion'],
'dba':['Red Umbrella','Sunday Squirrel','Silk Button','Round Biscuit','Peach Hammock','Tiny Violin','Wooden Penguin','Green Hourglass','Orange Hammock','Ivory Tambourine','Purple Parasol','Little Watermill'],
'contact':['Ada Mercer','Noel Fernandes','Rina Abbas','Pavel Mensah','Dina Laurent','Hector Ruiz','Sana Bhat','Evan Morita','Lucia Becker','Imani Frost','Otto Navarro','Vera Singh'],
'domain':['rowan-office.example','marigold-notes.example','flint-desk.example','hazel-routing.example']}
OPEN=[
'The caseworker dictated a call note after a conversation that kept moving between practical arrangements and the application. These passages preserve what each participant said, with no decision about statements that disagree. The recording begins with a discussion of who would collect the duplicate papers from reception.',
'A CRM handoff arrived as a long account rather than a completed questionnaire. The incoming caseworker left the passages in their conversational order so a colleague could distinguish client assertions from the small logistical problems being discussed around them. This account has not been reconciled with the other messages.',
'The afternoon digest combines a compliance conversation with notes from a telephone follow-up. The editor recorded claims in prose instead of making an authoritative register. Whenever two accounts differ, the team still needs a client explanation; this digest does not supply one.',
'For the compliance handoff, a caseworker recounted the conversation in complete sentences and retained the detours. The colleague receiving it asked for an account of what was actually said, rather than an interpretation of the business arrangement. The notes below have not been endorsed as a unified version.',
'The telephone scribe and the relationship assistant contributed separate passages to this handoff. They discussed the application amid interruptions about deliveries and access to the office. Their accounts are presented together without deciding between incompatible statements.'
]
CLAIM={
'legal_entity_name':['Our caller identifies the applicant’s registered legal identity as {v}','In the compliance conversation, the business names {v} as the legal party applying'],
'dba_name':['Customers know the applicant by its expressly adopted trading name, {v}','The caller confirms that the applicant operates using the DBA {v}'],
'entity_type':['Discussing its own constitution, the applicant describes the organization as {v}','The business expressly gives its entity classification as {v}'],
'primary_contact':['For client onboarding correspondence, the business appoints {v} to represent it','The person explicitly assigned as the applicant’s onboarding contact is {v}'],
'amount_due':['The unpaid charge on this applicant’s current onboarding account comes to {v}','The business confirms a presently payable onboarding balance of {v}'],
'payment_terms':['The agreed payment terms covering this applicant’s onboarding are {v}','According to the business, this onboarding payment is governed by {v}'],
'effective_date':['The arrangement for the applicant’s services becomes effective on {v}','The business specifies that service under this onboarding commences on {v}']}
NEAR={
'legal_entity_name':'A receipt mentions {v}, the storage operator, not the applicant; the latter’s registered legal identity has not been disclosed.',
'dba_name':'They used {v} as an exhibition title, not an adopted DBA; the applicant has given no trading name.',
'entity_type':'The description {v} concerns the building owner, not this applicant, whose organizational type is still unknown.',
'primary_contact':'{v} can translate a message but is not appointed to handle client onboarding; the applicant’s appointment is pending.',
'amount_due':'The figure {v} was already paid for freight, not currently due; the applicant’s onboarding balance has not been stated.',
'payment_terms':'The arrangement {v} belongs to the caterer, not this onboarding; the applicant has yet to state its payment terms.',
'effective_date':'They mentioned {v} as only a rehearsal date, not service commencement; an effective date for this account has not been given.'}
ASIDES=[
'The assistant recalled that a visitor had brought the papers in a reusable shopping bag. Reception offered a document wallet, and they spent a few minutes deciding whether the large sheets would fit without folding. None of that conversation concerned the substance of the application, but it explains the pause in the call.',
'Before continuing, the caseworker checked that the hearing loop in the reception area was working. A visitor had trouble hearing announcements over the ventilation system. Staff agreed to speak directly to anyone waiting rather than rely on the loudspeaker until the maintenance team could investigate the problem.',
'The relationship assistant described an awkward exchange about borrowing a pen. Several pens at the counter had dried out, and a visitor went back to the car for another one. The conversation resumed after the visitor returned, without anyone claiming that the interruption had settled a disputed business detail.',
'A colleague remembered that the lift display had stopped showing floor numbers. Reception was directing visitors by the signs beside the stairs instead. The caseworker included this small detail to explain why a caller had been away from the desk when the first attempt to connect the telephone conversation was made.',
'The compliance reviewer asked how the handwritten notes had reached the shared file. The assistant said they were transcribed after the conversation while the wording was still clear. The original notebook stayed with reception, and the transcription preserved uncertainty instead of filling gaps from memory.',
'There was a short discussion of whether refreshments could be brought into the waiting area. A member of staff fetched a tray so cups would not be placed on the papers. The applicant’s business arrangements were not being discussed during this detour; the team was simply trying to keep the documents dry.',
'The caseworker described a conversation about the front door sticking in damp weather. Staff had put a notice beside the handle explaining how to open it gently. The facilities question delayed the telephone handoff, but nobody treated it as an instruction about the business applying for an account.',
'The colleague accepting the handoff asked where to find the paper notebook if a passage needed checking. It was kept in a numbered cabinet rather than sent with the digital copy. They agreed to preserve each speaker’s wording and to ask a question where necessary instead of inventing a missing detail.',
'The visitor had accidentally taken a reusable badge home after an earlier appointment. Reception arranged for it to be returned with the next delivery and issued a temporary replacement. This did not identify a client representative or change who had authority to discuss the application.',
'The caseworker noted that a nearby construction crew had made the call difficult to hear for a moment. The speakers waited for the noise to stop before continuing with the business discussion. No quoted business statement was reconstructed from the inaudible portion of the conversation.'
]
LEADS=['Returning to the caller’s own account, the scribe recorded this passage. ','The CRM author relayed another statement from the business. ','At this point the compliance reviewer read back a sentence supplied by the applicant. ','The relationship assistant preserved a further client statement here. ','Another passage from the business conversation follows in the handoff. ']

def value(field,rng):
    if field=='legal_entity_name': return rng.choice(POOL['company'])+' Enterprises'
    if field=='dba_name': return rng.choice(POOL['dba'])
    if field=='primary_contact': return rng.choice(POOL['contact'])
    if field=='entity_type': return rng.choice(['LLC','S-Corp','C-Corp','General Partnership','Sole Proprietorship','501(c)(3) Nonprofit','Revocable Trust'])
    if field=='payment_terms': return rng.choice(['Net 15','Net 30','Net 45','Net 60','Due on Receipt','50% upfront / 50% on delivery'])
    if field=='amount_due': return f'${rng.randrange(1800,45000):,}.{rng.choice([0,20,40,60,80]):02d}'
    return f'{rng.choice(["January","February","April","June","August","October"])} {rng.randrange(1,29)}, 2030'

def generate(tok):
    rng=random.Random(SEED); bundle={}; original={}
    for split,filename,count in [('train','train_sft.jsonl',800),('validation','val.jsonl',100)]:
        old=(ROOT/'data'/filename).read_text().splitlines(); original[split]=old; system=json.loads(old[0])['prompt'][0]['content']
        schedules={}
        for f in FIELDS:
            s=['FOUND']*(count*45//100)+['MISSING']*(count*25//100)+['CONFLICT']*(count*30//100); rng.shuffle(s); schedules[f]=s
        rows=[]; quoted=0
        for i in range(count):
            first=[]; later=[]; gt={}; conflicts={}; used=set()
            for f in FIELDS:
                state=schedules[f][i]; v=value(f,rng)
                if state=='MISSING': first.append(NEAR[f].format(v=v)); gt[f]={'status':state,'value':None,'evidence':[]}; continue
                # Most date evidence is sentence-final period; retain natural
                # nonperiod conversational quotations in the remaining fifth.
                period=True
                if f=='effective_date': quoted+=1; period=quoted%5!=0
                def claim(x):
                    text=rng.choice(CLAIM[f]).format(v=x)
                    return text+'.' if period else text
                q=claim(v); first.append(q); used.add(v)
                if state=='FOUND': gt[f]={'status':state,'value':v,'evidence':[q]}
                else:
                    other=value(f,rng)
                    while other==v: other=value(f,rng)
                    q2=claim(other); later.append(q2); used.add(other); conflicts[f]=[v,other]; gt[f]={'status':state,'value':None,'evidence':[q,q2]}
            rng.shuffle(first); rng.shuffle(later); asides=rng.sample(ASIDES,len(ASIDES))
            person=rng.choice([x for x in POOL['contact'] if x not in used]); firm=rng.choice(POOL['company'])+' Archive Services'; domain=rng.choice(POOL['domain'])
            decoy=f'{person} at {firm}, the separate archival supplier, wrote from notes@{domain} about its own closed stationery invoice of $71,350.00. This sender is not a client representative and its settled supplier account does not establish the applicant’s balance.'
            paragraphs=[OPEN[i%len(OPEN)]+f' The tracking reference for this particular conversation is {split.upper()}V{i:04d}.']
            for j,q in enumerate(first):
                paragraphs.append(LEADS[(i+j)%len(LEADS)]+q)
                if j==2: paragraphs.append(asides.pop())
            paragraphs.extend([decoy,asides.pop()])
            for j,q in enumerate(later): paragraphs.append(LEADS[(i+j+2)%len(LEADS)]+q)
            def record():
                source='\n\n'.join(paragraphs); messages=[{'role':'system','content':system},{'role':'user','content':source}]
                return {'id':f'v2-{split}-{i:04d}','source_text':source,'messages':messages,'ground_truth':gt,'conflict_values':conflicts,'tags':['fresh_narrative','has_decoy'],'prompt':messages+[{'role':'assistant','content':json.dumps(gt,ensure_ascii=False)}]}
            r=record(); target=rng.randrange(950,1260)
            def tokens(row): return len(tok(tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False,truncation=False)['input_ids'])
            n=tokens(r)
            while n<target and asides:
                paragraphs.insert(-1,asides.pop()); r=record(); n=tokens(r)
            assert 850<=n<=1750,(r['id'],n)
            full=tok.apply_chat_template(r['prompt'],tokenize=False,add_generation_prompt=False,enable_thinking=False)
            assert len(tok(full,add_special_tokens=False,truncation=False)['input_ids'])<=2450
            rows.append(json.dumps(r,ensure_ascii=False))
        mixed=old+rows; rng.shuffle(mixed); bundle[split]=mixed
    preview=[json.loads(l) for l in bundle['train'] if json.loads(l)['id'].startswith('v2-')][:3]
    return {'seed':SEED,'lines':bundle,'preview':preview}

if __name__=='__main__':
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('google/gemma-4-E4B-it',local_files_only=True)
    print(json.dumps(generate(tok),ensure_ascii=False))
