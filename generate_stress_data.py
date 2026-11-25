"""Seeded narrative stress packets; tokenizer-only, no model load.
Run inside existing image with project mounted read-only. JSON bundle to stdout.
Host exports the two new data files exclusively from that bundle.
"""
import json
import random
from pathlib import Path

ROOT=Path(__import__("os").environ.get("PROJECT_DIR",str(Path(__file__).resolve().parent))).resolve()
SEED=20261124
STRESS_POOL={
 'company_base':['Basalt Meridian','Kestrel Fjord','Tamarind Relay','Obsidian Current','Wisteria Spindle','Peregrine Inlet','Lichen Aperture','Quartz Causeway','Zephyr Alcove','Persimmon Vector','Cypress Gyroscope','Ochre Passage'],
 'dba':['Tin Firefly','Violet Sextant','Porcelain Lynx','Brocade Turnip','Indigo? Never','Hushed Marmot','Tangerine Astrolabe','Woolen Walrus','Dented Sundial','Plum Accordion','Glass Cicada','Crooked Tulip'],
 'contacts':['Petra Ivers','Dorian Okafor','Nell Banerjee','Casper Duong','Alina Serrat','Yusuf Tanaka','Greta Salcedo','Malik Rios','Ines Kovalenko','Hugo Das','Farah Mbeki','Cleo Anders'],
 'domains':['meridian-dispatch.example','kestrel-pouch.example','tamarind-relay.example','wisteria-courier.example'],
 'streets':['614 Basalt Crescent','29 Kestrel Mews','780 Tamarind Walk','152 Wisteria Rise'],
 'cities':['Basaltreach','Kestrelhaven','Tamarindcross','Wisteriagate'],
 'templates':[
  'I pasted the conversation out of our shared channel because the attachment viewer kept dropping replies. Please keep the voices separate while reading; we have not agreed on a consolidated account of what the client said.',
  'This is the intake volunteer retelling a phone call, followed by the contract clerk copying a passage from a different message. Neither person has signed off on the other account, and the original attachments are still stuck in the upload queue.',
  'The conversation wandered between a Slack thread and a draft service agreement. I am preserving both as received rather than smoothing them into an application form. We have not had a reconciliation call with the client.',
  'What follows is a reconstruction from several messages in the case channel. Some came through the phone bridge and some through the contract inbox; the clerk kept them together so the unresolved statements would not disappear.'
 ]}
FIELDS=('legal_entity_name','dba_name','entity_type','primary_contact','amount_due','payment_terms','effective_date')
CLAIMS={
 'legal_entity_name':[
  'When asked which organization is actually applying, the client wrote that its registered legal name is {v}.',
  'The applicant identified itself in the agreement text as {v}, and said that is its legal name rather than a shop sign.',
  'In the client message about the contracting party, it gives its own legal identity as {v}.'
 ],
 'dba_name':[
  'The client says it trades under the DBA {v}, even though that is not the registered name.',
  'In describing what customers see above the door, the applicant explicitly names {v} as its trading name.',
  'The applicant explains in the contract conversation that {v} is the DBA it uses for this business.'
 ],
 'entity_type':[
  'The client describes its own organizational classification as {v} in the same conversation.',
  'For the applicant itself, the agreement writer records the entity classification as {v}.',
  'The applicant says its legal organizational type is {v}, not a description of the agency helping it.'
 ],
 'primary_contact':[
  'The applicant has designated {v} as its contact for onboarding questions, rather than merely copying that person.',
  'When the clerk asked who speaks for the client during onboarding, the client appointed {v}.',
  'The client directs all onboarding questions to {v}, whom it names as its designated onboarding contact.'
 ],
 'amount_due':[
  'For the current onboarding invoice, the client says the outstanding amount due is {v}.',
  'The agreement passage puts the amount now due for this onboarding at {v}, with no payment against that charge.',
  'The applicant says it currently owes {v} for this onboarding, separate from any equipment or historical payments.'
 ],
 'payment_terms':[
  'The applicant states that the current onboarding payment arrangement is {v}.',
  'For this onboarding charge specifically, the client writes that the payment terms are {v}.',
  'In discussing how this onboarding is to be paid, the applicant gives the terms as {v}.'
 ],
 'effective_date':[
  'The applicant says services under this onboarding take effect on {v}.',
  'The agreement passage sets the effective date for these services at {v}.',
  'The client states that the service commencement date for this onboarding is {v}.'
 ]}
NEAR={
 'legal_entity_name':'The courier receipt names {v}, but that is the warehouse operator; the applicant has not sent its registered name.',
 'dba_name':'The booth sign said {v}, which was the event theme rather than a trading name; nobody supplied a DBA for this applicant.',
 'entity_type':'The landlord is a {v}, but the applicant’s own organizational classification was left out of the conversation.',
 'primary_contact':'{v} offered to forward messages as the outside translator, without being appointed the client’s onboarding contact; that appointment has not arrived.',
 'amount_due':'The bank screen shows {v} already settled for the equipment deposit, while the current onboarding charge is still being calculated.',
 'payment_terms':'The printer’s own bill allowed {v}, but nobody has stated what payment arrangement applies to this onboarding.',
 'effective_date':'{v} was the day the draft was emailed, not when services take effect; the commencement date remains undecided.'}
FILLERS=[
 'Someone on the channel asked whether the envelope needed a wet signature on the outside. The desk clerk said the envelope was only there to keep loose pages from being separated in transit, and asked people not to staple the scanning cover to the agreement. That exchange took most of the call because the microphone kept cutting out.',
 'The room booking was moved to the smaller meeting space when the projector failed. Nobody was discussing a service commencement date during that exchange; they were trying to find a place with a working speakerphone. The volunteer kept the exchange in the transcript because it explains why two different people were taking notes.',
 'A phone photo arrived sideways and the clerk rotated it before attaching it to the conversation. The edges of the paper are visible, but nothing from the text has been rewritten here. Several people asked for a cleaner scan, and the person at the desk said the scanner was unavailable until the technician finished with it.',
 'One reply consisted entirely of a complaint about the upload window expiring. Another person suggested sending smaller attachments, but the clerk asked them to keep the original files together so the conversation would remain intelligible. This discussion was about transmission rather than approval of any business statement in the packet.',
 'The contract clerk paused to explain why an empty signature box appeared in the thumbnail. The preview had cropped the margin and was showing part of an unused page. Nobody treated that thumbnail as a signed authorization, and the full attachment remained in the queue while people tried a different browser.',
 'A participant joined halfway through and asked for the earlier messages to be pasted again. The volunteer declined to paraphrase them into a neat summary because doing that had caused confusion on another case. Instead, the clerk copied the conversational passages with their original wording and left the surrounding discussion visible.',
 'The reception team found a loose delivery slip under the keyboard and asked if it belonged with these papers. The volunteer said to keep it separate until the courier could identify the envelope. The slip did not establish anything about the applicant’s requested services, although the interruption explains the gap in the conversation.',
 'There was a long side conversation about the access ramp behind the building. The person collecting the papers needed to know which entrance was open, not which organization was signing the agreement. The clerk kept that distinction clear because the building is shared by unrelated tenants using the same mail counter.',
 'The person transcribing the call asked everyone to stop speaking over one another. A sentence had been lost when somebody disconnected, and the clerk did not want to guess what it had said. The passages reproduced here were audible or visible in full; the missing audio was only a discussion of where to put the spare chairs.',
 'An automatic notice warned that the shared folder would become read only during maintenance. The clerk saved a local copy of the correspondence so the voices could still be reviewed together. That maintenance notice did not approve, amend, or settle any of the client’s statements about its onboarding arrangements.',
 'Someone wondered why the paper clip in the photograph looked rusty. The volunteer said the bundle had been sitting near an open window while the courier waited, and that a fresh sleeve would be used for transport. The exchange was mundane, but it interrupted the conversation before the next message was copied.',
 'A short exchange about translation followed when a browser offered to translate the entire page. The clerk asked people to leave the quoted client wording alone and translate only their own administrative comments if needed. The passages in this transcript retain the spelling and punctuation of the messages actually received.',
 'The intake desk asked whether the people joining remotely could hear the rustling paper. One caller could hear it and another could not, so they changed the microphone position and resumed. This was not a discussion of the applicant’s representatives; the speakers were simply trying to make the recording usable.',
 'A courier asked if the parcel could be left beside the loading door. The clerk said to use the staffed counter instead because loose documents sometimes get mixed with deliveries for other tenants. No inference about the contracting party should be drawn from who happened to collect the envelope at that counter.',
 'The agreement clerk noticed a faint coffee ring along the edge of a scan. It did not cover the quoted passage, but the clerk requested another image for the archive anyway. While waiting, the channel went back to discussing the order in which the attachments had reached the shared folder.',
 'The volunteer had handwritten a reminder to charge the conference speaker. Somebody mistook the reminder for an instruction about the account, and the volunteer clarified that it concerned the battery in the meeting room. They kept talking for a while before returning to the actual correspondence from the applicant.'
]

def value(field,rng):
    if field=='legal_entity_name': return rng.choice(STRESS_POOL['company_base'])+' Collective'
    if field=='dba_name': return rng.choice(STRESS_POOL['dba'])
    if field=='primary_contact': return rng.choice(STRESS_POOL['contacts'])
    if field=='entity_type': return rng.choice(['LLC','S-Corp','C-Corp','General Partnership','Sole Proprietorship','501(c)(3) Nonprofit','Revocable Trust'])
    if field=='amount_due': return f'${rng.randrange(1200,47000):,}.{rng.choice([10,35,65,90]):02d}'
    if field=='payment_terms': return rng.choice(['Net 30','Net 45','Due on Receipt','50% upfront / 50% on delivery','Net 15','Net 60'])
    return f'{rng.choice(["February","April","June","August","October","December"])} {rng.randrange(1,29)}, 2029'

def generate(processor):
    rng=random.Random(SEED)
    original=json.loads((ROOT/'data/eval_test.jsonl').read_text().splitlines()[0])
    system=original['messages'][0]['content']
    statuses=['FOUND']*630+['MISSING']*350+['CONFLICT']*420
    rng.shuffle(statuses)
    rows=[]; lengths=[]
    for index in range(200):
        first=[]; second=[]; truth={}; conflicts={}; used=set()
        for field in FIELDS:
            status=statuses.pop(); v=value(field,rng)
            if status=='MISSING':
                first.append(NEAR[field].format(v=v)); truth[field]={'status':status,'value':None,'evidence':[]}; continue
            q=rng.choice(CLAIMS[field]).format(v=v); first.append(q); used.add(v)
            if status=='FOUND': truth[field]={'status':status,'value':v,'evidence':[q]}
            else:
                other=value(field,rng)
                while other==v: other=value(field,rng)
                q2=rng.choice(CLAIMS[field]).format(v=other); second.append(q2); used.add(other)
                truth[field]={'status':status,'value':None,'evidence':[q,q2]}; conflicts[field]=[v,other]
        rng.shuffle(first); rng.shuffle(second)
        person=rng.choice([p for p in STRESS_POOL['contacts'] if p not in used])
        company=rng.choice(STRESS_POOL['company_base'])+' Referral Studio'
        domain=rng.choice(STRESS_POOL['domains']); street=rng.choice(STRESS_POOL['streets']); city=rng.choice(STRESS_POOL['cities'])
        opening=STRESS_POOL['templates'][index%4]+f' The desk filed this conversation under case R{index:04d}; the paper copy was collected at {street} in {city}.'
        decoy=f'{person} works for {company}, the separate referral agency, and sent this from dispatch@{domain}. The agency is helping route the paperwork, not applying for this account, and its staff member is not the client’s designated contact. Its own settled printing charge of $83,219.40 belongs to a closed agency job rather than the current onboarding invoice.'
        # The second account has no later-wins, correction, or resolution language.
        voices=rng.sample([
            'A volunteer interrupts the Slack conversation to paste a sentence from the client. ',
            'The clerk returns from the phone bridge with a passage they could hear clearly. ',
            'In a separate reply, the client stops discussing the upload problem long enough to explain this. ',
            'Someone asks for the actual wording instead of a summary, and the intake clerk copies this passage. ',
            'A message from the applicant arrives while the desk is sorting the attachments. ',
            'The contract clerk reads a sentence aloud from the client’s correspondence. ',
            'The conversation drifts back to the application when the volunteer finds this line in the thread. '
        ],7)
        fillers=rng.sample(FILLERS,len(FILLERS))
        paragraphs=[opening]
        # Each client statement belongs to a conversational turn, with unrelated
        # interruptions interleaved instead of a seven-field enumerative block.
        for j,claim in enumerate(first):
            paragraphs.append(voices[j]+claim)
            if j in (0,2,4): paragraphs.append(fillers.pop())
        paragraphs.extend([decoy,fillers.pop()])
        # Conflicting statements live in separate later turns, never a correction.
        for j,claim in enumerate(second):
            paragraphs.append(voices[(j+3)%7]+claim)
            if j%2==0 and fillers: paragraphs.append(fillers.pop())
        if not second:
            paragraphs.append('The contract clerk had no additional client statements to copy into this conversation.')
        target=rng.randrange(1390,1751)
        def packet():
            source='\n\n'.join(paragraphs)
            return {'id':f'stress-{index:04d}','source_text':source,'messages':[{'role':'system','content':system},{'role':'user','content':source}],'ground_truth':truth,'conflict_values':conflicts,'tags':(['has_conflict'] if conflicts else [])+['has_near_miss' if any(g['status']=='MISSING' for g in truth.values()) else 'no_missing','has_decoy','narrative_stress']}
        row=packet()
        def token_count(r):
            rendered=processor.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            return len(processor.tokenizer(rendered,truncation=False)['input_ids'])
        n=token_count(row)
        while n<target and fillers:
            # Finite, distinct natural interruptions; an internal random target
            # never overrides the external acceptance interval.
            paragraphs.insert(-1,fillers.pop()); row=packet(); n=token_count(row)
        assert 1150<=n<=1850, (index,n)
        rows.append(row); lengths.append(n)
    return {'seed':SEED,'records':rows,'token_lengths':lengths,'STRESS_POOL':STRESS_POOL,'tokenizer_only':True}

if __name__=='__main__':
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(__import__('os').environ.get('TOKENIZER_PATH','google/gemma-4-E4B-it'),local_files_only=True)
    assert processor.tokenizer.model_max_length>=2048, 'HARD STOP: input clipping below 2048'
    print(json.dumps(generate(processor),ensure_ascii=False))
