# Strict extractor SFT v2

**Model adapter:** [Strikz/gemma-4-e4b-strict-extractor-lora](https://huggingface.co/Strikz/gemma-4-e4b-strict-extractor-lora) · [Pinned SFT v2 release](https://huggingface.co/Strikz/gemma-4-e4b-strict-extractor-lora/tree/v1.0-sft-v2)

Both repositories initially have private visibility; access to one does not grant access to the other.
Model/base licensing: Apache-2.0, verified from pinned and main upstream README. Separate Halo code/patch supplemental terms are not relicensed by this file.

This is a specialized seven-field extraction artifact, not an agent brain. Statuses FOUND/MISSING/CONFLICT require verbatim evidence. It is not deployed and has not been evaluated on real customer documents.

## Three-stage evidence
Structured in-distribution SFTv1 looked strong; narrative stress exposed premature model-selected stopping with missing outer braces. A CPU audit and eight bounded traces established selected turn-end stops, not a 768-token ceiling. Token suffix association is not proof of a causal tokenizer/BPE bug. One mixed-batch Base trace differs from the saved result.
SFTv2 starts from the original base, adds fresh training narratives, and changes epochs/LR/length budget jointly. Stress was used for development and is not an untouched final test. Prospective confirmation was sealed before inference and used frozen scoring. The reserve remains unreleased. Its deterministic generator is withheld to prevent reconstruction from the published material.

| Dataset | Model | JSON validity | FOUND F1 | MISSING accuracy | CONFLICT accuracy | Detected hallucinations |
|---|---|---:|---:|---:|---:|---:|
| base_gemma4_e4b | google/gemma-4-E4B-it | 0.963333 | 0.968559 | 0.803810 | 0.938095 | 16 |
| sft_gemma4_e4b | checkpoints/sft_gemma4_e4b_merged | 0.996667 | 0.996966 | 0.996190 | 0.980952 | 2 |
| sft_v2_gemma4_e4b | checkpoints/sft_v2_gemma4_e4b_merged | 0.996667 | 0.997400 | 0.998095 | 0.990476 | 0 |
| base_gemma4_e4b_stress | google/gemma-4-E4B-it | 0.625000 | 0.647005 | 0.605714 | 0.447619 | 54 |
| sft_gemma4_e4b_stress | checkpoints/sft_gemma4_e4b_merged | 0.275000 | 0.351942 | 0.357143 | 0.152381 | 19 |
| sft_v2_gemma4_e4b_stress | checkpoints/sft_v2_gemma4_e4b_merged | 1.000000 | 0.989674 | 0.991429 | 0.992857 | 4 |
| base_gemma4_e4b_confirm | google/gemma-4-E4B-it | 0.820000 | 0.811864 | 0.837143 | 0.659524 | 18 |
| sft_v1_gemma4_e4b_confirm | checkpoints/sft_gemma4_e4b_merged | 0.400000 | 0.484163 | 0.528571 | 0.273810 | 2 |
| sft_v2_gemma4_e4b_confirm | checkpoints/sft_v2_gemma4_e4b_merged | 1.000000 | 0.993620 | 1.000000 | 0.992857 | 0 |

Prospective decision: SKIP. No SMPO run. Nine imperfect v2 confirmation packets remain. The original v2 evaluation has one invalid row; stress has four detected field hallucinations. Invalid rows are separately unclassified for hallucination, not proof of safety. Narrow exact third-party FOUND leakage misses false CONFLICT mixing. Full exact field/packet, format, slice and paired counts are in evidence_manifest.json.

## Inference template — NOT EXECUTED during preparation
Use the composite multimodal class, not a generic causal-only substitute. The exact original system prompt is embedded below; no GitHub import is needed. Use a pinned approved revision, supply plain packet text, and validate output with grader.py. The shown loader is based on the measured evaluator; this staging task did not load a model.
```python
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForMultimodalLM
path = "OWNER_APPROVED_MERGED_REPO"  # replace explicitly
revision = "OWNER_APPROVED_IMMUTABLE_COMMIT"  # replace explicitly
adapter_path = None  # or approved adapter repo/local path
adapter_revision = None  # set immutable commit for a remote adapter
packet_text = "PASTE THE FULL APPLICANT PACKET HERE"
if adapter_path is not None:
    path = "google/gemma-4-E4B-it"
    revision = "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c"
processor = AutoProcessor.from_pretrained(path, revision=revision)
config = AutoConfig.from_pretrained(path, revision=revision)
attention = {"text_config": "sdpa"}
for name, backend in [("vision_config", "sdpa"), ("audio_config", "eager")]:
    if getattr(config, name, None) is not None:
        attention[name] = backend
model = AutoModelForMultimodalLM.from_pretrained(path, revision=revision, config=config,
    torch_dtype=torch.bfloat16, attn_implementation=attention, device_map="cuda")
if adapter_path is not None:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_path, revision=adapter_revision)
model.eval()
processor.tokenizer.padding_side = "left"
SYSTEM_PROMPT = "Extract the current onboarding vendor/client's information from the packet. Return raw JSON only, without markdown or commentary, with exactly these seven keys: legal_entity_name, dba_name, entity_type, primary_contact, amount_due, payment_terms, effective_date. Each key must map to an object with exactly status, value, evidence. FOUND: an unambiguous stated string value and a list containing exactly one exact verbatim substring from the packet that contains that value. MISSING: absent, redacted, TBD, pending, or promised later; value null and evidence []. CONFLICT: two contradictory claims in different sections with no explicit resolution; value null and exactly two verbatim evidence substrings, one per claim. Do not resolve contradictions by assuming the later document wins. Preserve stated spelling, currency formatting and date formatting. primary_contact means the client's designated onboarding contact, not the sender, a broker, counsel or the vendor's employee. amount_due means the current onboarding invoice/retainer/balance due, not a prior paid invoice or illustrative amount. Ignore explicitly unrelated referring companies and old closed invoices. Missing or pending DBA does not imply the legal name is also the DBA."
messages = [{"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":packet_text}]
prompt = processor.apply_chat_template(messages, tokenize=False,
    add_generation_prompt=True, enable_thinking=False)
inputs = processor.tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")
with torch.inference_mode():
    ids = model.generate(**inputs, do_sample=False, max_new_tokens=768,
        pad_token_id=processor.tokenizer.pad_token_id)
text = processor.tokenizer.decode(ids[0, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True, clean_up_tokenization_spaces=False)
```

## Track B compatibility patches
Local fix/gemma4-audio-sdpa has two modified files, 28 inserted lines: checkpoint_coverage.py (21) and sdpa_flex_sliding.py (7). Audio eager and consumer-SRAM training compatibility measured locally only. Cross-model/high-SRAM testing pending. Not upstream-ready and not merged/pushed.

## Release scope
This adapter-first release uses Apache-2.0 and requires google/gemma-4-E4B-it as its base model. Merged weights and optimizer/resume state are not included.

## Reproduction layout
Scripts/configs are sanitized audit copies. PROJECT_DIR placeholders must be resolved by an owner-approved setup; no training or generation is authorized by this README. Bulk datasets and cached model are not bundled. Reports are saved results, not recomputed model inference.

See github_stage/CPU_REPRODUCTION.md (or CPU_REPRODUCTION.md inside GitHub payload) for tested CPU commands and limits. Explorer canonical path: explorer_bundle.json at stage root.

## Withheld reserve generator
The prospective generator is intentionally withheld from the public GitHub payload solely to preserve the 150-packet procedural reserve. Original and private staged copies remain retained. Reserve SHA256: 85715972a2021ae64622ea07278852c205507ed2a7e301879527fb224d0a7643. No reserve rows are included. Do not publish private staging helpers; generator plus seed can reconstruct the reserve.

## License scope
Standard Apache-2.0 applies to the model/base and owner-designated original project material. Halo code or patch material remains subject to the complete Halo License including supplemental terms; no blanket relicensing is intended. The local compatibility patch is retained as experimental engineering evidence; broader compatibility review remains pending.
