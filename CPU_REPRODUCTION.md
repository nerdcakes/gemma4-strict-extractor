# CPU reproduction and limits

Environment used: existing public.ecr.aws/whitecircle/halo image ID sha256:9f5056ad144465da09faf5bd597288cd4956abe37459688302fb156a078ef662. This is the locally verified immutable image ID, not a claimed registry RepoDigest. Transformers 5.20.2 and torch 2.11.0+cu130 were recorded by experiments; do not infer a fresh dependency install. No installation performed.

The staged scripts use their own directory or PROJECT_DIR environment override. TOKENIZER_PATH optionally selects an offline tokenizer directory. New prospective generator output is controlled by PROSPECTIVE_OUTPUT (defaults to PROJECT_DIR/data); do not run it against frozen archives. Example placeholders in documentation must be replaced by explicit local paths, not copied as literal path names.

CPU smoke command exercised, from project containing release_stage:

    docker run --rm --network=none -e HF_HUB_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 -e PROJECT_DIR=/sandbox -v "$PWD/release_stage/github_stage:/code:ro" -v "$PWD/release_stage/cpu_sandbox:/sandbox" -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" -w /code --entrypoint python sha256:9f5056ad144465da09faf5bd597288cd4956abe37459688302fb156a078ef662 -B cpu_smoke.py

The actual local test used image ID sha256:9f5056ad144465da09faf5bd597288cd4956abe37459688302fb156a078ef662 instead of the registry-digest placeholder above. For exact same-host replay, replace the image argument with that local image ID. There is no GPU flag. This imports all copied scripts, generates four small original-style packets, grades GT, checks PROJECT_DIR overrides and renders/tokenizes through the cached original Gemma template. Outputs go only into the new sandbox. Receipt: ../cpu_smoke_receipt.json. No model loaded. Full successor corpus regeneration, training, inference and GPU compatibility are UNRUN in this stage.

Training config mount recipe (documentation only, NOT authorized to run): mount the approved project layout at /workspace; configs point to /workspace/data and /workspace/checkpoints. Mount Halo checkout separately at /halo, set PYTHONPATH=/halo, and invoke /halo/scripts/training/sft.py /workspace/configs/NAME.yaml from /halo. Cache is a separate mount. Provide the missing generated datasets under /workspace/data. Preserve original assistant template and all frozen decoding; review license first.

Compatibility patch: compatibility.patch is the actual two-file, 28-insertion local diff. Against Halo HEAD 3a21b4dbdb56cb840d42a59e98e40f967a1d2807, first run `git apply --check /path/to/compatibility.patch`, then apply only after owner approval in a separate working checkout. This preparation did not apply or commit anything. Cross-model/high-SRAM GPU tests pending. Never apply twice or describe this as upstream-ready.

PROSPECTIVE RESERVE HOLD: generator and fixed seed can reconstruct the unreleased reserve. Owner must decide whether to withhold generator/seed, retire the reserve, or accept reconstruction before publication. Current reserve rows are not bundled. Do not run prospective generation as a casual reproduction test.
