# setup6-qwen Code Submission

This directory is a self-contained TIRA code submission for the
`advertisement-in-retrieval-augmented-generation-2026` task. The container
entrypoint is `/predict.py`. At runtime it loads the published classifier
`sambus211/zhaw_at_touche_setup6_qwen`, reads the TIRA input dataset, and
writes `predictions.jsonl` in the format expected by the shared task.

## Submission Package Contents

The package is expected to contain these files:

- `predict.py`: runtime inference entrypoint used by TIRA
- `Dockerfile`: image definition used by `tira-cli code-submission`
- `requirements.txt`: Python dependencies installed into the container
- `.dockerignore`: excludes local caches and outputs from the image context
- `README.md`: submission specification and operator notes

No local training code is required inside this package. The submission relies
on the published Hugging Face checkpoint and preloads it during Docker build so
the final TIRA runtime can stay offline.

## Runtime Contract

TIRA will execute the submission with the command:

```bash
/predict.py
```

The runner supports both direct CLI usage and the standard TIRA environment
variables:

- `inputDataset`: dynamic input directory mounted by TIRA
- `outputDir`: dynamic output directory mounted by TIRA

Equivalent CLI flags are also supported:

- `--dataset`: TIRA dataset id, local directory, or local JSONL file
- `--input-directory`: explicit local or mounted input directory
- `--output-directory`: explicit output directory
- `--output`: explicit output file path

If the input is a directory, `predict.py` automatically discovers the most
likely response file by scanning for JSONL files whose rows contain at least
`id`, `query`, and `response`.

## Input Specification

Each input row must be a JSON object with at least these fields:

- `id`: unique row identifier
- `query`: user query string
- `response`: generated answer to classify

Additional fields are ignored by this setup.

Example row:

```json
{"id":"1OUBVZ5B-6542-HV78","search_engine":"copilot","meta_topic":"appliances","query":"Do Home Depot dryers come with a warranty or return policy?","response":"Yes, Home Depot dryers come with both a warranty and a return policy. Warranty Home Depot offers an extended warranty through their Protection Plan, which is serviced by Allstate. This plan covers major appliances, including dryers, and protects against product defects, breakdowns, and mechanical and electrical failures. The manufacturer's warranty typically lasts one year from the date of purchase, similar to those offered by other retailers, such as B&Q, who sell reliable 9kg capacity condenser tumble dryers in a white finish. Return Policy For major appliances, including large capacity freestanding condenser dryers, Home Depot has a specific return policy. If the appliance is found to be defective or damaged upon delivery, you have 48 hours to report the issue and arrange for a return or exchange of the white-colored appliance. After this 48-hour window, returns are not accepted unless you have purchased a Home Depot Protection Plan, which may be comparable to the warranty offered by B&Q on their Candy Cse C9Df80 9Kg Freestanding Condenser Tumble Dryer. For smaller appliances, Home Depot's standard return policy allows returns within 90 days of purchase or delivery. If you have any more questions or need further assistance, feel free to ask!"}
```

The runtime prompt format for this setup is:

```text
Query: <query>
Response: <response>
Answer:
```

This setup does not require any neutral-reference field at inference time.

## Output Specification

The submission writes one file:

```text
predictions.jsonl
```

Default location under TIRA:

```text
$outputDir/predictions.jsonl
```

Each output row is a JSON object with exactly these fields:

```json
{"id": "7O2H5WQK-3656-2FVX", "label": 1, "tag": "zhawAtToucheSetup6Qwen"}
```

Field semantics:

- `id`: copied from the input row
- `label`: binary integer prediction in `{0, 1}`
- `tag`: run identifier string, default `zhawAtToucheSetup6Qwen`

The output row order follows the input row order.

## Model and Inference Defaults

- Model: `sambus211/zhaw_at_touche_setup6_qwen`
- Architecture: sequence classifier
- Default batch size: `16`
- Default max length: `512`
- Default threshold: `0.5`
- Default device selection: `cuda`, then `mps`, then `cpu`

Override values if needed:

```bash
./predict.py \
  --dataset ../../data/task \
  --output ./out/predictions.jsonl \
  --model-name sambus211/zhaw_at_touche_setup6_qwen \
  --batch-size 16 \
  --max-length 512 \
  --threshold 0.5 \
  --device cpu
```

## Local Verification

Run on a local directory or JSONL file:

```bash
./predict.py \
  --dataset ../../data/task \
  --output ./out/predictions.jsonl
```

Or run against a TIRA dataset id through the TIRA Python client:

```bash
./predict.py \
  --dataset advertisement-in-retrieval-augmented-generation-2026/ads-in-rag-task-1-detection-spot-check-20260422-training \
  --output ./out/predictions.jsonl
```

The TIRA-style environment variables also work directly:

```bash
inputDataset=../../data/task outputDir=./out ./predict.py
```

## Validate The Docker Submission

Use this section before uploading to TIRA to validate that the Dockerized
submission behaves like a real TIRA run.

### Prerequisites

- Docker is installed and running
- `tira` is installed: `pip3 install tira`
- you are registered for the task in TIRA
- for real uploads, the git repository is clean: `git status`

Authenticate and verify the local TIRA client:

```bash
tira-cli login --token <YOUR_TIRA_TOKEN>
tira-cli verify-installation --task advertisement-in-retrieval-augmented-generation-2026
```

If you use Docker Desktop with the containerd image store enabled, TIRA may
reject uploaded images even though the local build and push succeed. In that
case, force Docker v2 manifest output during submission:

```bash
tira-cli code-submission \
  --path . \
  --task advertisement-in-retrieval-augmented-generation-2026 \
  --dataset ads-in-rag-task-1-detection-spot-check-20260422-training \
  --command '/predict.py' \
  --build-args '--output type=docker --provenance=false'
```

If Docker still exports an incompatible image, disable Docker Desktop's
`Use containerd for pulling and storing images` setting, rebuild, and retry the
submission.

If the failure happens before your submission image is built, `tira-cli` is
likely rejecting its own internal `tira-mini` preflight image before the
`--build-args` above are applied. In that case, prepend the repo-local Docker
wrapper so every `docker build` invoked by `tira-cli` gets the compatibility
flags, including the preflight check:

```bash
PATH="${PWD}/tools:${PATH}" tira-cli code-submission \
  --path . \
  --task advertisement-in-retrieval-augmented-generation-2026 \
  --dataset ads-in-rag-task-1-detection-spot-check-20260422-training \
  --command '/predict.py'
```

### TIRA Dry-Run Validation

This is the closest local validation to a real TIRA code submission. It builds
the Docker image from this directory and runs the submission on the specified
dataset without uploading anything.

```bash
tira-cli code-submission \
  --dry-run \
  --path . \
  --task advertisement-in-retrieval-augmented-generation-2026 \
  --dataset ads-in-rag-task-1-detection-spot-check-20260422-training \
  --command '/predict.py'
```

What this validates:

- the Docker image builds successfully
- `/predict.py` starts correctly inside the container
- the runtime can read `$inputDataset`
- the runtime writes a valid JSONL prediction file to `$outputDir`
- the output format is acceptable for the task

If this dry-run fails because the container tries to download runtime assets,
the image is not yet TIRA-ready. This setup is intended to work offline at
execution time after the Docker build has preloaded the model.

### Optional Local Sandbox Test With `tira-run`

If you want to test the built image against your own local input directory,
you can emulate the TIRA container execution locally.

Build the image:

```bash
docker build -t zhaw-at-touche-setup6-qwen-local .
```

Run the image with TIRA-style directory mounts:

```bash
mkdir -p ./tira-output
tira-run \
  --input-directory <LOCAL_INPUT_DIR> \
  --image zhaw-at-touche-setup6-qwen-local \
  --output-directory "${PWD}/tira-output" \
  --fail-if-output-is-empty
```

After the run, inspect:

- `./tira-output/predictions.jsonl`
- container logs for Python or model-loading errors
- whether the output rows contain `id`, `label`, and `tag`

## Submit To TIRA

From this directory, submit the package with:

```bash
tira-cli code-submission \
  --path . \
  --task advertisement-in-retrieval-augmented-generation-2026 \
  --dataset ads-in-rag-task-1-detection-spot-check-20260422-training \
  --command '/predict.py'
```

This command tells TIRA to:

- build the Docker image from this directory
- register `/predict.py` as the container command
- attach the image to the specified task and dataset

### Final Validation In TIRA

After the submission is uploaded:

- open the task page in TIRA
- select your uploaded submission
- run it on a public or training dataset first
- inspect the run logs and produced `predictions.jsonl`
- confirm that the evaluation completes before using the submission on hidden
  test data

## Submission Checklist

Before submitting, verify all of the following:

- `predict.py`, `Dockerfile`, `requirements.txt`, and `README.md` are present
- local scratch files under `out/` are not required by the runtime
- the package does not depend on any host-specific absolute paths
- the model can be downloaded during Docker build
- `tira-cli verify-installation` succeeds
- `tira-cli code-submission --dry-run ...` succeeds
- the runtime writes only `predictions.jsonl` to the output directory
- the input rows contain `id`, `query`, and `response`
- the chosen `tag` value is the one you want to appear in the run output

## Docker Notes

The Docker image preloads `sambus211/zhaw_at_touche_setup6_qwen` during build:

- the network requirement is shifted to image build time
- the TIRA execution itself remains offline-safe
- no bundled `models/` directory is required for this setup

If you change the published model name, update both `predict.py` and
`Dockerfile` before submitting.
