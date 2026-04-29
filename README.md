# setup6-qwen Code Submission

This directory contains a TIRA-style code submission for the `setup6-qwen`
classifier.

The runner loads the RoBERTa classifier from
`sambus211/zhaw_at_touche_setup6_qwen`, reads either a TIRA dataset id via
`--dataset` or a dynamic input directory via `--input-directory` / `$inputDataset`,
and writes `predictions.jsonl` to `--output`, `--output-directory`, or
`$outputDir`.

Each output row has the required format:

```json
{"id": "7O2H5WQK-3656-2FVX", "label": 1, "tag": "zhawAtToucheSetup6Qwen"}
```

## Expected Input

The input rows are expected to contain at least `id`, `query`, and `response`.
Additional fields are ignored by the classifier. Example:

```json
{"id":"1OUBVZ5B-6542-HV78","search_engine":"copilot","meta_topic":"appliances","query":"Do Home Depot dryers come with a warranty or return policy?","response":"Yes, Home Depot dryers come with both a warranty and a return policy. Warranty Home Depot offers an extended warranty through their Protection Plan, which is serviced by Allstate. This plan covers major appliances, including dryers, and protects against product defects, breakdowns, and mechanical and electrical failures. The manufacturer's warranty typically lasts one year from the date of purchase, similar to those offered by other retailers, such as B&Q, who sell reliable 9kg capacity condenser tumble dryers in a white finish. Return Policy For major appliances, including large capacity freestanding condenser dryers, Home Depot has a specific return policy. If the appliance is found to be defective or damaged upon delivery, you have 48 hours to report the issue and arrange for a return or exchange of the white-colored appliance. After this 48-hour window, returns are not accepted unless you have purchased a Home Depot Protection Plan, which may be comparable to the warranty offered by B&Q on their Candy Cse C9Df80 9Kg Freestanding Condenser Tumble Dryer. For smaller appliances, Home Depot's standard return policy allows returns within 90 days of purchase or delivery. If you have any more questions or need further assistance, feel free to ask!"}
```

## Local Run

Run on a local input directory or JSONL file:

```bash
./predict.py \
  --dataset ../../data/task \
  --output ./out/predictions.jsonl
```

Or run directly against a TIRA dataset id via the TIRA Python client:

```bash
./predict.py \
  --dataset advertisement-in-retrieval-augmented-generation-2026/ads-in-rag-task-1-detection-spot-check-20260422-training \
  --output ./out/predictions.jsonl
```

The code submission entrypoint still supports the standard TIRA runtime
environment variables directly:

```bash
inputDataset=../../data/task outputDir=./out ./predict.py
```

## TIRA Submission

Submit this directory as a code submission:

```bash
tira-cli code-submission \
  --path . \
  --task advertisement-in-retrieval-augmented-generation-2026 \
  --dataset ads-in-rag-task-1-detection-spot-check-20260422-training \
  --command '/predict.py'
```

## Notes

- Runtime input format is `query + response`.
- This submission does not need a neutral response at inference time.
- The Docker image preloads `sambus211/zhaw_at_touche_setup6_qwen` during build
  so execution remains offline-safe inside TIRA.
- The default tag is `zhawAtToucheSetup6Qwen`. Override it with `--tag` if you
  want a different run identifier.
