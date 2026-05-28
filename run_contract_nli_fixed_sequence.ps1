# Run from project root in the activated venv.
# This sequence fixes the Contract NLI cache, deletes stale predictions, and reruns.

python scripts/repair_contract_nli_cache.py --cache data/contract_nli_confidentiality_cache.json --backup --force
python scripts/purge_contract_nli_outputs.py --yes

# Smoke test first. Check that the prediction JSON input_text now contains CONTRACT CONTEXT and TARGET STATEMENT.
python scripts/run_evaluation_suite.py --suite contract_nli --backend lmstudio --max 5 --save-predictions --skip-baselines --skip-contestability --skip-robustness

# If smoke test looks good, run full Contract NLI.
# python scripts/run_evaluation_suite.py --suite contract_nli --backend lmstudio --save-predictions --resume

# After the full run:
# python scripts/make_strict_reports.py --predictions artifacts\evaluation_suite\contract_nli\full_adss_predictions.json --out artifacts\evaluation_suite\contract_nli
# python scripts/make_paper_workbook.py --results-dir artifacts\evaluation_suite\contract_nli --out artifacts\evaluation_suite\contract_nli\contract_nli_paper_tables.xlsx
# python scripts/audit_contract_nli_predictions.py --workbook artifacts\evaluation_suite\contract_nli\contract_nli_paper_tables.xlsx
