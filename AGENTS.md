# Agent Directives: LSTM NNDL Project

## Objective
Implement an end-to-end Multivariate LSTM time-series forecasting project using PyTorch.

## Operational Constraints
- No external authentication, private API keys, or manual downloads required.
- Everything runs via `make` targets.
- Datasets must download automatically if not cached in `data/`.
- Training must support a lightweight `--smoke-test` flag that runs 2 CPU epochs.

## Verification Commands
1. `make install`
2. `make smoke-test`
3. `make eval`
