# Model Governance

## Lifecycle
- New models are registered as `shadow`.
- Shadow models are compared against governance thresholds.
- Promotion is explicit via command; no auto-promotion.
- Previous production model is archived automatically on promotion.
- Rollback is explicit via command and uses existing artifacts.

## Metadata Tracked
- `model_id`
- `version`
- `trained_at`
- `training_window`
- `featureset_version`
- `metrics`
- `status` (`shadow | production | archived`)

## Commands
- Train/register shadow: `python -m training.train train`
- Promote: `python -m training.train promote --model-id <id> --version <version>`
- Rollback: `python -m training.train rollback --model-id <id> --version <version>`
- List registry: `python -m training.train list`

## Continuous Learning Definition
Continuous learning is controlled retraining with drift checks, shadow evaluation, explicit promotion decisions, and reversible rollback.
