# Configuration

YAML files record the reported neighbourhood radius, prompt flags, model type,
model identifier, and replicate naming. `src.paths` reads the bundled STARmap
example and downloaded Figshare directories from `examples/` by default;
`LLM_ST_*` variables are optional external-data overrides.

Canonical paper configs:

- STARmap: `config_zeroshot_starmap.yaml`, `config_finetunePro_starmap.yaml`
- Visium/LIBD: `config_zeroshot_libd.yaml`, `config_finetunePro_libd.yaml`
- MERFISH: `config_zeroshot_merfish.yaml`, `config_finetunePro_merfish.yaml`
- COPD Slide2: `config_zeroshot_copd.yaml`, `config_finetunePro_copd.yaml`

Fine-tuned provider model identifiers may require account access.  They are
provenance identifiers, not credentials.  API keys must be supplied through
environment variables.
