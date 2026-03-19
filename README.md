# Workflows

Repository of workflows for the FXI beamline.

## Running the Exporter Locally

Run the exporter:

```bash
pixi run exporter <scan_id_or_uid> [output_dir]
```

Examples:

```bash
exporter 12345                            # by scan_id
exporter "02b93a93-43cf-..."              # by uid
exporter 12345 /tmp/exports               # custom output directory
```

Output defaults to the proposal directory (`<proposal>/exports/`).

Notes:
- You will be prompted to log into tiled to access the data.
- Run from a user account (not the beamline account) to write to the default proposal directory.

### System-wide installation

Create `/usr/local/bin/exporter`:

```bash
#!/bin/bash
cd /path/to/fxi-workflows && pixi run exporter "$@"
```

Then `chmod +x /usr/local/bin/exporter`.
