# Workflows

Repository of workflows for the FXI beamline.

## Running the Exporter Locally

Run the exporter:

```bash
pixi run exporter <scan_id | uid | scan_id_range> [output_dir]
```

Examples:

```bash
exporter 12345                            # by scan_id
exporter "02b93a93-43cf-..."              # by uid
exporter 12345-12350                      # by scan_id range (inclusive)
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
if [ -n "$2" ]; then
    output_path="$(realpath -m "$2")"
    cd /nsls2/data/fxi-new/shared/config/bluesky/fxi-workflows && pixi run exporter "$1" "$output_path"
else
    cd /nsls2/data/fxi-new/shared/config/bluesky/fxi-workflows && pixi run exporter "$1"
fi
```

Then `chmod +x /usr/local/bin/exporter`.
