# Workflows

Repository of workflows for the FXI beamline.

## Running the Exporter Locally

Install dependencies:

```bash
pixi install
```

Run the exporter:

```bash
pixi run exporter <uid> [output_dir]
```

Output defaults to the proposal directory (`<proposal>/exports/`).

### System-wide installation

Create `/usr/local/bin/exporter`:

```bash
#!/bin/bash
cd /path/to/fxi-workflows && pixi run exporter "$@"
```

Then `chmod +x /usr/local/bin/exporter`.
