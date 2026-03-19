# Workflows

Repository of workflows for the FXI beamline.

## Running the Exporter Locally

First, install dependencies:

```bash
pixi install
```

Then from the project directory:

```bash
pixi run exporter <uid> [output_dir]
```

For example:

```bash
pixi run exporter 02b93a93-43cf-45ad-8fda-792c1373dcec /tmp/exports
```

If `output_dir` is omitted, it defaults to `/tmp/exports`.

### Running from anywhere

Use the `-m` flag to specify the manifest path:

```bash
pixi run -m /path/to/fxi-workflows/pixi.toml exporter <uid>
```

Or add an alias to your shell config (`~/.zshrc` or `~/.bashrc`):

```bash
alias exporter='pixi run -m /path/to/fxi-workflows/pixi.toml exporter'
```

Then run from anywhere:

```bash
exporter <uid>
```
