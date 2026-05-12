import logging
import os
import re
import sys
from pathlib import Path

from export import export
from tiled.client import from_profile, from_uri


CATALOG_NAME = "fxi"


def run_export_scan(scan_id_or_uid_or_range, output_dir=None):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Login to tiled with username/password (will prompt interactively).
    logger.info("Connecting to tiled (will prompt for credentials)...")
    tiled_client = from_uri("https://tiled.nsls2.bnl.gov")[CATALOG_NAME]
    tiled_client_fxi = tiled_client["raw"]

    # Override the tiled clients in the export module so export functions use them.
    import export as export_module
    export_module.tiled_client = tiled_client
    export_module.tiled_client_fxi = tiled_client_fxi
    export_module.tiled_client_processed = tiled_client["sandbox"]

    # Parse input: scan_id range (e.g., "12345-12350"), scan_id (int), or uid (string).
    range_match = re.match(r"^(\d+)-(\d+)$", scan_id_or_uid_or_range)
    if range_match:
        start_id, end_id = int(range_match.group(1)), int(range_match.group(2))
        keys = list(range(start_id, end_id + 1))
    else:
        try:
            keys = [int(scan_id_or_uid_or_range)]
        except ValueError:
            keys = [scan_id_or_uid_or_range]

    for key in keys:
        start_doc = tiled_client_fxi[key].start
        uid = start_doc["uid"]
        scan_id = start_doc["scan_id"]
        scan_type = start_doc["plan_name"]

        if output_dir is None:
            filepath = export_module.lookup_directory(start_doc) / "exports"
        else:
            filepath = Path(output_dir).resolve()
        filepath.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting uid={uid} scan_id={scan_id} to {filepath}")
        export_module.export_scan(uid, filepath=filepath)
        logger.info(f"Export complete: uid={uid} scan_id={scan_id}")
        print(f"\nExport complete: {filepath}/{scan_type}_id_{scan_id}.h5")


if len(sys.argv) < 2:
    print("Usage: exporter <scan_id | uid | scan_id_range> [output_dir]")
    sys.exit(1)

# Clear Prefect Cloud env vars before importing prefect so the flow runs locally.
os.environ.pop("PREFECT_API_URL", None)
os.environ.pop("PREFECT_API_KEY", None)

scan_id_or_uid_or_range = sys.argv[1]
output_dir = sys.argv[2] if len(sys.argv) > 2 else None
run_export_scan(scan_id_or_uid_or_range, output_dir=output_dir)
