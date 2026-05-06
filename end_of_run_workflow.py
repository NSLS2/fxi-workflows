# Clear Prefect Cloud env vars before importing prefect so the flow runs locally.
if __name__ == "__main__":
    import os
    os.environ.pop("PREFECT_API_URL", None)
    os.environ.pop("PREFECT_API_KEY", None)

from prefect import flow, get_run_logger, task

# from data_validation import general_data_validation
from export import export


import traceback

from prefect.blocks.notifications import SlackWebhook
from prefect.blocks.system import Secret
from prefect.context import FlowRunContext

from tiled.client import from_profile, from_uri
from data_validation import get_run

CATALOG_NAME = "fxi"


def slack(func):
    """
    Send a message to mon-prefect slack channel about the flow-run status.
    Send a message to mon-bluesky slack channel if the bluesky-run failed.

    NOTE: the name of this inner function is the same as the real end_of_workflow() function because
    when the decorator is used, Prefect sees the name of this inner function as the name of
    the flow. To keep the naming of workflows consistent, the name of this inner function had to match the expected name.
    """

    def end_of_run_workflow(stop_doc):
        flow_run_name = FlowRunContext.get().flow_run.dict().get("name")

        # Load slack credentials that are saved in Prefect.
        mon_prefect = SlackWebhook.load("mon-prefect")
        mon_bluesky = SlackWebhook.load("mon-bluesky")

        # Get the uid.
        uid = stop_doc["run_start"]

        # Get the scan_id.
        scan_id = get_run(uid).start["scan_id"]

        # Send a message to mon-bluesky if bluesky-run failed.
        if stop_doc.get("exit_status") == "fail":
            mon_bluesky.notify(
                f":bangbang: {CATALOG_NAME} bluesky-run failed. (*{flow_run_name}*)\n ```run_start: {uid}\nscan_id: {scan_id}``` ```reason: {stop_doc.get('reason', 'none')}```"
            )

        try:
            result = func(stop_doc)

            # Send a message to mon-prefect if flow-run is successful.
            mon_prefect.notify(
                f":white_check_mark: {CATALOG_NAME} flow-run successful. (*{flow_run_name}*)\n ```run_start: {uid}\nscan_id: {scan_id}```"
            )
            return result
        except Exception as e:
            tb = traceback.format_exception_only(e)

            # Send a message to mon-prefect if flow-run failed.
            mon_prefect.notify(
                f":bangbang: {CATALOG_NAME} flow-run failed. (*{flow_run_name}*)\n ```run_start: {uid}\nscan_id: {scan_id}``` ```{tb[-1]}```"
            )
            raise

    return end_of_run_workflow


@task
def log_completion(uid):
    logger = get_run_logger()
    logger.info(f"Export of bluesky run complete. {uid=}")


@flow
@slack
def end_of_run_workflow(stop_doc):
    uid = stop_doc["run_start"]
    # general_data_validation(uid)
    export(uid)
    log_completion(uid)


def end_of_run_workflow_local(scan_id_or_uid_or_range, output_dir=None):
    import logging
    import re
    from pathlib import Path

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


def main():
    import os
    import sys

    # Clear Prefect Cloud env vars so the flow runs locally.
    os.environ.pop("PREFECT_API_URL", None)
    os.environ.pop("PREFECT_API_KEY", None)

    if len(sys.argv) < 2:
        print("Usage: exporter <scan_id | uid | scan_id_range> [output_dir]")
        sys.exit(1)

    scan_id_or_uid_or_range = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    end_of_run_workflow_local(scan_id_or_uid_or_range, output_dir=output_dir)


if __name__ == "__main__":
    main()
