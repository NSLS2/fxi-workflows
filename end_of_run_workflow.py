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
        api_key = Secret.load("tiled-fxi-api-key", _sync=True).get()
        tiled_client = from_profile("nsls2", api_key=api_key)[CATALOG_NAME]
        tiled_client_raw = tiled_client["raw"]
        scan_id = tiled_client_raw[uid].start["scan_id"]

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
    logger.info(f"Complete: {uid}")


@flow
@slack
def end_of_run_workflow(stop_doc):
    uid = stop_doc["run_start"]
    # general_data_validation(uid)
    #export(uid)
    log_completion(uid)


def end_of_run_workflow_local(uid, output_dir="/tmp/exports"):
    import logging
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

    # Look up scan_id and build local output path.
    start_doc = tiled_client_fxi[uid].start
    scan_id = start_doc["scan_id"]
    filepath = Path(output_dir) / str(scan_id)
    filepath.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting uid={uid} scan_id={scan_id} to {filepath}")
    export_module.export_scan(uid, filepath=filepath)
    logger.info(f"Export complete: uid={uid} scan_id={scan_id}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python end_of_run_workflow.py <uid> [output_dir]")
        sys.exit(1)

    uid = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/exports"
    end_of_run_workflow_local(uid, output_dir=output_dir)
