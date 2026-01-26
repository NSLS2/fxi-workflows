from prefect import flow, get_run_logger, task

# from data_validation import general_data_validation
from export import export


import traceback

from prefect import task, flow, get_run_logger
from prefect.blocks.notifications import SlackWebhook
from prefect.blocks.system import Secret
from prefect.context import FlowRunContext

from tiled.client import from_profile

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
    export(uid)
    log_completion(uid)
