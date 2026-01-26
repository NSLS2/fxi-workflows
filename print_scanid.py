import databroker
import prefect
from prefect import task, Flow


@task
def print_scanid():
    client = databroker.from_profile("nsls2", username=None)
    scan_id = client["fxi"][-1].start["scan_id"]
    logger = prefect.context.get("logger")
    logger.info(f"Scan ID: {scan_id}")


with Flow("scan_id") as flow:
    print_scanid()

# flow.register(project_name='TST',
#              labels=['fxi-2022-2.2'],
#              add_default_labels=False,
#              set_schedule_active=False)
