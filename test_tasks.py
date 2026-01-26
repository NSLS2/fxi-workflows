import prefect
from prefect import task, Flow


@task
def print_stuff():
    logger = prefect.context.get("logger")
    logger.info("printing stuff")


with Flow("print_stuff") as flow1:
    print_stuff()

# flow1.register(project_name="TST", labels=["tst-2022-2.2"], add_default_labels=False, set_schedule_active=False)
