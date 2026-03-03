"""
Customer Churn TFX Pipeline (Local Execution)
"""

import os
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Evaluator,
    Pusher,
)
from tfx.orchestration import pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.proto import trainer_pb2, pusher_pb2

PIPELINE_NAME = "churn_pipeline"
PIPELINE_ROOT = os.path.join(os.getcwd(), "pipeline_output")
DATA_PATH = os.path.join(os.getcwd(), "data")
SERVING_MODEL_DIR = os.path.join(os.getcwd(), "serving_model")

def create_pipeline():

    example_gen = CsvExampleGen(input_base=DATA_PATH)

    statistics_gen = StatisticsGen(
        examples=example_gen.outputs["examples"]
    )

    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"]
    )

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"]
    )

    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=os.path.abspath("transform.py")
    )

    trainer = Trainer(
        module_file=os.path.abspath("model.py"),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(num_steps=100),
        eval_args=trainer_pb2.EvalArgs(num_steps=50)
    )

    pusher = Pusher(
        model=trainer.outputs["model"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=SERVING_MODEL_DIR
            )
        )
    )

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator,
            transform,
            trainer,
            pusher
        ],
    )


if __name__ == "__main__":
    LocalDagRunner().run(create_pipeline())
