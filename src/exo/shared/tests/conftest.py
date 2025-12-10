"""Pytest configuration and shared fixtures for shared package tests."""

import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger

from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata


def get_pipeline_shard_metadata(
    model_id: ModelId, device_rank: int, world_size: int = 1
) -> ShardMetadata:
    return PipelineShardMetadata(
        model_meta=ModelMetadata(
            model_id=model_id,
            pretty_name=str(model_id),
            storage_size=Memory.from_mb(100000),
            n_layers=32,
        ),
        device_rank=device_rank,
        world_size=world_size,
        start_layer=0,
        end_layer=32,
        n_layers=32,
    )


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=True,
    )
    yield caplog
    logger.remove(handler_id)
