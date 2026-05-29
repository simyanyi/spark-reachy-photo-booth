# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from workmesh.config import BaseConfig

logger = logging.getLogger(__name__)


class AgentServiceConfig(BaseConfig):
    nat_config: Path = Field(
        description="The path to the NAT agent configuration file.",
        default=Path("configs/photobooth.yml"),
    )
    user_utterance_threshold: int = Field(
        default=1_000,
        description="""
The difference in milliseconds between the timestamp of the user utterance
and the timestamp of the last message in the history.
If the difference is greater than the threshold,
the user utterance is considered invalid and is not processed.""",
        ge=0,
        le=10_000,
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for the LLM service. If not provided, will attempt to download from HuggingFace.",
    )

    @field_validator("nat_config", mode="after")
    def nat_config_validator(cls, value: Path) -> Path:
        if not value.resolve().exists():
            raise ValueError(f"NAT agent configuration file not found: {value}")
        return value

    @model_validator(mode="after")
    def download_openai_api_key_if_needed(self) -> "AgentServiceConfig":
        """Download OpenAI API key from HuggingFace if not provided via env or config."""
        # Check env var first
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key and env_key.strip():
            self.openai_api_key = env_key
            logger.info("Using OPENAI_API_KEY from environment")
            return self

        # If no key from env/config, try to download from HuggingFace
        if not self.openai_api_key or not self.openai_api_key.strip():
            logger.info("OPENAI_API_KEY not set, attempting to download from HuggingFace...")
            try:
                from gradio_client import Client

                client = Client("HuggingFaceM4/gradium_setup", verbose=False)
                key, _ = client.predict(api_name="/claim_b_key")
                if key and key.strip():
                    self.openai_api_key = key
                    logger.info("Successfully downloaded OpenAI API key from HuggingFace")
                    # Optionally persist to env for later use
                    os.environ["OPENAI_API_KEY"] = key
                else:
                    logger.warning("HuggingFace returned empty API key")
            except Exception as e:
                logger.warning(f"Failed to download OpenAI API key from HuggingFace: {e}")

        return self
