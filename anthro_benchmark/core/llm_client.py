# Copyright 2025 The Anthropomorphism Benchmark Project Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LLM client initialization and management for dialogue generation.

Alternative Gem_version
"""

import dataclasses
import os
from openai import OpenAI


@dataclasses.dataclass
class LLMClient:
    """Base class for LLM clients."""

    model: str
    temperature: float = 0.7

    def generate(self, messages: list, **kwargs) -> str:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters to pass to the LLM

        Returns:
            Generated text response
        """
        # Route OpenRouter models through the official OpenAI client
        if self.model.startswith("openrouter/"):
            # Strip the prefix to get the exact Model ID (e.g., google/gemma-4-31b-it:free)
            actual_model = self.model.replace("openrouter/", "")
            
            # Point the standard OpenAI client to OpenRouter
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY")
            )
            
            # Extract our custom effort flags from kwargs (if passed down)
            reasoning-mode = kwargs.pop("reasoning-mode", "off")
            reasoning-effort = kwargs.pop("reasoning-effort", None)
            
            extra_body = {}
            
            if reasoning-mode == "on":
                # OpenRouter standard for exposing reasoning tokens in the response
                extra_body["include-reasoning"] = True
                
            if reasoning-effort:
                # Standard OpenAI-compatible param for o-series and similar effort-based models
                kwargs["reasoning-effort"] = reasoning-effort
                
            if extra_body:
                kwargs["extra_body"] = extra_body

            # Execute the call
            response = client.chat.completions.create(
                model=actual_model,
                messages=messages,
                temperature=self.temperature,
                **kwargs
            )
            
            # Return the generated text
            return response.choices[0].message.content
            
        # Fallback to litellm for all other providers
        else:
            import litellm

            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                **kwargs
            )
            return response.choices[0].message.content
