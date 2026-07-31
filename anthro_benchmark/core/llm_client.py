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
from typing import Optional
from openai import OpenAI


@dataclasses.dataclass
class LLMClient:
    """Base class for LLM clients."""

    model: str
    temperature: float = 0.7
    # --- NEW: Explicitly define fields so __init__ doesn't crash ---
    reasoning_mode: bool = False
    reasoning_effort: Optional[str] = None

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
            # Strip the prefix to get the exact Model ID
            actual_model = self.model.replace("openrouter/", "")
            
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY")
            )
            
            extra_body = kwargs.pop("extra_body", {})
            
            # Use instance variables (self) instead of popping kwargs
            if self.reasoning_mode:
                # OpenRouter standard for exposing reasoning tokens uses an underscore
                extra_body["include_reasoning"] = True
                
            if self.reasoning_effort:
                # Standard OpenAI-compatible param uses an underscore
                kwargs["reasoning_effort"] = self.reasoning_effort
                
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
            
            # Pass reasoning effort to litellm if it's set
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort

            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                **kwargs
            )
            #return response.choices[0].message.content #changing return to...

            message = response.choices[0].message
            content = message.content
            
            # If you want to append the reasoning to the content:
            if self.reasoning_mode and hasattr(message, "model_extra") and message.model_extra:
                reasoning = message.model_extra.get("reasoning")
                if reasoning:
                    return f"<think>\n{reasoning}\n</think>\n\n{content}"
                    
            return content
