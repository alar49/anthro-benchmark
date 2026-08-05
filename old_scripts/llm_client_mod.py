# Copyright 2025 The Anthropomorphism Benchmark Project Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LLM client initialization and management for dialogue generation.
"""

import dataclasses
from typing import Any, Optional

@dataclasses.dataclass
class LLMClient:
    model: str
    temperature: float = 0.7
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    reasoning_effort: Optional[str] = None  # off => "none", on => "medium"/"high"/etc

    def generate(self, messages: list) -> str:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters to pass to the LLM

        Returns:
            Generated text response
        """
        import litellm

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.reasoning_effort is not None:
            kwargs["extra_body"] = {
                "reasoning": {
                    "effort": self.reasoning_effort
                }
            }
        #if self.reasoning_effort is not None:
            #kwargs["reasoning_effort"] = self.reasoning_effort

        response = litellm.completion(**kwargs)
        return response.choices[0].message.content

 
""" 
### ORIGINAL SCRIPT ###
        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        response_text = response.choices[0].message.content
        return response_text
"""
