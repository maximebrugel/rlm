from rlm.clients.base_lm import BaseLM, CostSummary

from typing import Dict, Any, Optional
from portkey_ai import Portkey
from portkey_ai.api_resources.types.chat_complete_type import ChatCompletions


class PortkeyClient(BaseLM):
    """
    LM Client for running models with the Portkey API.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: Optional[str] = "https://api.portkey.ai/v1",
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.client = Portkey(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.call_count = 0
        self.total_cost = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def completion(
        self, prompt: str | Dict[str, Any], model: Optional[str] = None
    ) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        actual_model = model or self.model_name

        # Portkey's .chat.completions.create interface is compatible with OpenAI syntax.
        response = self.client.chat.completions.create(
            model=actual_model,
            messages=messages,
        )
        self._track_cost(response)
        return response.choices[0].message.content

    async def acompletion(
        self, prompt: str | Dict[str, Any], model: Optional[str] = None
    ) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name

        response = await self.client.chat.completions.create(
            model=model, messages=messages
        )
        self._track_cost(response)
        return response.choices[0].message.content

    def _track_cost(self, response: ChatCompletions):
        self.call_count += 1
        self.total_cost += response.usage.total_tokens
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens

    def get_cost_summary(self) -> CostSummary:
        return CostSummary(
            total_calls=self.call_count,
            total_cost=self.total_cost,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
        )
