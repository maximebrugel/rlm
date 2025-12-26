from rlm.clients import get_client, BaseLM
from rlm.environments import get_environment, BaseEnv
from rlm.core.lm_handler import LMHandler
from rlm.core.types import (
    RLMIteration,
    REPLResult,
    CodeBlock,
    ClientBackend,
    EnvironmentType,
)
from rlm.utils.prompts import RLM_SYSTEM_PROMPT
from rlm.utils.parsing import (
    extract_code_blocks,
    extract_final_answer,
    format_iteration,
)

from typing import Dict, Any, Optional, List


class RLM:
    """
    Recursive Language Model class that the user instantiates and runs on their tasks.
    """

    def __init__(
        self,
        backend: ClientBackend = "openai",
        backend_kwargs: Dict[str, Any] = {},
        environment: EnvironmentType = "local",
        environment_kwargs: Dict[str, Any] = {},
        max_depth: int = 1,
        max_iterations: int = 10,
        custom_system_prompt: Optional[str] = None,
        other_backends: Optional[List[ClientBackend]] = None,
        other_backend_kwargs: Optional[List[Dict[str, Any]]] = None,
    ):
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.system_prompt = (
            custom_system_prompt if custom_system_prompt else RLM_SYSTEM_PROMPT
        )

        # Create client and wrap in handler
        client: BaseLM = get_client(backend, backend_kwargs)
        self.lm_handler = LMHandler(client)

        # Register other clients to be available as sub-call options
        if other_backends:
            for backend, kwargs in zip(other_backends, other_backend_kwargs):
                client: BaseLM = get_client(backend, kwargs)
                self.lm_handler.register_client(client.model_name, client)

        self.lm_handler.start()

        # Pass handler address to environment so it can make llm_query() calls
        environment_kwargs["LM_HANDLER_HOST"] = self.lm_handler.host
        environment_kwargs["LM_HANDLER_PORT"] = self.lm_handler.port

        self.environment: BaseEnv = get_environment(environment, **environment_kwargs)

    def setup(self) -> None:
        # TODO: Set up system prompt
        pass

    def _setup_prompt(self, prompt: str | Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Setup the prompt for the RLM.
        """
        message_history = []
        message_history.append({"role": "user", "content": prompt})
        return message_history

    def completion(self, prompt: str | Dict[str, Any]) -> str:
        """
        Recursive Language Model completion call.
        Args:
            prompt: A single string or dictionary of messages to pass to the model.
        Returns:
            A final answer as a completion object.
        """

        message_history = self._setup_prompt(prompt)

        for i in range(self.max_iterations):
            iteration: RLMIteration = self._completion_turn(message_history)

            # Check if RLM is done and has a final answer.
            final_answer = extract_final_answer(iteration)
            if final_answer:
                return final_answer

            # Format the iteration for the next prompt.
            new_messages = format_iteration(iteration)
            message_history.extend(new_messages)

        # Default behavior: we run out of iterations, provide one final answer
        final_answer = self._default_answer(message_history)
        return final_answer

    def _completion_turn(self, prompt: str | Dict[str, Any]) -> RLMIteration:
        """
        Perform a single iteration of the RLM, including prompting the model
        and code execution + tool execution.
        """
        response = self.lm_handler.completion(prompt)
        code_block_strs = extract_code_blocks(response)
        code_blocks = []

        # TODO: Handle this correctly
        for code_block_str in code_block_strs:
            code_result: REPLResult = self.environment.execute_code(code_block_str)
            code_blocks.append(CodeBlock(code=code_block_str, result=code_result))

        return RLMIteration(response=response, code_blocks=code_blocks)

    def _default_answer(self, message_history: List[Dict[str, Any]]) -> str:
        """
        TODO: Implement this.
        """
        return "The RLM ran out of iterations and did not find a final answer."

    def cleanup(self):
        """Stop the LM handler and clean up resources."""
        self.lm_handler.stop()
        if hasattr(self.environment, "cleanup"):
            self.environment.cleanup()

    def __del__(self):
        self.cleanup()
