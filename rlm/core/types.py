from dataclasses import dataclass
from typing import Literal, List

ClientBackend = Literal["openai", "portkey"]
EnvironmentType = Literal["local", "prime", "modal"]


@dataclass
class REPLResult:
    stdout: str
    stderr: str
    locals: dict
    execution_time: float

    def __init__(
        self, stdout: str, stderr: str, locals: dict, execution_time: float = None
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.locals = locals
        self.execution_time = execution_time

    def __str__(self):
        return f"REPLResult(stdout={self.stdout}, stderr={self.stderr}, locals={self.locals}, execution_time={self.execution_time})"


@dataclass
class CodeBlock:
    code: str
    result: REPLResult


@dataclass
class RLMIteration:
    response: str
    code_blocks: List[CodeBlock]
