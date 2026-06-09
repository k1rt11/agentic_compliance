"""
Shared data type for the compliance experiment.

An AgentTurn captures the two stages we compare:
  intent_text what the agent says it will do
  tool_calls  the (tool_name, args) calls it then makes

The real experiment lives in llm_agent.py, which fills an AgentTurn from a
live model, executes the tool calls against mock_tools, and scores stated
vs enacted compliance with compliance_checker.
"""

from dataclasses import dataclass


@dataclass
class AgentTurn:
    intent_text: str
    tool_calls: list[tuple]
