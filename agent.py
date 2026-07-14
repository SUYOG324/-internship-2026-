"""
Agent loop: sends the conversation to Claude with tools attached, executes any tool calls
Claude makes against the ToolBox, feeds the results back, and repeats until Claude produces
a final text answer. This is the standard Anthropic tool-use loop.
"""

import os
from typing import List, Dict

import anthropic

from tools import ToolBox

MODEL = "claude-sonnet-4-5"  # swap for any current Claude model string

SYSTEM_PROMPT = """You are the Manufacturing Maintenance Copilot, an AI assistant embedded in a \
factory maintenance team's workflow. You help technicians and maintenance engineers:

- Diagnose equipment issues from symptoms and fault codes
- Ground every technical claim in the uploaded machine manuals or historical logs — use the \
search_manual and search_maintenance_logs tools rather than guessing
- Rank likely failure causes using predict_failure_causes, which combines manual data with \
historical recurrence
- Recommend spare parts with lookup_spare_parts, and proactively flag when stock is 0 or lead \
time is long, since that changes the repair timeline
- Generate preventive maintenance checklists grounded in the manual's PM schedule
- Generate a structured service report with generate_service_report once a diagnosis is settled

Rules:
- Always call search_manual or search_maintenance_logs before making a specific technical claim \
about a fault code, part, or procedure. Do not rely on general knowledge about industrial \
equipment for anything the manuals could answer, since real machines have model-specific quirks.
- When a tool returns no results, say so plainly rather than filling the gap with a guess.
- When recommending a part, always mention if lookup_spare_parts shows 0 in stock or a long \
lead time, and suggest an interim mitigation if the repair is safety-relevant.
- Keep answers scoped to what a technician needs on the shop floor: concrete steps, part \
numbers, and next actions — not generic advice.
- Only call generate_service_report after root cause and action taken are actually known from \
the conversation, not speculatively.
"""


class CopilotAgent:
    def __init__(self, toolbox: ToolBox, api_key: str = None):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.toolbox = toolbox

    def run_turn(self, messages: List[Dict], max_tool_rounds: int = 6) -> Dict:
        """
        messages: list of {"role": "user"|"assistant", "content": ...} in Anthropic format.
        Returns {"messages": updated_messages, "final_text": str, "tool_trace": [...]}
        appended with everything Claude/tools produced this turn, so the caller can persist
        the full history for the next turn.
        """
        tool_trace = []
        working_messages = list(messages)

        for _ in range(max_tool_rounds):
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                tools=self.toolbox.schemas(),
                messages=working_messages,
            )

            working_messages.append({"role": "assistant", "content": response.content})

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                final_text = "".join(b.text for b in response.content if b.type == "text")
                return {"messages": working_messages, "final_text": final_text, "tool_trace": tool_trace}

            tool_results = []
            for call in tool_uses:
                result = self.toolbox.run(call.name, call.input)
                tool_trace.append({"tool": call.name, "input": call.input, "result": result})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": str(result),
                })
            working_messages.append({"role": "user", "content": tool_results})

        # Safety valve: if we somehow loop past max_tool_rounds, force a final answer.
        return {
            "messages": working_messages,
            "final_text": "I gathered a lot of information but hit my tool-call limit for this "
                           "turn — could you narrow the question so I can give you a final answer?",
            "tool_trace": tool_trace,
        }
