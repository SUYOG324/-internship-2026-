"""
Tool layer: each tool is (1) a JSON schema Claude sees, and (2) a Python function that
actually executes against local data. This is what turns the app from "chatbot that talks
about maintenance" into "copilot that can look things up and act."
"""

import json
from datetime import datetime, timezone
from typing import Dict, List

from rag import BM25Index


class ToolBox:
    def __init__(self, manual_index: BM25Index, logs: List[Dict], parts: List[Dict]):
        self.manual_index = manual_index
        self.logs = logs
        self.parts = parts

    # ---- schemas exposed to Claude -------------------------------------------------
    def schemas(self) -> List[Dict]:
        return [
            {
                "name": "search_manual",
                "description": (
                    "Search uploaded equipment manuals for relevant sections — fault code "
                    "meanings, troubleshooting steps, PM schedules, spec sheets. Use this "
                    "before answering any question about a specific fault code or procedure."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search text, e.g. 'E-101 low pressure'"},
                        "k": {"type": "integer", "description": "Number of chunks to return", "default": 4},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_maintenance_logs",
                "description": (
                    "Search historical maintenance logs for past incidents matching a fault "
                    "code, machine type, machine id, or symptom keywords. Use this to find "
                    "how similar issues were diagnosed and fixed before."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "fault_code": {"type": "string"},
                        "machine_type": {"type": "string"},
                        "machine_id": {"type": "string"},
                        "keyword": {"type": "string", "description": "Free-text symptom keyword search"},
                    },
                },
            },
            {
                "name": "lookup_spare_parts",
                "description": (
                    "Look up spare parts by part number or by machine type. Returns stock "
                    "level, cost, and lead time so recommendations can flag if a part needs "
                    "to be ordered ahead of a repair."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "part_no": {"type": "string"},
                        "machine_type": {"type": "string"},
                    },
                },
            },
            {
                "name": "predict_failure_causes",
                "description": (
                    "Given a fault code and machine type, cross-reference the manual's fault "
                    "code table with historical log root-causes to rank the most likely "
                    "causes for THIS occurrence, weighted by how often each cause has "
                    "recurred historically."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "fault_code": {"type": "string"},
                        "machine_type": {"type": "string"},
                    },
                    "required": ["fault_code", "machine_type"],
                },
            },
            {
                "name": "create_pm_checklist",
                "description": (
                    "Generate a structured preventive maintenance checklist for a machine "
                    "type at a given interval (daily/weekly/monthly/hours-based), pulled "
                    "from the manual's PM schedule section."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "machine_type": {"type": "string"},
                        "interval": {"type": "string", "description": "e.g. 'daily', 'monthly', 'every 2000 hours'"},
                    },
                    "required": ["machine_type"],
                },
            },
            {
                "name": "generate_service_report",
                "description": (
                    "Produce a structured service report record (JSON) once a diagnosis and "
                    "action have been established in conversation. Call this only near the "
                    "end of a diagnostic conversation, after root cause and action are known."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "machine_id": {"type": "string"},
                        "machine_type": {"type": "string"},
                        "fault_code": {"type": "string"},
                        "symptom": {"type": "string"},
                        "root_cause": {"type": "string"},
                        "action_taken": {"type": "string"},
                        "parts_used": {"type": "array", "items": {"type": "string"}},
                        "technician": {"type": "string"},
                    },
                    "required": ["machine_type", "symptom", "root_cause", "action_taken"],
                },
            },
        ]

    # ---- execution -------------------------------------------------------------------
    def run(self, name: str, tool_input: Dict) -> Dict:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return {"error": f"Unknown tool '{name}'"}
        try:
            return handler(**tool_input)
        except TypeError as e:
            return {"error": f"Bad arguments for {name}: {e}"}

    def _tool_search_manual(self, query: str, k: int = 4) -> Dict:
        results = self.manual_index.query(query, k=k)
        return {"results": results, "count": len(results)}

    def _tool_search_maintenance_logs(self, fault_code: str = None, machine_type: str = None,
                                       machine_id: str = None, keyword: str = None) -> Dict:
        matches = []
        for log in self.logs:
            if fault_code and log.get("fault_code", "").lower() != fault_code.lower():
                continue
            if machine_type and machine_type.lower() not in log.get("machine_type", "").lower():
                continue
            if machine_id and machine_id.lower() != log.get("machine_id", "").lower():
                continue
            if keyword and keyword.lower() not in json.dumps(log).lower():
                continue
            matches.append(log)
        return {"matches": matches, "count": len(matches)}

    def _tool_lookup_spare_parts(self, part_no: str = None, machine_type: str = None) -> Dict:
        matches = []
        for part in self.parts:
            if part_no and part["part_no"].lower() != part_no.lower():
                continue
            if machine_type and machine_type.lower() not in " ".join(
                m.lower() for m in part["compatible_machines"]
            ):
                continue
            matches.append(part)
        return {"matches": matches, "count": len(matches)}

    def _tool_predict_failure_causes(self, fault_code: str, machine_type: str) -> Dict:
        # Pull manual context for this fault code.
        manual_hits = self.manual_index.query(f"{fault_code} {machine_type}", k=3)

        # Tally historical root causes for this exact fault code + machine type.
        cause_counts: Dict[str, int] = {}
        matching_logs = []
        for log in self.logs:
            if log.get("fault_code", "").lower() == fault_code.lower() and \
               machine_type.lower() in log.get("machine_type", "").lower():
                cause = log["root_cause"]
                cause_counts[cause] = cause_counts.get(cause, 0) + 1
                matching_logs.append(log)

        ranked_causes = sorted(cause_counts.items(), key=lambda x: x[1], reverse=True)
        return {
            "fault_code": fault_code,
            "machine_type": machine_type,
            "manual_context": manual_hits,
            "historically_ranked_causes": [
                {"cause": c, "times_seen": n} for c, n in ranked_causes
            ],
            "supporting_logs": matching_logs,
            "note": (
                "If historically_ranked_causes is empty, no exact-match logs exist yet — "
                "rely on manual_context only and say so explicitly."
            ),
        }

    def _tool_create_pm_checklist(self, machine_type: str, interval: str = None) -> Dict:
        query = f"{machine_type} preventive maintenance schedule {interval or ''}"
        results = self.manual_index.query(query, k=5)
        return {
            "machine_type": machine_type,
            "interval_requested": interval,
            "manual_context": results,
            "note": "Build the final checklist from manual_context; do not invent steps not grounded in it.",
        }

    def _tool_generate_service_report(self, machine_type: str, symptom: str, root_cause: str,
                                       action_taken: str, machine_id: str = None,
                                       fault_code: str = None, parts_used: List[str] = None,
                                       technician: str = None) -> Dict:
        report = {
            "report_id": f"SR-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "machine_id": machine_id,
            "machine_type": machine_type,
            "fault_code": fault_code,
            "symptom": symptom,
            "root_cause": root_cause,
            "action_taken": action_taken,
            "parts_used": parts_used or [],
            "technician": technician,
        }
        return {"report": report}
