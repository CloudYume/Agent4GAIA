"""Provenance-bearing memory scoped to a single GAIA task."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import Evidence, Observation, StageResult


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class TaskMemory:
    task_id: str
    evidence: list[Evidence] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    stages: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_previous(cls, task_id: str, previous: dict[str, Any]) -> TaskMemory:
        saved = previous.get("task_memory")
        if not isinstance(saved, dict) or saved.get("task_id") != task_id:
            return cls(task_id)
        return cls(
            task_id=task_id,
            evidence=[Evidence(**item) for item in saved.get("evidence", [])],
            observations=[Observation(**item) for item in saved.get("observations", [])],
            stages=list(saved.get("stages", [])),
            warnings=list(saved.get("warnings", [])),
        )

    def add_prepared(self, prepared: Any, attachment_path: str = "") -> None:
        for warning in prepared.warnings:
            if warning not in self.warnings:
                self.warnings.append(warning)
        if prepared.review_text:
            text = prepared.review_text
            self.add_evidence(Evidence(
                source_id=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                kind="attachment_text", locator=attachment_path or "prepared_input",
                status="extracted", excerpt=text[:300],
                sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                truncated=len(text) > 300,
            ))
        for path, timestamp, label in prepared.review_images:
            image = Path(path)
            if not image.is_file():
                continue
            digest = _sha256_file(image)
            self.add_observation(Observation(
                asset_id=digest, kind="prepared_image", label=label,
                locator=str(image), timestamp=timestamp,
            ))

    def add_evidence(self, item: Evidence) -> None:
        if item not in self.evidence:
            self.evidence.append(item)

    def add_observation(self, item: Observation) -> None:
        if item not in self.observations:
            self.observations.append(item)

    def add_model_result(self, result: StageResult) -> None:
        for source in result.payload.get("sources", []):
            if not isinstance(source, dict):
                continue
            url = str(source.get("url") or "")
            if not url:
                continue
            self.add_evidence(Evidence(
                source_id=hashlib.sha256(url.encode("utf-8")).hexdigest(),
                kind="web", locator=url, status=str(source.get("status") or "unknown"),
                excerpt=str(source.get("excerpt") or source.get("snippet") or "")[:1_200],
                sha256=str(source.get("sha256") or ""),
                truncated=bool(source.get("truncated")),
                warning=str(source.get("warning") or ""),
            ))
        for url in result.payload.get("citations", []):
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                self.add_evidence(Evidence(
                    source_id=hashlib.sha256(url.encode("utf-8")).hexdigest(),
                    kind="web", locator=url, status="cited",
                ))
        for trace in result.payload.get("tool_trace", []):
            if not isinstance(trace, dict):
                continue
            tool = trace.get("tool")
            if tool == "query_attachment":
                locator = str(trace.get("source_path") or "")
                digest = str(trace.get("sha256") or "")
                preview = json.dumps(trace.get("result_preview"), ensure_ascii=False)[:1_200]
                self.add_evidence(Evidence(
                    source_id=digest or hashlib.sha256(locator.encode("utf-8")).hexdigest(),
                    kind="data_query", locator=locator,
                    status="failed" if trace.get("error") else "queried",
                    excerpt=preview, sha256=digest,
                    truncated=bool(trace.get("truncated")),
                    warning=str(trace.get("warning") or ""),
                ))
            elif tool == "calculate":
                expression_hash = str(trace.get("expression_sha256") or "")
                self.add_evidence(Evidence(
                    source_id=expression_hash, kind="calculation", locator=expression_hash,
                    status="failed" if trace.get("error") else "computed",
                    excerpt=str(trace.get("result") or trace.get("error") or "")[:1_200],
                    truncated=bool(trace.get("truncated")),
                    warning=str(trace.get("warning") or ""),
                ))
        prepared_images = [item for item in self.observations if item.kind == "prepared_image"]
        for visual in result.payload.get("vision_outputs", []):
            if not isinstance(visual, dict):
                continue
            image_index = visual.get("image_index")
            original = prepared_images[image_index] if isinstance(image_index, int) and 0 <= image_index < len(prepared_images) else None
            self.add_observation(Observation(
                asset_id=str(visual.get("asset_id") or (original.asset_id if original else "") or visual.get("sha256") or ""),
                kind="visual_inspection", label=str(visual.get("label") or ""),
                locator=str(visual.get("locator") or (original.locator if original else "") or visual.get("label") or ""),
                timestamp=visual.get("timestamp", original.timestamp if original else None),
                text=str(visual.get("observation") or visual.get("description") or visual.get("text") or "")[:6_000],
                uncertain=bool(visual.get("uncertain", False)),
                question=str(visual.get("question") or "")[:500],
                region=(visual.get("region") if isinstance(visual.get("region"), str)
                        else json.dumps(visual.get("region"), ensure_ascii=False) if visual.get("region") is not None else ""),
                cache_hit=bool(visual.get("cache_hit", False)),
            ))
        self.stages.append(result.summary())

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "evidence": [item.to_record() for item in self.evidence],
            "observations": [item.to_record() for item in self.observations],
            "stages": self.stages,
            "warnings": self.warnings,
        }
