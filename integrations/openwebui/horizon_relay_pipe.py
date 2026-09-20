"""
title: Horizon Relay
author: Horizon Relay
version: 0.5.0
description: Local Horizon models answer and are scored; the cloud plans only when needed. Includes a routing test runner and a simulated demo.
"""

# Open WebUI adapter only: request/response translation and progress display. Routing lives in horizon_relay.

import json
import os
import re
import time
from uuid import uuid4

from horizon_relay import Relay, RelayError, RelayEvent
from horizon_relay.demo import SCENARIOS, run_demo
from horizon_relay.evaluation import load_cases, run_case, select_cases, summarize
from horizon_relay.graph import RunGraph, render_html

# Events that change what the decision graph shows. Each one re-sends the graph embed.
GRAPH_UPDATES = {"local_skipped", "local_solve", "local_sample", "local_critique", "local_verdict", "extract_failed", "escalated",
                 "local_accepted", "answering", "plan_invalid", "plan_created", "task_started", "task_escalated",
                 "task_invalid", "task_completed", "synthesizing"}

DEMO_HELP = ("**Horizon Relay — simulated prototype**\n\n"
             "Run `/relay-demo`, `/relay-demo trusted`, `/relay-demo local`, `/relay-demo repair`, or `/relay-demo escalate`. "
             "These use two built-in bug reports and scripted responses, not your conversation or a real model.")


def status(description: str, done: bool) -> dict:
    return {"type": "status", "data": {"action": "horizon_relay", "description": description, "done": done}}


def graph_embed(graph: RunGraph) -> dict:
    """The decision graph as a message embed. Open WebUI renders it in a frame and saves it with the chat;
    replace=True swaps in the latest version instead of stacking copies."""
    return {"type": "embeds", "data": {"embeds": [render_html(graph.to_dict())], "replace": True}}


def graph_emitter(event_emitter, statuses=True):
    """Relay event callback that keeps a live decision graph in the message and one short status line."""
    graph = RunGraph()

    async def emit(event: RelayEvent):
        graph.add(event)
        prefix = "[SIMULATED] " if event.simulated else ""
        if statuses and event.name == "run_started":
            await event_emitter(status(prefix + "Relay running", False))
        if event.name in GRAPH_UPDATES or event.done:
            await event_emitter(graph_embed(graph))
        if statuses and event.done:
            await event_emitter(status(prefix + event.message, True))
    return emit


TESTS_FOLDER = "Tests"
TESTS_HELP = ("**Horizon Relay routing tests**\n\n"
              "Send `/run all` to run every test (a bare `/run` opens Open WebUI's command menu instead), or narrow it: `/run 1 4 9`, `/run 0.9b`, `/run 4b`, `/run cloud`, `/run forced`, `/run hard`. "
              "Each run first empties the **Tests** folder, then saves every test as its own chat there, with its decision graph. "
              "**Expected** lists the models allowed to answer; ✓/✗ under **Answer** checks the answer itself. "
              "*Forced* tests override the policy so a path always runs. Tests use the real models; cloud routes spend credits.\n\n")


class OpenWebUIChats:
    """Saves test runs as chats in folders. Uses Open WebUI internals, so it only works inside Open WebUI."""

    async def clear_tests_folder(self, user_id: str, keep_chat_id: str | None = None) -> str:
        """Empty the Tests folder (its chats and any subfolders), creating it if needed. The chat running the
        tests is kept, moved to the top of the folder if it was in a subfolder. Returns the folder's id."""
        from open_webui.models.chats import Chats
        from open_webui.models.folders import FolderForm, Folders
        folder = await Folders.get_folder_by_parent_id_and_user_id_and_name(None, user_id, TESTS_FOLDER)
        folder = folder or await Folders.insert_new_folder(user_id, FolderForm(name=TESTS_FOLDER))
        subtree = await Folders.get_folder_ids_by_id_and_user_id_in_subtree(folder.id, user_id)
        for chat in await Chats.get_chats_by_folder_ids_and_user_id(subtree, user_id):
            if chat.id == keep_chat_id:
                await Chats.update_chat_folder_id_by_id_and_user_id(chat.id, user_id, folder.id)
            else:
                await Chats.delete_chat_by_id_and_user_id(chat.id, user_id)
        for child in await Folders.get_folders_by_parent_id_and_user_id(folder.id, user_id):
            await Folders.delete_folder_by_id_and_user_id(child.id, user_id)
        return folder.id

    async def save_chat(self, user_id: str, folder_id: str, title: str, prompt: str, content: str,
                        embed: str, final_status: str) -> str:
        from open_webui.models.chats import ChatForm, Chats
        model, now = "horizon_relay.live", int(time.time())
        user_id_msg, reply_id, chat_id = str(uuid4()), str(uuid4()), str(uuid4())
        question = {"id": user_id_msg, "parentId": None, "childrenIds": [reply_id], "role": "user",
                    "content": prompt, "timestamp": now, "models": [model]}
        reply = {"id": reply_id, "parentId": user_id_msg, "childrenIds": [], "role": "assistant", "content": content,
                 "done": True, "model": model, "modelName": "Horizon Relay", "modelIdx": 0, "timestamp": now,
                 "statusHistory": [{"action": "horizon_relay", "description": final_status, "done": True}],
                 "embeds": [embed]}
        chat = {"id": "", "title": title, "models": [model], "params": {}, "tags": [], "files": [], "timestamp": now * 1000,
                "history": {"messages": {user_id_msg: question, reply_id: reply}, "currentId": reply_id},
                "messages": [question, reply]}
        await Chats.insert_new_chat(chat_id, user_id, ChatForm(chat=chat, folder_id=folder_id))
        return chat_id


def short_model(name: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)?B)", name or "", re.I)
    return match.group(1) if match else name


def demo_background_reply(task) -> str:
    """Canned title/tag/follow-up replies so the demo never spends a model call on them."""
    task = str(task)
    if "title" in task:
        return json.dumps({"title": "Horizon Relay simulated demo"})
    if "tag" in task:
        return json.dumps({"tags": ["Demo"]})
    if "follow" in task:
        return json.dumps({"follow_ups": []})
    return "Simulated background task; no model called."


class Pipe:
    chats = OpenWebUIChats()

    def pipes(self):
        if os.getenv("RELAY_PROVIDER", "simulated") == "live":
            return [{"id": "live", "name": "Horizon Relay"}, {"id": "tests", "name": "Horizon Relay Tests"},
                    {"id": "demo", "name": "Horizon Relay (Simulated Demo)"}]
        return [{"id": "demo", "name": "Horizon Relay (Simulated Demo)"}]

    async def pipe(self, body: dict, __event_emitter__=None, __task__=None,
                   __files__=None, __user__=None, __metadata__=None, __chat_id__=None):
        mode = os.getenv("RELAY_PROVIDER", "simulated")
        if mode not in ("live", "simulated"):
            raise ValueError("Unknown RELAY_PROVIDER; use live or simulated")
        if __files__ or body.get("files") or (__metadata__ or {}).get("files"):
            return "Horizon Relay currently supports text only; remove attachments to continue."

        emit = graph_emitter(__event_emitter__) if __event_emitter__ else None

        try:
            if mode == "live" and body.get("model", "").endswith(".tests"):
                if __task__ is not None:
                    return json.dumps({"title": "Relay test run"}) if "title" in str(__task__) else json.dumps({"tags": ["Tests"]})
                return await self.tests(body, __user__ or {}, __event_emitter__, __chat_id__)
            if mode == "live" and not body.get("model", "").endswith(".demo"):
                relay = Relay.from_env()
                if __task__ is not None:  # title, tags, follow-ups: one local call, no plan, no cloud
                    return await relay.complete(body.get("messages", []))
                result = await relay.chat(body.get("messages", []), emit=emit)
                return result.content + "\n\n---\n" + result.summary()
            return await self.demo(body, __task__, emit)
        except RelayError as exc:
            return f"**Relay stopped — no complete answer.** {exc}"

    async def tests(self, body, user, event_emitter, chat_id=None):
        """Run routing tests; each one is saved as a chat with its decision graph."""
        cases = load_cases()
        relay = Relay.from_env()
        names = {tier: short_model(model) for tier, model in relay.provider.models.items()}
        expected = lambda case: " or ".join(names.get(t, t) for t in case.expected_route)
        messages = body.get("messages", [])
        latest = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        latest = latest.strip() if isinstance(latest, str) else ""
        if not latest.startswith("/run"):
            rows = "\n".join(f"| {i} | {c.name}{' *(forced)*' if c.policy else ''} | {c.category} | {c.difficulty} | {expected(c)} | "
                              f"{'✓' if c.answer_pattern else ''} |" for i, c in enumerate(cases, 1))
            return (TESTS_HELP + "| # | Test | Category | Difficulty | Expected | Answer checked |\n|---|---|---|---|---|---|\n"
                    + rows)
        selected = select_cases(cases, latest[len("/run"):])
        if not selected:
            return "No test matches that selection. Send any other message to list the tests."
        async def send(event):
            if event_emitter:
                await event_emitter(event)
        user_id = user.get("id")
        folder_id = None
        if user_id:
            await send(status("Clearing the Tests folder", False))
            folder_id = await self.chats.clear_tests_folder(user_id, keep_chat_id=chat_id)
        results, links = [], []
        for i, case in enumerate(selected, 1):
            await send(status(f"Test {i}/{len(selected)}: {case.name}", False))
            live = graph_emitter(event_emitter, statuses=False) if event_emitter else None
            result = await run_case(relay, case, emit=live)
            results.append(result)
            link = ""
            if user_id:
                check = "" if result.correct is None else f" Answer check: {'passed' if result.correct else 'failed'}."
                content = (result.answer + "\n\n---\n" + f"Expected **{expected(case)}**, routed to **{names.get(result.route, result.route)}**.{check}"
                           if result.route != "error" else f"**Relay stopped — no complete answer.** {result.error}")
                chat_id = await self.chats.save_chat(
                    user_id, folder_id, f"{'✓' if result.matched else '✗'} {i:02d} {case.name}", case.prompt, content,
                    render_html(result.graph), f"Expected {expected(case)} · routed to {names.get(result.route, result.route)}")
                link = f"[open](/c/{chat_id})"
            links.append(link)
        await send({"type": "embeds", "data": {"embeds": [], "replace": True}})
        summary = summarize(results)
        await send(status(f"{summary['matched']}/{summary['cases']} routed as expected", True))
        rows = []
        for i, (r, link) in enumerate(zip(results, links), 1):
            scores = " · ".join(f"{names.get(t, t)} {s:.2f}" for t, s in r.scores.items()) or "–"
            route = names.get(r.route, r.route)
            answer = "–" if r.correct is None else "✓" if r.correct else "✗"
            rows.append(f"| {i} | {r.case.name} | {expected(r.case)} | {route} | {'✓' if r.matched else '✗'} | {answer} | "
                        f"{scores} | {r.cloud_calls} | {r.elapsed_s:.1f}s | {link} |")
        tiers = " · ".join(f"{names.get(t, t)} {n}" for t, n in summary["by_tier"].items() if n)
        return (f"**{summary['matched']}/{summary['cases']} routed as expected** · "
                f"**{summary['answers_correct']}/{summary['answers_checked']} answers correct** · answered by {tiers or 'none'} · "
                f"{summary['errors']} errors · {summary['cloud_calls']} cloud calls · {summary['seconds']}s\n\n"
                "| # | Test | Expected | Routed to | Route | Answer | Scores | Cloud calls | Time | Chat |\n"
                "|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows)
                + (f"\n\nEach test is saved in **{TESTS_FOLDER}** in the sidebar with its decision graph." if user_id else ""))

    async def demo(self, body, task, emit):
        if task is not None:
            return demo_background_reply(task)
        messages = body.get("messages", [])
        if not isinstance(messages, list) or any(not isinstance(m, dict) or not isinstance(m.get("content"), str) for m in messages):
            return "This prototype supports text messages only."
        latest = next((m["content"].strip() for m in reversed(messages) if m.get("role") == "user"), "")
        parts = latest.split()
        if not parts or parts[0] != "/relay-demo" or len(parts) > 2 or (len(parts) == 2 and parts[1] not in SCENARIOS):
            return DEMO_HELP
        result = await run_demo(parts[1] if len(parts) == 2 else "standard", delay=0.15,
                                cloud_enabled=os.getenv("CLOUD_ENABLED", "true").lower() == "true", emit=emit)
        # Both streaming and non-streaming requests use Open WebUI's normal completion persistence.
        return result.content + "\n\n" + result.summary()
