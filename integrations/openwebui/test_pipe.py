import unittest
from unittest.mock import patch

from horizon_relay import Relay, SimulatedProvider

import horizon_relay_pipe as pipe_module


class PipeTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_registered(self):
        self.assertEqual(pipe_module.Pipe().pipes()[0]["id"], "demo")

    async def test_streaming_and_nonstreaming_return_content_without_emitter(self):
        for stream in (False, True):
            result = await pipe_module.Pipe().pipe({"stream": stream, "messages": [{"role": "user", "content": "/relay-demo local"}]})
            self.assertIn("SIMULATED", result)
            self.assertIn("0 cloud", result)

    async def test_background_bypasses_engine(self):
        with patch.object(pipe_module, "run_demo", side_effect=AssertionError("Must not run")):
            result = await pipe_module.Pipe().pipe({}, __task__="title_generation")
        self.assertIn('"title"', result)

    async def test_progress_is_one_status_line_plus_live_graph(self):
        events = []
        async def emit(event):
            events.append(event)
        result = await pipe_module.Pipe().pipe({"messages": [{"role": "user", "content": "/relay-demo escalate"}]}, __event_emitter__=emit)
        self.assertIn("Engineering plan", result)
        statuses = [e for e in events if e["type"] == "status"]
        embeds = [e for e in events if e["type"] == "embeds"]
        self.assertEqual([s["data"]["done"] for s in statuses], [False, True])
        self.assertTrue(statuses[0]["data"]["description"].startswith("[SIMULATED] "))
        self.assertGreater(len(embeds), 3)
        self.assertTrue(all(e["data"]["replace"] and len(e["data"]["embeds"]) == 1 for e in embeds))
        final = embeds[-1]["data"]["embeds"][0]
        self.assertIn('"status": "completed"', final)
        self.assertIn('"type": "escalated"', final)
        self.assertNotIn("/*RELAY_GRAPH*/null", final)

    async def test_does_not_answer_arbitrary_prompts_with_fixture(self):
        result = await pipe_module.Pipe().pipe({"messages": [{"role": "user", "content": "What is my budget?"}]})
        self.assertIn("/relay-demo", result)
        self.assertNotIn("Engineering plan", result)

    async def test_rejects_attachments_and_multimodal(self):
        result = await pipe_module.Pipe().pipe({"messages": []}, __files__=[{"id": "file"}])
        self.assertIn("remove attachments", result)
        result = await pipe_module.Pipe().pipe({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]})
        self.assertIn("text messages only", result)

    async def test_explicitly_rejects_unimplemented_real_provider(self):
        with patch.dict("os.environ", {"RELAY_PROVIDER": "real"}):
            with self.assertRaisesRegex(ValueError, "Unknown RELAY_PROVIDER"):
                await pipe_module.Pipe().pipe({})

    async def test_live_mode_uses_relay_chat_and_reports_errors(self):
        events = []
        async def emit(event):
            events.append(event)
        with patch.dict("os.environ", {"RELAY_PROVIDER": "live"}), \
                patch.object(pipe_module.Relay, "from_env", return_value=Relay(SimulatedProvider())):
            result = await pipe_module.Pipe().pipe(
                {"model": "horizon_relay.live", "messages": [{"role": "user", "content": "Summarize this"}]},
                __event_emitter__=emit)
            self.assertIn("---\nRelay:", result)
            self.assertTrue(any(e["type"] == "embeds" for e in events))
            self.assertTrue(events[-1]["data"]["done"])
            stopped = await pipe_module.Pipe().pipe({"model": "horizon_relay.live", "messages": []})
            self.assertIn("Relay stopped", stopped)


class FakeChats:
    def __init__(self):
        self.cleared, self.saved = [], []

    async def clear_tests_folder(self, user_id, keep_chat_id=None):
        self.cleared.append(keep_chat_id)
        return "tests-folder"

    async def save_chat(self, user_id, folder_id, title, prompt, content, embed, final_status):
        self.saved.append({"folder": folder_id, "title": title, "prompt": prompt, "content": content, "embed": embed})
        return f"chat-{len(self.saved)}"


class TestRunnerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.pipe = pipe_module.Pipe()
        self.pipe.chats = FakeChats()
        self.env = patch.dict("os.environ", {"RELAY_PROVIDER": "live"})
        self.env.start()
        self.addCleanup(self.env.stop)

    async def test_lists_tests_without_running_anything(self):
        with patch.object(pipe_module, "run_case", side_effect=AssertionError("Must not run")):
            reply = await self.pipe.pipe({"model": "horizon_relay.tests", "messages": [{"role": "user", "content": "hi"}]})
        self.assertIn("| 1 | City extraction |", reply)
        self.assertIn("tests", [p["id"] for p in self.pipe.pipes()])

    async def test_run_saves_one_chat_per_test_with_its_graph(self):
        events = []
        async def emit(event):
            events.append(event)
        with patch.object(pipe_module.Relay, "from_env", return_value=Relay(SimulatedProvider("trusted"))):
            reply = await self.pipe.pipe({"model": "horizon_relay.tests", "messages": [{"role": "user", "content": "/run 1 3"}]},
                                         __event_emitter__=emit, __user__={"id": "u1"}, __chat_id__="runner-chat")
        saved = self.pipe.chats.saved
        self.assertEqual(self.pipe.chats.cleared, ["runner-chat"])
        self.assertTrue(all(c["folder"] == "tests-folder" for c in saved))
        self.assertEqual([c["title"] for c in saved], ["✓ 01 City extraction", "✓ 02 Simple multiplication"])
        self.assertTrue(all("const G = {" in c["embed"] for c in saved))
        self.assertIn("2/2 routed as expected", reply)
        self.assertIn("[open](/c/chat-1)", reply)
        self.assertEqual(events[-1]["data"]["description"], "2/2 routed as expected")
        self.assertEqual(events[-2]["data"], {"embeds": [], "replace": True})


if __name__ == "__main__":
    unittest.main()
