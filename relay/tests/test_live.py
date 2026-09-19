import asyncio
import importlib.util
import json
import unittest

if importlib.util.find_spec("httpx"):
    import httpx
    from horizon_relay.live import Endpoint, LiveProvider

from horizon_relay import RelayEngine
from horizon_relay.schemas import RelayError


@unittest.skipUnless(importlib.util.find_spec("httpx"), "Install the live extra to test HTTP adapters")
class LiveTests(unittest.IsolatedAsyncioTestCase):
    def provider(self, handler):
        return LiveProvider(Endpoint("http://local/v1", "small"),
                            Endpoint("https://cloud/v1", "large", "test-secret"),
                            transport=httpx.MockTransport(handler))

    async def test_credentials_only_reach_cloud_and_usage_preserved(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"content": '```json\n{"needs_plan":true}\n```'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4}})
        provider = self.provider(handler)
        local = await provider.complete("local", "attempt", {"goal": "Goal", "sources": {"text": "source"}})
        await provider.complete("cloud", "plan", {"goal": "Goal", "sources": {"text": "source"}})
        self.assertNotIn("authorization", requests[0].headers)
        self.assertEqual(requests[1].headers["authorization"], "Bearer test-secret")
        self.assertNotIn("test-secret", requests[1].content.decode())
        self.assertNotIn("test-secret", repr(provider.cloud))
        self.assertEqual(local.value, {"needs_plan": True})
        self.assertEqual(local.prompt_tokens, 12)

    async def test_errors_do_not_echo_credentials_or_response_body(self):
        provider = self.provider(lambda r: httpx.Response(401, text="test-secret request was invalid"))
        with self.assertRaises(RelayError) as caught:
            await provider.complete("cloud", "plan", {})
        self.assertIn("401", str(caught.exception))
        self.assertNotIn("test-secret", str(caught.exception))

    async def test_malformed_json_is_returned_for_bounded_validation(self):
        provider = self.provider(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]}))
        reply = await provider.complete("cloud", "plan", {})
        self.assertEqual(reply.value, "not json")
        self.assertIsNone(reply.completion_tokens)

    async def test_k2_inline_reasoning_is_not_returned_to_user(self):
        provider = self.provider(lambda r: httpx.Response(200, json={"choices": [{"message": {
            "content": 'Private reasoning here.\n</ifm|think_faster>\n{"answer":"Hello!"}'}}]}))
        reply = await provider.complete("local", "attempt", {})
        self.assertEqual(reply.value, {"answer": "Hello!"})

    async def test_exact_extraction_uses_focused_prompt(self):
        seen = []
        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"report_id":"text","quote":"Friday"}'}}]})
        await self.provider(handler).complete("local", "attempt", {
            "goal": "Extract text", "sources": {"text": "Friday"}, "local_extract": True,
        })
        self.assertEqual(seen[0]["messages"][1]["content"], "Friday")
        self.assertIn('"text"', seen[0]["messages"][0]["content"])

    async def test_truncation_is_not_a_success(self):
        provider = self.provider(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}))
        with self.assertRaisesRegex(RelayError, "token limit"):
            await provider.complete("local", "attempt", {})

    async def test_cancellation_propagates_to_http_transport(self):
        started = asyncio.Event()
        closed = asyncio.Event()
        async def handler(request):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
        job = asyncio.create_task(self.provider(handler).complete("local", "attempt", {}))
        await asyncio.wait_for(started.wait(), 1)
        job.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await job
        self.assertTrue(closed.is_set())

    async def test_real_protocol_plan_workers_synthesis(self):
        operations = []
        def handler(request):
            body = json.loads(request.content)
            payload = json.loads(body["messages"][1]["content"])
            if "task" in payload:
                operations.append("work")
                self.assertEqual(body["model"], "small")
                value = {"answer": "A bounded analysis of the supplied text."}
            elif "results" in payload:
                operations.append("synthesize")
                self.assertIn("conversation", payload["sources"])
                value = "Final answer for the actual request."
            elif "validation_error" in payload:
                operations.append("plan")
                value = {"version": 1, "final_instruction": "Answer the question", "tasks": [
                    {"id": "analyze", "instruction": "Analyze the text", "preferred_tier": "local",
                     "input_refs": ["sources.conversation"], "depends_on": [], "result_type": "task_answer", "checks": ["required_fields"]},
                    {"id": "review", "instruction": "Review the analysis", "preferred_tier": "local",
                     "input_refs": ["results.analyze"], "depends_on": ["analyze"], "result_type": "task_answer", "checks": ["required_fields"]}]}
            else:
                operations.append("attempt")
                value = {"needs_plan": True}
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value) if isinstance(value, dict) else value}}]})
        result = await RelayEngine(self.provider(handler)).run("Analyze this", {"conversation": "Actual user input"})
        self.assertEqual(operations, ["attempt", "plan", "work", "work", "synthesize"])
        self.assertFalse(result.metrics["simulated"])
        self.assertNotIn("SIMULATED", result.content)
