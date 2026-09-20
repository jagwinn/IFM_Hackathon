import asyncio
import importlib.util
import json
import unittest

if importlib.util.find_spec("httpx"):
    import httpx
    from horizon_relay.providers.openai_compatible import OpenAICompatibleProvider

from horizon_relay import Endpoint, Relay, RelayEngine, RelayError


@unittest.skipUnless(importlib.util.find_spec("httpx"), "Install the live extra to test HTTP adapters")
class LiveTests(unittest.IsolatedAsyncioTestCase):
    def provider(self, handler):
        return OpenAICompatibleProvider(Endpoint("http://local/v1", "small"),
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

    async def test_http_timeout_follows_the_call_limit(self):
        from horizon_relay import Limits, RelaySettings
        relay = Relay.from_settings(RelaySettings(Endpoint("http://local/v1", "small"), Endpoint("https://c/v1", "large", "k"),
                                                  Limits(call_timeout=42)), transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
        self.assertEqual(relay.provider.timeout, 42)

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
        reply = await provider.complete("local", "solve", {"task": "Hi"})
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

    async def test_solve_requests_and_returns_token_logprobs(self):
        seen = []
        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer": "391"}'}, "logprobs": {"content": [
                {"token": '{"answer": "', "logprob": 0.0, "top_logprobs": []},
                {"token": "391", "logprob": -0.02, "top_logprobs": [{"token": "391", "logprob": -0.02}, {"token": "393", "logprob": -5.5}]},
                {"token": '"}', "logprob": 0.0, "top_logprobs": []}]}}]})
        provider = self.provider(handler)
        reply = await provider.complete("local", "solve", {"task": "17*23"})
        self.assertTrue(seen[0]["logprobs"])
        self.assertEqual(reply.tokens[1]["top"][1][0], "393")
        await provider.complete("cloud", "answer", {"task": "17*23"})
        self.assertNotIn("logprobs", seen[1])

    async def test_streaming_reports_text_and_splits_thinking(self):
        chunks = [{"choices": [{"delta": {"content": "Let me check the sentence. "}}]},
                  {"choices": [{"delta": {"content": "</ifm|think_faster>"}}]},
                  {"choices": [{"delta": {"content": '{"answer": "Detroit"'}, "logprobs": {"content": [
                      {"token": "Detroit", "logprob": -0.01, "top_logprobs": []}]}}]},
                  {"choices": [{"delta": {"content": ', "confidence": 0.9}'}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 10, "completion_tokens": 7}}]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        seen, live = [], []
        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})
        reply = await self.provider(handler).complete("local", "solve", {"task": "city?"},
                                                      on_delta=lambda text, kind: live.append((kind, text)))
        self.assertTrue(seen[0]["stream"])
        self.assertEqual(len(live), 3)  # every chunk is reported as it arrives; the marker chunk carries no text
        self.assertEqual(live[0], ("thinking", "Let me check the sentence. "))  # before the marker
        self.assertEqual(live[-1][0], "output")  # after it
        self.assertFalse(any("</ifm|think" in text for _, text in live))  # the marker itself is not shown
        self.assertTrue(seen[0]["stream_options"]["include_usage"])
        self.assertEqual(reply.thinking, "Let me check the sentence.")
        self.assertEqual(reply.text, '{"answer": "Detroit", "confidence": 0.9}')
        self.assertEqual(reply.value, {"answer": "Detroit", "confidence": 0.9})
        self.assertEqual((reply.prompt_tokens, reply.tokens[0]["t"]), (10, "Detroit"))

    async def test_reasoning_effort_is_configurable(self):
        seen = []
        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        provider = OpenAICompatibleProvider(Endpoint("http://local/v1", "small", "", "", "high"),
                                            Endpoint("https://cloud/v1", "large", "k"),
                                            transport=httpx.MockTransport(handler))
        await provider.complete("local", "solve", {"task": "x"})
        await provider.complete("cloud", "answer", {"task": "x"})
        self.assertEqual(seen[0]["chat_template_kwargs"], {"reasoning_effort": "high"})
        self.assertNotIn("chat_template_kwargs", seen[1])

    async def test_streaming_truncation_is_not_a_success(self):
        body = 'data: {"choices": [{"delta": {"content": "partial"}, "finish_reason": "length"}]}\n\ndata: [DONE]\n\n'
        provider = self.provider(lambda r: httpx.Response(200, content=body.encode()))
        with self.assertRaisesRegex(RelayError, "token limit"):
            await provider.complete("local", "solve", {"task": "x"}, on_delta=lambda *args: None)

    async def test_truncation_is_not_a_success(self):
        provider = self.provider(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}))
        with self.assertRaisesRegex(RelayError, "token limit"):
            await provider.complete("local", "solve", {"task": "Hi"})

    async def test_cancellation_propagates_to_http_transport(self):
        started = asyncio.Event()
        closed = asyncio.Event()
        async def handler(request):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
        job = asyncio.create_task(self.provider(handler).complete("local", "solve", {"task": "Hi"}))
        await asyncio.wait_for(started.wait(), 1)
        job.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await job
        self.assertTrue(closed.is_set())

    async def test_locations_from_device_reports(self):
        from horizon_relay.providers import openai_compatible
        openai_compatible._devices.clear()
        devices = {"local": {"kind": "gpu", "gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 3070"}]},
                   "mid": {"kind": "gpu", "gpus": [{"index": 1, "name": "NVIDIA GeForce RTX 3090"}]}}
        def handler(request):
            host = request.url.host
            return httpx.Response(200, json=devices[host]) if request.url.path == "/device.json" else httpx.Response(404)
        provider = OpenAICompatibleProvider(Endpoint("http://local:8080/v1", "small"), Endpoint("https://api.ifm.ai/v1", "large", "k"),
                                            mid=Endpoint("http://mid:8080/v1", "medium"), transport=httpx.MockTransport(handler))
        self.assertEqual(await provider.locations(),
                         {"local": "GPU 0 · RTX 3070", "mid": "GPU 1 · RTX 3090", "cloud": "api.ifm.ai"})
        openai_compatible._devices.clear()
        devices["mid"] = devices["local"]
        self.assertEqual((await provider.locations())["mid"], "RTX 3070")  # one GPU in use: no index needed
        openai_compatible._devices.clear()
        named = OpenAICompatibleProvider(Endpoint("http://gone:1/v1", "small"), Endpoint("https://c/v1", "large", "k", "IFM API"),
                                         transport=httpx.MockTransport(lambda r: httpx.Response(500)))
        self.assertEqual(await named.locations(), {"local": "", "cloud": "IFM API"})

    def test_describe_device(self):
        from horizon_relay.providers.openai_compatible import describe_device
        two = {"kind": "gpu", "gpus": [{"index": 0, "name": "NVIDIA RTX A6000"}, {"index": 1, "name": "NVIDIA RTX A6000"}]}
        self.assertEqual(describe_device(two, True), "GPU 0+1 · 2× RTX A6000")
        self.assertEqual(describe_device({"kind": "cpu", "name": "Apple M3"}, False), "CPU")
        self.assertEqual(describe_device(None, False), "")

    async def test_real_protocol_cascade_plan_workers_synthesis(self):
        """Local answer is not trusted (low confidence), so the cloud plans, delegates and synthesizes."""
        operations, temperatures = [], []
        def handler(request):
            if request.method == "GET":
                return httpx.Response(404)
            body = json.loads(request.content)
            system = body["messages"][0]["content"]
            if "local member of a model-routing" in system:
                operations.append("solve")
                temperatures.append(body["temperature"])
                value = {"answer": "Maybe", "confidence": 0.2, "task_type": "analysis", "difficulty": "high",
                         "needs_escalation": False, "reason": "Unsure"}
            elif "strict verifier" in system:
                operations.append("critique")
                value = {"passed": False, "severity": 1.0, "critique": "Unsupported"}  # score 0.50: escalates
            else:
                payload = json.loads(body["messages"][1]["content"])
                if "results" in payload:
                    operations.append("synthesize")
                    self.assertIn("conversation", payload["sources"])
                    value = "Final answer for the actual request."
                elif "validation_error" in payload:
                    operations.append("plan")
                    self.assertEqual(payload["local_attempts"][0]["answer"], "Maybe")
                    value = {"version": 2, "final_instruction": "Answer the question", "tasks": [
                        {"id": "analyze", "instruction": "Analyze the text", "preferred_tier": "local", "why": "Easy",
                         "input_refs": ["sources.conversation"], "depends_on": [], "result_type": "task_answer", "checks": ["required_fields"]},
                        {"id": "review", "instruction": "Review the analysis", "preferred_tier": "local", "why": "Easy",
                         "input_refs": ["results.analyze"], "depends_on": ["analyze"], "result_type": "task_answer", "checks": ["required_fields"]}]}
                else:
                    operations.append("work")
                    self.assertEqual(body["model"], "small")
                    value = {"answer": "A bounded analysis of the supplied text.", "confidence": "high", "concern": ""}
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value) if isinstance(value, dict) else value}}]})
        from horizon_relay import RoutingPolicy
        result = await RelayEngine(self.provider(handler), policy=RoutingPolicy(cloud_mode="plan")).run(
            "Analyze this", {"conversation": "user: Analyze this"})
        self.assertEqual(operations, ["solve", "solve", "critique", "plan", "work", "work", "synthesize"])
        self.assertEqual(temperatures, [0.25, 0.4])
        self.assertEqual(result.metrics["route"], "cloud")
        self.assertTrue(result.metrics["verdicts"][0]["escalate"])
        self.assertIn("critic rejects", result.metrics["verdicts"][0]["reason"])
        self.assertFalse(result.metrics["simulated"])
        self.assertNotIn("SIMULATED", result.content)
