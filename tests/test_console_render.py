import json
from io import StringIO
from unittest.mock import patch

from rich.console import Console
import pytest

from system.console_render import (
    _render_tool_call,
    _render_tool_output,
    _terminal_output_text,
    render_content_assistant_message,
    render_model_markdown,
)
from system.stream_render import StreamRenderer
from system.tui_app import TuiRegion
from utils.llm_client import (
    AsyncChatAPIClient,
    build_anthropic_request_messages,
    build_llm_result,
    sanitize_openai_messages,
)


def _render_tool_event(render, value):
    posted = []
    with patch(
        "system.console_render.post_tui",
        side_effect=lambda region, payload=None, **kwargs: posted.append(payload),
    ):
        render(value)

    panel = next(payload for payload in posted if payload is not None)
    buffer = StringIO()
    Console(file=buffer, force_terminal=False, width=120, color_system=None).print(panel.renderable)
    return buffer.getvalue()


def _render_terminal_output(output):
    return _render_tool_event(
        lambda value: _render_tool_output("RunTerminalCommand", value, TuiRegion.BACKGROUND),
        output,
    )


def _render_tool_arguments(arguments):
    return _render_tool_event(
        lambda value: _render_tool_call("FileEdit", value, TuiRegion.BACKGROUND),
        arguments,
    )


def test_tool_output_does_not_emit_terminal_screen_control_sequences():
    rendered = _render_terminal_output(
        "normal-before\x1b[31mRED\x1b[0m\x1b[2J\x1b[10;10Hnormal-after"
    )

    assert "normal-before" in rendered
    assert "normal-after" in rendered
    assert "\x1b[2J" not in rendered
    assert "\x1b[10;10H" not in rendered


def test_structured_tool_output_does_not_emit_terminal_screen_control_sequences():
    rendered = _render_terminal_output({"output": "before\x1b[2J\x1b[10;10Hafter"})

    assert "before" in rendered
    assert "after" in rendered
    assert "\x1b[2J" not in rendered
    assert "\x1b[10;10H" not in rendered


def test_structured_tool_arguments_use_json_layout_and_expand_multiline_strings():
    rendered = _render_tool_arguments({
        "edits": [{
            "search_content": "old line\n\nold code",
            "replace_content": "new line\n\nnew code",
        }],
        "count": 3,
        "enabled": True,
    })

    assert '"edits": [' in rendered
    assert '"search_content":\n        "\n        old line\n        \n        old code\n        "' in rendered
    assert "old line\\n\\nold code" not in rendered
    assert '"count": 3' in rendered
    assert '"enabled": true' in rendered
    assert "❖" not in rendered
    assert "[Item 1]" not in rendered


def test_json_string_tool_arguments_expand_real_newlines_but_preserve_literal_escapes():
    rendered = _render_tool_arguments(
        '{"content":"first line\\n\\nsecond line","literal":"first\\\\nsecond"}'
    )

    assert '"content":\n    "\n    first line\n    \n    second line\n    "' in rendered
    assert "first line\\n\\nsecond line" not in rendered
    assert '"literal": "first\\nsecond"' in rendered


def test_tool_arguments_render_code_special_characters_without_escaping():
    code = '\t\tusername, _, err := auth.RequireLogin(c.String("username"))'
    rendered = _render_tool_arguments({"replace_content": code})

    assert 'username, _, err := auth.RequireLogin(c.String("username"))' in rendered
    assert "\\t" not in rendered
    assert '\\"username\\"' not in rendered


def test_structured_tool_output_uses_json_layout_and_expands_multiline_strings():
    rendered = _render_terminal_output(
        '{"summary":"first line\\n\\nsecond line","details":{"code":"def run():\\n    return 1"}}'
    )

    assert '"summary":\n    "\n    first line\n    \n    second line\n    "' in rendered
    assert "first line\\n\\nsecond line" not in rendered
    assert '"details": {' in rendered
    assert '"code":\n      "\n      def run():\n          return 1\n      "' in rendered
    assert "❖" not in rendered


def test_terminal_output_preserves_text_before_carriage_return_control_sequences():
    rendered = _terminal_output_text(
        "session manager\r\x1b[2K\nstarted\r\x1b[2K\nshutting down"
    ).plain

    assert rendered == "session manager\nstarted\nshutting down"


def test_plain_tool_output_remains_plain_text():
    rendered = _render_terminal_output("first line\nsecond line")

    assert "first line\n" in rendered
    assert "second line" in rendered
    assert '"first line' not in rendered


def test_json_scalar_like_tool_output_remains_original_text():
    assert _render_terminal_output("true").strip() == "true"
    assert _render_terminal_output("123").strip() == "123"
    assert _render_terminal_output('"quoted"').strip() == '"quoted"'


def test_live_display_formatting_does_not_change_tool_result_sent_to_model():
    nested_json = json.dumps(
        {"content": "first line\nsecond line", "literal": r"first\nsecond"},
        ensure_ascii=False,
    )
    output = {"payload": nested_json, "count": 2}
    expected_model_content = json.dumps(output, ensure_ascii=False)

    rendered = _render_terminal_output(output)
    tool_message = AsyncChatAPIClient(None, "test").format_tool_result(
        "call_1",
        "RunTerminalCommand",
        output,
    )

    assert '"payload": {' in rendered
    assert '\\"content\\"' not in rendered
    assert output == {"payload": nested_json, "count": 2}
    assert tool_message["content"] == expected_model_content
    assert sanitize_openai_messages([tool_message])[0]["content"] == expected_model_content

    _, anthropic_messages = build_anthropic_request_messages([tool_message])
    assert anthropic_messages[0]["content"][0]["content"] == expected_model_content


def test_live_display_expands_double_encoded_json_without_changing_model_content():
    structured = json.dumps(
        {"summary": "first line\nsecond line", "items": [1, 2]},
        ensure_ascii=False,
    )
    output = json.dumps(structured, ensure_ascii=False)

    rendered = _render_terminal_output(output)
    tool_message = AsyncChatAPIClient(None, "test").format_tool_result(
        "call_1",
        "RunTerminalCommand",
        output,
    )

    assert '"summary":\n    "\n    first line\n    second line\n    "' in rendered
    assert '\\"summary\\"' not in rendered
    assert tool_message["content"] == output


def _render_to_plain_text(renderable):
    buffer = StringIO()
    Console(file=buffer, width=100).print(renderable)
    return buffer.getvalue()


def test_model_markdown_displays_html_like_tags_as_text():
    rendered = _render_to_plain_text(render_model_markdown("<xxxx> aaaa </xxxx>"))

    assert "<xxxx> aaaa </xxxx>" in rendered


def test_assistant_panel_displays_tags_and_keeps_markdown_formatting():
    rendered = _render_to_plain_text(
        render_content_assistant_message("**正文** <thinking>内容</thinking>")
    )

    assert "正文" in rendered
    assert "<thinking>内容</thinking>" in rendered


@pytest.mark.anyio
async def test_stream_error_releases_tool_calls_background_active_state():
    events = []

    async def failing_stream():
        yield {"type": "tool_calls", "content": None}
        raise RuntimeError("stream interrupted")

    with patch(
        "system.stream_render.post_tui",
        side_effect=lambda region, payload=None, **kwargs: events.append((region, payload, kwargs)),
    ), patch("system.stream_render.is_cancelled", return_value=False):
        with pytest.raises(RuntimeError, match="stream interrupted"):
            await StreamRenderer().render_async(failing_stream())

    background_active = [
        kwargs["active"]
        for region, _, kwargs in events
        if region == TuiRegion.BACKGROUND and "active" in kwargs
    ]
    assert background_active == [True, False]


@pytest.mark.anyio
async def test_stream_error_releases_reasoning_and_content_active_state():
    events = []

    async def failing_stream():
        yield {"type": "reasoning", "content": "思考中"}
        yield {"type": "text", "content": "正文"}
        raise RuntimeError("stream interrupted")

    with patch(
        "system.stream_render.post_tui",
        side_effect=lambda region, payload=None, **kwargs: events.append((region, payload, kwargs)),
    ), patch("system.stream_render.is_cancelled", return_value=False):
        with pytest.raises(RuntimeError, match="stream interrupted"):
            await StreamRenderer().render_async(failing_stream())

    active_by_region = {
        region: [
            kwargs["active"]
            for event_region, _, kwargs in events
            if event_region == region and "active" in kwargs
        ]
        for region in (TuiRegion.REASONING, TuiRegion.CONTENT)
    }
    assert active_by_region == {
        TuiRegion.REASONING: [True, False],
        TuiRegion.CONTENT: [True, False],
    }


@pytest.mark.anyio
async def test_cancelled_stream_releases_tool_calls_background_active_state():
    events = []

    async def cancelled_stream():
        yield {"type": "tool_calls", "content": None}
        yield {"type": "text", "content": "不会继续渲染"}

    with patch(
        "system.stream_render.post_tui",
        side_effect=lambda region, payload=None, **kwargs: events.append((region, payload, kwargs)),
    ), patch(
        "system.stream_render.is_cancelled",
        side_effect=[False, True, True],
    ):
        text_content, tool_calls, raw_message = await StreamRenderer().render_async(cancelled_stream())

    background_active = [
        kwargs["active"]
        for region, _, kwargs in events
        if region == TuiRegion.BACKGROUND and "active" in kwargs
    ]
    assert background_active == [True, False]
    assert (text_content, tool_calls, raw_message) == ("", [], None)


@pytest.mark.anyio
async def test_async_stream_renderer_consumes_unified_done_result():
    result = build_llm_result(
        text="async answer",
        reasoning="reason",
        source_format="openai_chat",
        source_model="test-model",
        stop_reason="stop",
    )

    async def stream():
        yield {"type": "reasoning", "content": "reason"}
        yield {"type": "text", "content": "async answer"}
        yield {"type": "done", "result": result}

    with patch("system.stream_render.post_tui"), patch(
        "system.stream_render.is_cancelled", return_value=False
    ):
        rendered = await StreamRenderer().render_async(stream())

    assert rendered == ("async answer", [], result.assistant_message)


@pytest.mark.anyio
async def test_stream_renderer_posts_reasoning_status_only_once_per_response():
    async def stream():
        for _ in range(100):
            yield {"type": "reasoning", "content": "reasoning chunk "}
        yield {"type": "text", "content": "answer"}

    renderer = StreamRenderer()
    with patch("system.stream_render.post_tui") as post, patch(
        "system.stream_render.is_cancelled", return_value=False
    ):
        for _ in range(2):
            await renderer.render_async(stream())

    statuses = [
        call.args[1]
        for call in post.call_args_list
        if call.args[0] == TuiRegion.STATUS
    ]
    assert statuses.count("Orchestrator reasoning") == 2


def _record_stream_events(events):
    return patch(
        "system.stream_render.post_tui",
        side_effect=lambda region, payload=None, **kwargs: events.append((region, payload, kwargs)),
    )


def test_committed_paragraph_carries_the_remaining_tail_in_a_single_event():
    renderer = StreamRenderer()
    events = []

    with _record_stream_events(events):
        remaining, emitted = renderer._process_block_commit("第一段\n\n第二段尾巴", "第一段\n\n第二段尾巴")
        renderer._update_tail(remaining, force=True)

    assert (remaining, emitted) == ("第二段尾巴", "第一段\n\n")
    assert len(events) == 1  # 不再先清尾部；提交后的强制刷新也不重复投递相同尾部
    region, payload, kwargs = events[0]
    assert region == TuiRegion.CONTENT
    assert payload.copy_text == "第一段\n\n"
    assert kwargs["tail_after"].plain == "第二段尾巴"
    assert not kwargs.get("tail")


def test_update_tail_skips_text_the_ui_already_shows_until_the_tail_is_cleared():
    renderer = StreamRenderer()
    events = []

    with _record_stream_events(events):
        renderer._update_tail("同一段尾巴", force=True)
        renderer._update_tail("同一段尾巴", force=True)
        renderer._clear_tail()
        renderer._update_tail("同一段尾巴", force=True)

    tails = [getattr(payload, "plain", payload) for _, payload, kwargs in events if kwargs.get("tail")]
    assert tails == ["同一段尾巴", "", "同一段尾巴"]


def test_content_and_reasoning_share_one_tail_slot_for_skipping_duplicates():
    renderer = StreamRenderer()
    events = []

    with _record_stream_events(events):
        renderer._update_tail("正文尾巴", region=TuiRegion.CONTENT, force=True)
        renderer._update_tail("思考尾巴", region=TuiRegion.REASONING, force=True)
        # 两个区域共用同一个 UI 尾部控件：此时控件里是“思考尾巴”，正文尾巴必须重新投递
        renderer._update_tail("正文尾巴", region=TuiRegion.CONTENT, force=True)

    assert [getattr(payload, "plain", payload) for _, payload, _ in events] == ["正文尾巴", "思考尾巴", "正文尾巴"]


def test_commit_marks_the_tail_as_just_updated_for_throttling():
    renderer = StreamRenderer()
    events = []

    with _record_stream_events(events):
        remaining, _ = renderer._process_block_commit("第一段\n\n第二段尾巴", "第一段\n\n第二段尾巴")
        renderer._update_tail(remaining + "继续", force=False)  # 刚随提交更新过尾部，0.3 秒内不再刷新
        renderer._update_tail(remaining + "继续", force=True)

    assert [bool(kwargs.get("tail")) for _, _, kwargs in events] == [False, True]
