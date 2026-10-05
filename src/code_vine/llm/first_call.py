"""第一次真实调用。运行: uv run python -m code_vine.llm.first_call"""
from code_vine.core.config import get_settings
from code_vine.llm.client import get_client


def main() -> None:
    settings = get_settings()
    client = get_client()
    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        messages=[{"role": "user", "content": "用一句话解释什么是无状态的 HTTP API。"}],
    )

    print("stop_reason =", response.stop_reason)
    print("usage: input =", response.usage.input_tokens, " output =", response.usage.output_tokens)

    # content 是块列表 —— 按 type 分支，绝不整体 print 当字符串
    for content in response.content:
        if content.type == "text":
            print("text> ", content.text)
        elif content.type == "thinking":
            print("thinking> ", content.thinking)
        else:
            print("unknown content:", content.type)


if __name__ == "__main__":
    main()
