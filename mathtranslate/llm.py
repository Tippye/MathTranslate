from openai import OpenAI

from .config import config


class Translator:
    def __init__(self, lang_to='zh', lang_from='en'):
        self.client = OpenAI(
            api_key=config.api_key,  # <--在这里将 MOONSHOT_API_KEY 替换为你从 Kimi 开放平台申请的 API Key
            base_url="https://api.moonshot.cn/v1",
            # <-- 将 base_url 从 https://api.openai.com/v1 替换为 https://api.moonshot.cn/v1
        )
        self.message = [
            {"role": "system",
             "content": f"你是一个精通论文写作的翻译官，擅长{lang_from}和{lang_to}的翻译，你可以将latex代码翻译为{lang_to}，并保持latex代码格式。专业名词和公式可以不翻译。"},
        ]

    # 翻译函数，整合预处理和后处理
    def translate(self, text, _lang_to, _lang_from):
        # 我们将用户最新的问题构造成一个 message（role=user），并添加到 messages 的尾部
        self.message.append({
            "role": "user",
            "content": text,
        })

        # 携带 messages 与 Kimi 大模型对话
        completion = self.client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=self.message,
            temperature=0.3,
        )

        # 通过 API 我们获得了 Kimi 大模型给予我们的回复消息（role=assistant）
        assistant_message = completion.choices[0].message

        # 为了让 Kimi 大模型拥有完整的记忆，我们必须将 Kimi 大模型返回给我们的消息也添加到 messages 中
        self.message.append(assistant_message)

        return assistant_message.content
