import os
import requests

import dashscope
from config.logger import setup_logging
from core.providers.tts.base import TTSProviderBase

TAG = __name__
logger = setup_logging()


class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)

        self.api_key = config.get("api_key") or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("QwenTTS 需要配置 api_key（或设置 DASHSCOPE_API_KEY 环境变量）")

        self.model_name = config.get("model_name", "qwen3-tts-flash")
        self.voice = config.get("voice", "Cherry")
        self.language_type = config.get("language_type", "Auto")
        self.output_dir = config.get("output_dir", "tmp/")
        # 合成音频采样率：qwen-audio-3.1-tts 传给服务端，qwen3-tts 由返回音频自带；
        # base 层 pydub 会统一重采样到 conn.sample_rate
        self.sample_rate = int(config.get("sample_rate", 24000))
        self.audio_file_type = config.get("format", "wav")
        # Qwen-Audio-TTS 非实时合成端点（qwen-audio-3.1-tts-* 系列）；
        # 有 workspace 专属域名时可配 base_url: https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com
        self.tts_base_url = (config.get("base_url") or "https://dashscope.aliyuncs.com").rstrip("/")

        os.makedirs(self.output_dir, exist_ok=True)

    def _qwen_tts_audio_url(self, text):
        """qwen3-tts 系列: MultiModalConversation 非流式合成，返回 audio url"""
        response = dashscope.MultiModalConversation.call(
            model=self.model_name,
            api_key=self.api_key,
            text=text,
            voice=self.voice,
            language_type=self.language_type,
        )
        if response.status_code != 200:
            raise Exception(
                f"qwen-tts 请求失败 status={response.status_code} message={response.message}"
            )

        audio = response.output.audio
        audio_url = getattr(audio, "url", None) or (
            audio.get("url") if isinstance(audio, dict) else None
        )
        if not audio_url:
            raise Exception(f"qwen-tts 响应中无 audio url: {response.output}")
        return audio_url

    def _qwen_audio_tts_audio_url(self, text):
        """qwen-audio-3.1-tts 系列: Qwen-Audio-TTS 非实时合成 HTTP API，返回 audio url"""
        resp = requests.post(
            f"{self.tts_base_url}/api/v1/services/audio/tts/SpeechSynthesizer",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model_name,
                "input": {
                    "text": text,
                    "voice": self.voice,
                    "format": self.audio_file_type,
                    "sample_rate": self.sample_rate,
                },
            },
            timeout=self.tts_timeout,
        )
        if resp.status_code != 200:
            raise Exception(
                f"qwen-audio-tts 请求失败 status={resp.status_code} body={resp.text[:300]}"
            )
        audio_url = (resp.json().get("output") or {}).get("audio", {}).get("url")
        if not audio_url:
            raise Exception(f"qwen-audio-tts 响应中无 audio url: {resp.text[:300]}")
        return audio_url

    async def text_to_speak(self, text, output_file):
        try:
            if self.model_name.startswith("qwen-audio"):
                audio_url = self._qwen_audio_tts_audio_url(text)
            else:
                audio_url = self._qwen_tts_audio_url(text)

            audio_resp = requests.get(audio_url, timeout=self.tts_timeout)
            if audio_resp.status_code != 200:
                raise Exception(
                    f"TTS 音频下载失败 status={audio_resp.status_code}"
                )

            if output_file:
                with open(output_file, "wb") as f:
                    f.write(audio_resp.content)
                return output_file
            return audio_resp.content
        except Exception as e:
            raise Exception(f"{__name__} error: {e}")
