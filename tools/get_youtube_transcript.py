import re
from langchain_core.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi


@tool
def get_youtube_transcript(url: str) -> str:
    """
    当问题中给出了 YouTube 视频链接（如 https://www.youtube.com/watch?v=...）时使用。
    该工具能免费提取视频的完整英文字幕或自动生成的字幕文本，帮助你分析视频内容、计数或寻找特定事件。
    """
    try:
        video_id_match = re.search(r'(?:v=|youtu\.be/|\/)([0-9A-Za-z_-]{11})', url)
        if not video_id_match:
            return "NO_TRANSCRIPT_AVAILABLE: unable to parse a valid YouTube video ID from the URL."

        video_id = video_id_match.group(1)
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=['en', 'zh-CN', 'zh'])

        full_text = []
        for snippet in transcript:
            if snippet.text:
                full_text.append(snippet.text)

        if not full_text:
            return "NO_TRANSCRIPT_AVAILABLE: transcript object was returned, but it contained no readable text."

        return " ".join(full_text)

    except Exception as e:
        return f"NO_TRANSCRIPT_AVAILABLE: failed to fetch transcript. {str(e)}"
