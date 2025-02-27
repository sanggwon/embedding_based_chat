# -*- coding: utf-8 -*-

import os
import re
import glob
import uuid
import subprocess
import pandas as pd
import openai
import tiktoken
from scipy.spatial.distance import cosine
import ast

##############################
# 1. 유튜브 검색 및 자막 다운로드/CSV 저장
##############################

def load_api_key(file_path="OPENAI_API_KEY") -> str:
    """OPENAI_API_KEY 파일에서 API 키를 읽어 반환합니다."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def search_youtube_videos(keyword: str, num_results: int = 10) -> list:
    """
    yt-dlp를 이용해 검색어에 해당하는 영상 10건의 URL을 반환합니다.
    ytsearchN: 구문을 사용합니다.
    """
    search_query = f"ytsearch{num_results}:{keyword}"
    ydl_opts = {
        'quiet': True,
        'skip_download': True
    }
    import yt_dlp
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(search_query, download=False)
        videos = result.get('entries', [])
        urls = [video.get('webpage_url') for video in videos if video]
    return urls

def download_subtitles(video_url, lang: str = "ko") -> str:
    """
    yt_dlp를 이용해 유튜브 동영상에서 한국어 자막을 다운로드하고,
    생성된 .vtt 또는 .srt 파일의 절대 경로를 반환합니다.
    자막이 없으면 None을 반환합니다.
    """
    output_path = "./downloads"
    os.makedirs(output_path, exist_ok=True)
    # 언어 코드를 파일명에 포함하도록 출력 템플릿 설정
    output_template = os.path.join(output_path, f'downloaded_subs.{lang}.%(ext)s')
    ydl_opts = {
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': [lang],
        'outtmpl': output_template,
        'format': 'bestaudio/best',
        'quiet': False
    }
    import yt_dlp
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        subtitles_info = info_dict.get('requested_subtitles', {})
        if not subtitles_info or lang not in subtitles_info:
            print(f"[{video_url}] 해당 영상에 지정한 언어의 자막이 없습니다.")
            return None
        ydl.download([video_url])
    
    # 다운로드 후 생성된 파일을 glob로 찾음
    subtitle_files = glob.glob(os.path.join(output_path, "downloaded_subs*{}.vtt".format(lang)))
    if not subtitle_files:
        print(f"[{video_url}] 다운로드된 자막 파일을 찾을 수 없습니다.")
        return None
    subtitle_file = subtitle_files[0]
    subtitle_file = os.path.abspath(subtitle_file)
    
    # 파일명이 중복되어 'ko.ko.vtt'인 경우 수정
    if "ko.ko.vtt" in subtitle_file:
        fixed_subtitle_file = subtitle_file.replace("ko.ko.vtt", "ko.vtt")
        if os.path.exists(fixed_subtitle_file):
            os.remove(fixed_subtitle_file)
        os.rename(subtitle_file, fixed_subtitle_file)
        subtitle_file = fixed_subtitle_file
        print("파일명이 중복되어 수정되었습니다:", subtitle_file)
    return subtitle_file

def clean_subtitle_file(subtitle_file):
    """
    VTT/SRT 자막 파일에서 시간 코드, 태그 등 불필요한 정보를 제거하여 하나의 문자열로 반환합니다.
    """
    seen_lines = set()
    lines = []
    with open(subtitle_file, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            line = re.sub(r"<.*?>", "", line)
            if (line and not line.isdigit() and '-->' not in line 
                    and not line.startswith("WEBVTT")
                    and not line.startswith("Kind:")
                    and not line.startswith("Language:")):
                if line not in seen_lines:
                    seen_lines.add(line)
                    lines.append(line)
    cleaned_text = " ".join(lines)
    return cleaned_text

def save_subtitles_to_csv(data, csv_path="youtube_subtitles.csv"):
    """
    data: 리스트의 딕셔너리, 각 항목은 {"url": ..., "subtitles": ...}
    CSV 파일에는 "url"과 "subtitles" 열이 있으며, 기존 파일이 있으면 새로운 행을 추가합니다.
    """
    df_new = pd.DataFrame(data)
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_path, index=False)
    else:
        df_new.to_csv(csv_path, index=False)
    print(f"자막이 '{csv_path}' 파일에 저장되었습니다.")

##############################
# 2. 임베딩 기반 질의응답 모듈 (메모리 내 처리)
##############################

def get_embedding(text: str, model: str = "text-embedding-ada-002") -> list:
    """
    주어진 텍스트의 임베딩을 생성합니다.
    """
    response = openai.Embedding.create(
        model=model,
        input=text
    )
    return response["data"][0]["embedding"]

def split_text(text: str, max_tokens: int = 500, model: str = "text-embedding-ada-002") -> list:
    """
    tiktoken 라이브러리를 사용하여 텍스트를 max_tokens 이하의 청크로 분할합니다.
    """
    encoding = tiktoken.encoding_for_model(model)
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0
    for word in words:
        word_tokens = len(encoding.encode(word))
        if current_tokens + word_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_tokens = word_tokens
        else:
            current_chunk.append(word)
            current_tokens += word_tokens
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def prepare_subtitle_embeddings_from_text(subtitles_text: str, model: str = "text-embedding-ada-002") -> pd.DataFrame:
    """
    자막 텍스트를 적당한 길이로 분할한 후, 각 청크에 대해 임베딩을 계산하여
    "chunk"와 "embedding" 열을 가진 DataFrame을 반환합니다.
    """
    chunks = split_text(subtitles_text, max_tokens=500, model=model)
    records = []
    for chunk in chunks:
        embedding = get_embedding(chunk, model=model)
        records.append({"chunk": chunk, "embedding": embedding})
    return pd.DataFrame(records)

def rank_chunks(query: str, df_chunks: pd.DataFrame, model: str = "text-embedding-ada-002") -> pd.DataFrame:
    """
    질문 임베딩과 각 청크 임베딩 간의 코사인 유사도를 계산하여,
    관련성이 높은 순으로 정렬한 DataFrame을 반환합니다.
    """
    query_embedding = get_embedding(query, model=model)
    similarities = []
    for idx, row in df_chunks.iterrows():
        sim = 1 - cosine(query_embedding, row["embedding"])
        similarities.append(sim)
    df_chunks["similarity"] = similarities
    df_sorted = df_chunks.sort_values("similarity", ascending=False)
    return df_sorted

def answer_query(query: str, df_chunks: pd.DataFrame, top_n: int = 3, chat_model: str = "gpt-3.5-turbo") -> str:
    """
    질문을 받아 임베딩 기반 검색을 통해 관련 자막 청크를 찾고,
    해당 청크들을 참고하여 GPT를 통해 최종 답변을 생성합니다.
    """
    ranked = rank_chunks(query, df_chunks)
    top_chunks = ranked.head(top_n)["chunk"].tolist()
    prompt = "다음 자막 내용을 참고하여 질문에 답해주세요.\n\n"
    for chunk in top_chunks:
        prompt += f"자막 내용:\n{chunk}\n\n"
    prompt += f"질문: {query}"
    response = openai.ChatCompletion.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": "당신은 자막 기반 질의 응답 시스템입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=500
    )
    return response["choices"][0]["message"]["content"].strip()

##############################
# 3. 메인 실행부
##############################

def main():
    # OpenAI API 키 설정
    openai.api_key = load_api_key("OPENAI_API_KEY")
    
    # 검색어 입력: 유튜브에서 특정 단어로 검색하여 10건의 영상 URL을 가져옴
    search_keyword = input("유튜브 검색어를 입력하세요 (예: '테드 샌더스'): ").strip()
    if not search_keyword:
        print("검색어가 입력되지 않았습니다.")
        return
    
    # yt-dlp를 사용해 검색 결과 10건의 URL을 가져오는 함수 사용
    def search_youtube_videos(keyword: str, num_results: int = 10) -> list:
        search_query = f"ytsearch{num_results}:{keyword}"
        ydl_opts = {
            'quiet': True,
            'skip_download': True
        }
        import yt_dlp
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(search_query, download=False)
            videos = result.get('entries', [])
            urls = [video.get('webpage_url') for video in videos if video]
        return urls

    video_urls = search_youtube_videos(search_keyword, num_results=10)
    if not video_urls:
        print("검색 결과가 없습니다.")
        return
    print("검색된 URL:")
    for url in video_urls:
        print(url)
    
    # 각 영상마다 자막 다운로드 및 클린업
    all_data = []
    for url in video_urls:
        print(f"\n[{url}] 자막 다운로드 시작...")
        subtitle_file = download_subtitles(url, lang="ko")
        if subtitle_file is None:
            print(f"[{url}] 자막 다운로드 실패 또는 자막이 없습니다.")
            continue
        print(f"[{url}] 자막 다운로드 완료:", subtitle_file)
        subtitles_text = clean_subtitle_file(subtitle_file)
        print(f"[{url}] 정제된 자막 길이: {len(subtitles_text)} chars")
        all_data.append({"url": url, "subtitles": subtitles_text})
    
    # CSV 저장 (모든 영상의 자막 데이터 저장)
    csv_path = "youtube_subtitles.csv"
    df_all = pd.DataFrame(all_data)
    df_all.to_csv(csv_path, index=False)
    print(f"\n전체 자막 데이터가 '{csv_path}' 파일에 저장되었습니다.")
    
    # CSV 파일에서 자막 데이터 로드 및 임베딩 준비 (모든 영상의 자막을 하나로 결합)
    df_subtitles = pd.read_csv(csv_path)
    combined_subtitles = " ".join(df_subtitles["subtitles"].tolist())
    print(f"결합된 자막 길이: {len(combined_subtitles)} chars")
    
    df_chunks = prepare_subtitle_embeddings_from_text(combined_subtitles)
    print("임베딩 계산 완료.")
    
    # 질의응답 루프
    while True:
        query = input("질문을 입력하세요 (종료: exit): ").strip()
        if query.lower() == "exit":
            break
        answer = answer_query(query, df_chunks)
        print("답변:", answer)

if __name__ == "__main__":
    main()
