"""
ASS 字幕處理工具模組

包含 ASS 標籤處理、字幕合併等共用功能
"""

import re
from collections import defaultdict
from typing import List, Optional, Tuple

import pysubs2


def extract_text_and_tags(text: str) -> Tuple[str, List[str]]:
    """
    從 ASS 字幕文字中提取純文字和標籤

    Args:
        text: ASS 字幕文字

    Returns:
        (純文字, 標籤列表)
    """
    tags = re.findall(r"{.*?}", text)
    text_only = re.sub(r"{.*?}", "", text)
    return text_only, tags


def restore_tags(translated: str, tags: List[str]) -> str:
    """
    將標籤恢復到翻譯後的文字

    Args:
        translated: 翻譯後的文字
        tags: 原始標籤列表

    Returns:
        恢復標籤後的文字
    """
    return "".join(tags) + translated


def group_by_time(subs: pysubs2.SSAFile) -> dict:
    """
    按時間分組字幕事件

    Args:
        subs: ASS 字幕檔案

    Returns:
        按 (start, end) 分組的字幕字典
    """
    groups = defaultdict(list)
    for line in subs:
        key = (line.start, line.end)
        groups[key].append(line)
    return groups


def parse_pos(text: str) -> Tuple[Optional[int], Optional[int]]:
    """
    解析 ASS 文字中的位置標籤

    Args:
        text: ASS 字幕文字

    Returns:
        (x座標, y座標) 或 (None, None)
    """
    match = re.search(r"\\pos\(([+-]?\d+(?:\.\d+)?),([+-]?\d+(?:\.\d+)?)\)", text)
    if match:
        x = float(match.group(1))
        y = float(match.group(2))
        return int(x), int(y)  # 轉換為整數
    return None, None


def setup_default_style(subs: pysubs2.SSAFile) -> None:
    """
    設置 ASS 字幕檔案的預設樣式

    Args:
        subs: ASS 字幕檔案
    """
    default_style = pysubs2.SSAStyle()
    default_style.alignment = 5  # 中心對齊
    default_style.marginl = 0
    default_style.marginr = 0
    default_style.marginv = 0
    default_style.borderstyle = 1
    default_style.outline = 0
    default_style.shadow = 0
    default_style.margin_t = 0
    default_style.margin_b = 0
    subs.styles["Default"] = default_style


def merge_continuous_events(
    subs: pysubs2.SSAFile, position_tolerance: int = 10, time_gap_threshold: int = 500
) -> int:
    """
    合併連續的相同文字事件

    Args:
        subs: ASS 字幕檔案
        position_tolerance: 位置容差（像素）
        time_gap_threshold: 時間間隙閾值（毫秒）

    Returns:
        合併後的事件數量
    """
    if not subs.events:
        return 0

    # 按時間排序
    subs.events.sort(key=lambda x: x.start)

    merged_events = []
    current_event = subs.events[0].copy()

    for next_event in subs.events[1:]:
        # 檢查是否為相同文字且時間連續
        if (
            current_event.text == next_event.text
            and current_event.end >= next_event.start - time_gap_threshold
        ):
            # 延長當前事件的結束時間
            current_event.end = max(current_event.end, next_event.end)
        else:
            # 添加當前事件並開始新事件
            merged_events.append(current_event)
            current_event = next_event.copy()

    # 添加最後一個事件
    merged_events.append(current_event)

    # 更新字幕檔案的事件
    subs.events = merged_events
    return len(subs.events)


def merge_ass_subs_advanced(
    subs: pysubs2.SSAFile, position_tolerance: int = 10, time_gap_threshold: int = 500
) -> int:
    """
    進階字幕合併，考慮位置和內容相似性

    Args:
        subs: ASS 字幕檔案
        position_tolerance: 位置容差（像素）
        time_gap_threshold: 時間間隙閾值（毫秒）

    Returns:
        合併後的事件數量
    """
    if not subs.events:
        return 0

    merged_events = []
    events = sorted(subs.events, key=lambda e: e.start)  # 按時間排序

    current = events[0]

    for next_event in events[1:]:
        # 內容比對（去除 pos tag）
        cur_pos = parse_pos(current.text)
        next_pos = parse_pos(next_event.text)
        cur_text_only, _ = extract_text_and_tags(current.text)
        next_text_only, _ = extract_text_and_tags(next_event.text)

        # 位置相近
        pos_close = (
            cur_pos[0] is not None
            and next_pos[0] is not None
            and abs(cur_pos[0] - next_pos[0]) <= position_tolerance
            and abs(cur_pos[1] - next_pos[1]) <= position_tolerance
        )

        # 內容相同（比較純文字）
        text_same = cur_text_only == next_text_only

        # 時間連續
        time_gap = next_event.start - current.end
        time_close = 0 <= time_gap <= time_gap_threshold

        if pos_close and text_same and time_close:
            # 合併：延長 current 的 end
            current.end = next_event.end
        else:
            merged_events.append(current)
            current = next_event

    merged_events.append(current)
    subs.events = merged_events
    subs.sort()

    return len(merged_events)
