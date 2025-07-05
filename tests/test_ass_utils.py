"""
ASS 工具模組測試

測試 ASS 字幕處理功能
"""

import pysubs2
import pytest

from macos_video_auto_ocr_ass.ass_utils import (
    extract_text_and_tags,
    group_by_time,
    merge_ass_subs_advanced,
    merge_continuous_events,
    parse_pos,
    restore_tags,
    setup_default_style,
)


class TestExtractTextAndTags:
    """測試文字和標籤提取"""

    def test_extract_text_and_tags_simple(self):
        """測試簡單文字提取"""
        text = "Hello World"
        text_only, tags = extract_text_and_tags(text)
        assert text_only == "Hello World"
        assert tags == []

    def test_extract_text_and_tags_with_pos(self):
        """測試帶位置標籤的文字提取"""
        text = "{\\pos(100,200)}Hello World"
        text_only, tags = extract_text_and_tags(text)
        assert text_only == "Hello World"
        assert tags == ["{\\pos(100,200)}"]

    def test_extract_text_and_tags_multiple(self):
        """測試多個標籤的文字提取"""
        text = "{\\pos(100,200)\\fs24\\b1}Hello World"
        text_only, tags = extract_text_and_tags(text)
        assert text_only == "Hello World"
        assert len(tags) == 1
        assert "{\\pos(100,200)\\fs24\\b1}" in tags[0]

    def test_extract_text_and_tags_nested(self):
        """測試嵌套標籤的文字提取"""
        text = "{\\pos(100,200)}Hello{\\fs24}World"
        text_only, tags = extract_text_and_tags(text)
        assert text_only == "HelloWorld"
        assert len(tags) == 2
        assert "{\\pos(100,200)}" in tags
        assert "{\\fs24}" in tags

    def test_extract_text_and_tags_empty(self):
        """測試空文字提取"""
        text = ""
        text_only, tags = extract_text_and_tags(text)
        assert text_only == ""
        assert tags == []

    def test_extract_text_and_tags_only_tags(self):
        """測試只有標籤的文字提取"""
        text = "{\\pos(100,200)}"
        text_only, tags = extract_text_and_tags(text)
        assert text_only == ""
        assert tags == ["{\\pos(100,200)}"]


class TestRestoreTags:
    """測試標籤恢復"""

    def test_restore_tags_simple(self):
        """測試簡單標籤恢復"""
        translated = "你好世界"
        tags = ["{\\pos(100,200)}"]
        result = restore_tags(translated, tags)
        assert result == "{\\pos(100,200)}你好世界"

    def test_restore_tags_multiple(self):
        """測試多個標籤恢復"""
        translated = "你好世界"
        tags = ["{\\pos(100,200)}", "{\\fs24}"]
        result = restore_tags(translated, tags)
        assert result == "{\\pos(100,200)}{\\fs24}你好世界"

    def test_restore_tags_empty_tags(self):
        """測試空標籤恢復"""
        translated = "你好世界"
        tags = []
        result = restore_tags(translated, tags)
        assert result == "你好世界"

    def test_restore_tags_empty_text(self):
        """測試空文字恢復"""
        translated = ""
        tags = ["{\\pos(100,200)}"]
        result = restore_tags(translated, tags)
        assert result == "{\\pos(100,200)}"


class TestGroupByTime:
    """測試按時間分組"""

    def test_group_by_time_empty(self):
        """測試空字幕檔案分組"""
        subs = pysubs2.SSAFile()
        groups = group_by_time(subs)
        assert groups == {}

    def test_group_by_time_single(self):
        """測試單個字幕分組"""
        subs = pysubs2.SSAFile()
        event = pysubs2.SSAEvent(start=1000, end=3000, text="Hello")
        subs.events.append(event)

        groups = group_by_time(subs)
        assert len(groups) == 1
        assert (1000, 3000) in groups
        assert len(groups[(1000, 3000)]) == 1
        assert groups[(1000, 3000)][0].text == "Hello"

    def test_group_by_time_multiple_same_time(self):
        """測試相同時間的多個字幕分組"""
        subs = pysubs2.SSAFile()
        event1 = pysubs2.SSAEvent(start=1000, end=3000, text="Hello")
        event2 = pysubs2.SSAEvent(start=1000, end=3000, text="World")
        subs.events.extend([event1, event2])

        groups = group_by_time(subs)
        assert len(groups) == 1
        assert (1000, 3000) in groups
        assert len(groups[(1000, 3000)]) == 2
        assert groups[(1000, 3000)][0].text == "Hello"
        assert groups[(1000, 3000)][1].text == "World"

    def test_group_by_time_different_times(self):
        """測試不同時間的字幕分組"""
        subs = pysubs2.SSAFile()
        event1 = pysubs2.SSAEvent(start=1000, end=3000, text="Hello")
        event2 = pysubs2.SSAEvent(start=3000, end=5000, text="World")
        subs.events.extend([event1, event2])

        groups = group_by_time(subs)
        assert len(groups) == 2
        assert (1000, 3000) in groups
        assert (3000, 5000) in groups
        assert len(groups[(1000, 3000)]) == 1
        assert len(groups[(3000, 5000)]) == 1


class TestParsePos:
    """測試位置標籤解析"""

    def test_parse_pos_valid(self):
        """測試有效位置標籤解析"""
        text = "{\\pos(100,200)}Hello"
        x, y = parse_pos(text)
        assert x == 100
        assert y == 200

    def test_parse_pos_no_pos(self):
        """測試無位置標籤解析"""
        text = "Hello World"
        x, y = parse_pos(text)
        assert x is None
        assert y is None

    def test_parse_pos_invalid_format(self):
        """測試無效格式位置標籤解析"""
        text = "{\\pos(100)}Hello"  # 缺少 y 座標
        x, y = parse_pos(text)
        assert x is None
        assert y is None

    def test_parse_pos_negative_coordinates(self):
        """測試負座標位置標籤解析"""
        text = "{\\pos(-100,-200)}Hello"
        x, y = parse_pos(text)
        assert x == -100
        assert y == -200

    def test_parse_pos_decimal_coordinates(self):
        """測試小數座標位置標籤解析"""
        text = "{\\pos(100.5,200.7)}Hello"
        x, y = parse_pos(text)
        assert x == 100
        assert y == 200  # 應該被轉換為整數


class TestSetupDefaultStyle:
    """測試預設樣式設置"""

    def test_setup_default_style(self):
        """測試預設樣式設置"""
        subs = pysubs2.SSAFile()
        setup_default_style(subs)

        assert "Default" in subs.styles
        style = subs.styles["Default"]
        assert style.alignment == 5
        assert style.marginl == 0
        assert style.marginr == 0
        assert style.marginv == 0
        assert style.borderstyle == 1
        assert style.outline == 0
        assert style.shadow == 0
        # 已移除 margin_t 和 margin_b 的檢查


class TestMergeContinuousEvents:
    """測試連續事件合併"""

    def test_merge_continuous_events_empty(self):
        """測試空事件合併"""
        subs = pysubs2.SSAFile()
        result = merge_continuous_events(subs)
        assert result == 0
        assert len(subs.events) == 0

    def test_merge_continuous_events_single(self):
        """測試單個事件合併"""
        subs = pysubs2.SSAFile()
        event = pysubs2.SSAEvent(start=1000, end=3000, text="Hello")
        subs.events.append(event)

        result = merge_continuous_events(subs)
        assert result == 1
        assert len(subs.events) == 1
        assert subs.events[0].text == "Hello"

    def test_merge_continuous_events_continuous(self):
        """測試連續事件合併"""
        subs = pysubs2.SSAFile()
        event1 = pysubs2.SSAEvent(start=1000, end=3000, text="Hello")
        event2 = pysubs2.SSAEvent(start=3000, end=5000, text="Hello")
        subs.events.extend([event1, event2])

        result = merge_continuous_events(subs)
        assert result == 1
        assert len(subs.events) == 1
        assert subs.events[0].start == 1000
        assert subs.events[0].end == 5000
        assert subs.events[0].text == "Hello"

    def test_merge_continuous_events_different_text(self):
        """測試不同文字的事件合併"""
        subs = pysubs2.SSAFile()
        event1 = pysubs2.SSAEvent(start=1000, end=3000, text="Hello")
        event2 = pysubs2.SSAEvent(start=3000, end=5000, text="World")
        subs.events.extend([event1, event2])

        result = merge_continuous_events(subs)
        assert result == 2
        assert len(subs.events) == 2
        assert subs.events[0].text == "Hello"
        assert subs.events[1].text == "World"

    def test_merge_continuous_events_with_gap(self):
        """測試有間隙的事件合併"""
        subs = pysubs2.SSAFile()
        event1 = pysubs2.SSAEvent(start=1000, end=3000, text="Hello")
        event2 = pysubs2.SSAEvent(start=3500, end=5000, text="Hello")  # 500ms 間隙
        subs.events.extend([event1, event2])

        # 預設間隙閾值是 500ms，所以應該合併
        result = merge_continuous_events(subs, time_gap_threshold=500)
        assert result == 1
        assert len(subs.events) == 1

        # 間隙閾值是 100ms，所以不應該合併
        subs.events = [event1, event2]
        result = merge_continuous_events(subs, time_gap_threshold=100)
        assert result == 2
        assert len(subs.events) == 2


class TestMergeAssSubsAdvanced:
    """測試進階字幕合併"""

    def test_merge_ass_subs_advanced_empty(self):
        """測試空字幕進階合併"""
        subs = pysubs2.SSAFile()
        result = merge_ass_subs_advanced(subs)
        assert result == 0
        assert len(subs.events) == 0

    def test_merge_ass_subs_advanced_same_position(self):
        """測試相同位置的字幕合併"""
        subs = pysubs2.SSAFile()
        event1 = pysubs2.SSAEvent(start=1000, end=3000, text="{\\pos(100,200)}Hello")
        event2 = pysubs2.SSAEvent(start=3000, end=5000, text="{\\pos(100,200)}Hello")
        subs.events.extend([event1, event2])

        result = merge_ass_subs_advanced(subs, position_tolerance=10)
        assert result == 1
        assert len(subs.events) == 1
        assert subs.events[0].start == 1000
        assert subs.events[0].end == 5000

    def test_merge_ass_subs_advanced_different_position(self):
        """測試不同位置的字幕合併"""
        subs = pysubs2.SSAFile()
        event1 = pysubs2.SSAEvent(start=1000, end=3000, text="{\\pos(100,200)}Hello")
        event2 = pysubs2.SSAEvent(
            start=3000, end=5000, text="{\\pos(200,200)}Hello"
        )  # 不同 x 座標
        subs.events.extend([event1, event2])

        result = merge_ass_subs_advanced(subs, position_tolerance=10)
        assert result == 2  # 不應該合併
        assert len(subs.events) == 2

    def test_merge_ass_subs_advanced_within_tolerance(self):
        """測試在容差範圍內的位置合併"""
        subs = pysubs2.SSAFile()
        event1 = pysubs2.SSAEvent(start=1000, end=3000, text="{\\pos(100,200)}Hello")
        event2 = pysubs2.SSAEvent(
            start=3000, end=5000, text="{\\pos(105,200)}Hello"
        )  # 5px 差異
        subs.events.extend([event1, event2])

        result = merge_ass_subs_advanced(subs, position_tolerance=10)
        assert result == 1  # 應該合併
        assert len(subs.events) == 1

    def test_merge_ass_subs_advanced_no_pos_tags(self):
        """測試無位置標籤的字幕合併"""
        subs = pysubs2.SSAFile()
        event1 = pysubs2.SSAEvent(start=1000, end=3000, text="Hello")
        event2 = pysubs2.SSAEvent(start=3000, end=5000, text="Hello")
        subs.events.extend([event1, event2])

        result = merge_ass_subs_advanced(subs)
        assert result == 2  # 無位置標籤時不應該合併
        assert len(subs.events) == 2
