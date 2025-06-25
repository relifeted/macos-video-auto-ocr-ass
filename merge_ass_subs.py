import argparse
import re

import pysubs2


def parse_pos(text):
    match = re.search(r"\\pos\((\d+),(\d+)\)", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def merge_ass_subs(
    input_ass,
    output_ass,
    position_tolerance=10,
    time_gap_threshold=500,
    base_font_size=24,
):
    subs = pysubs2.load(input_ass)
    merged_events = []
    events = sorted(subs.events, key=lambda e: (e.text))
    if not events:
        subs.save(output_ass)
        return
    current = events[0]
    for next_event in events[1:]:
        # 內容比對（去除 pos tag）
        cur_pos = parse_pos(current.text)
        next_pos = parse_pos(next_event.text)
        cur_text = current.text
        next_text = next_event.text
        # 位置相近
        pos_close = (
            cur_pos[0] is not None
            and next_pos[0] is not None
            and abs(cur_pos[0] - next_pos[0]) <= position_tolerance
            and abs(cur_pos[1] - next_pos[1]) <= position_tolerance
        )
        # 內容相同
        text_same = cur_text == next_text
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
    subs.save(output_ass)
    print(f"[INFO] 合併完成，原始事件數: {len(events)}，合併後: {len(merged_events)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge duplicate ASS subtitle events.")
    parser.add_argument("input_ass", help="Input ASS file")
    parser.add_argument("output_ass", help="Output merged ASS file")
    parser.add_argument(
        "--position-tolerance",
        type=int,
        default=10,
        help="Position tolerance in pixels (default: 10)",
    )
    parser.add_argument(
        "--time-gap-threshold",
        type=int,
        default=500,
        help="Max time gap (ms) to merge (default: 500)",
    )
    parser.add_argument(
        "--base-font-size", type=int, default=24, help="基礎字體大小 (default: 24)"
    )
    args = parser.parse_args()
    merge_ass_subs(
        args.input_ass,
        args.output_ass,
        args.position_tolerance,
        args.time_gap_threshold,
        args.base_font_size,
    )
