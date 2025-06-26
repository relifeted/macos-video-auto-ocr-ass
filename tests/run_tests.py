#!/usr/bin/env python3
"""
測試運行腳本

提供便捷的測試運行功能
"""

import os
import subprocess
import sys
from pathlib import Path


def run_tests(test_type="all", coverage=False, verbose=False):
    """
    運行測試

    Args:
        test_type: 測試類型 ("all", "unit", "integration", "fast")
        coverage: 是否生成覆蓋率報告
        verbose: 是否詳細輸出
    """
    # 確保在正確的目錄
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # 構建 pytest 命令
    cmd = ["python", "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(
            ["--cov=macos_video_auto_ocr_ass", "--cov-report=html", "--cov-report=term"]
        )

    # 根據測試類型選擇測試
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    elif test_type == "macos":
        cmd.extend(["-m", "macos"])
    elif test_type == "all":
        pass  # 運行所有測試
    else:
        print(f"未知的測試類型: {test_type}")
        return False

    print(f"運行測試: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        print("✅ 所有測試通過！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 測試失敗，退出碼: {e.returncode}")
        return False


def run_specific_test(test_file, test_function=None):
    """
    運行特定測試

    Args:
        test_file: 測試檔案路徑
        test_function: 特定測試函數名稱（可選）
    """
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    cmd = ["python", "-m", "pytest", "-v"]

    if test_function:
        cmd.append(f"tests/{test_file}::{test_function}")
    else:
        cmd.append(f"tests/{test_file}")

    print(f"運行特定測試: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        print("✅ 測試通過！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 測試失敗，退出碼: {e.returncode}")
        return False


def main():
    """主函數"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python tests/run_tests.py [test_type]")
        print("  python tests/run_tests.py specific <test_file> [test_function]")
        print("")
        print("測試類型:")
        print("  all          - 運行所有測試")
        print("  unit         - 運行單元測試")
        print("  integration  - 運行整合測試")
        print("  fast         - 運行快速測試（排除慢速測試）")
        print("  macos        - 運行 macOS 特定測試")
        print("")
        print("範例:")
        print("  python tests/run_tests.py unit")
        print("  python tests/run_tests.py specific test_config.py")
        print("  python tests/run_tests.py specific test_config.py TestAppConfig")
        return

    command = sys.argv[1]

    if command == "specific":
        if len(sys.argv) < 3:
            print("請指定測試檔案")
            return

        test_file = sys.argv[2]
        test_function = sys.argv[3] if len(sys.argv) > 3 else None

        success = run_specific_test(test_file, test_function)
        sys.exit(0 if success else 1)

    else:
        # 檢查是否要生成覆蓋率報告
        coverage = "--coverage" in sys.argv
        verbose = "--verbose" in sys.argv or "-v" in sys.argv

        success = run_tests(command, coverage, verbose)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
