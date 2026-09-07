from __future__ import annotations

import unittest

from scripts.file_detector import detect_language


class DetectLanguageTest(unittest.TestCase):
    def test_chinese_majority(self):
        self.assertEqual(detect_language(["什么是机器学习", "深度学习简介", "hello"]), "zh")

    def test_english_majority(self):
        self.assertEqual(detect_language(["What is ML?", "Deep learning basics", "你好"]), "en")

    def test_empty_defaults_english(self):
        self.assertEqual(detect_language([]), "en")

    def test_tie_defaults_english(self):
        self.assertEqual(detect_language(["中文", "English"]), "en")


if __name__ == "__main__":
    unittest.main()
