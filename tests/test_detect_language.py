"""Tests for FileDetector.detect_language multi-language support.

Script-based checks (zh/ar/ru) need no external model. fast-langdetect based
checks (fr/id/en) are skipped when the package or its model is unavailable.
"""

import unittest

from scripts.file_detector import FileDetector

AR_QUERY = "ما المقصود بالبراغماتية في دراسة تفسير المعنى؟"
RU_QUERY = "Можно ли подать документы в Академию через Госуслуги или только лично?"
ID_QUERY = "Melalui cara apa warga negara asing dapat memiliki Perusahaan Pergadaian?"
FR_QUERY = "À quels étudiants la bourse d'encouragement de l'Ambassade de Chine au Togo est-elle destinée ?"
EN_QUERY = "Which factor was a better predictor of performance in the sentence-level insertion task?"


def _make_detector(query_text: str) -> FileDetector:
    import pandas as pd
    detector = FileDetector.__new__(FileDetector)
    detector.df = pd.DataFrame({"query": [query_text] * 3})
    detector.columns = {"query": "query"}
    return detector


class ScriptBasedDetectionTest(unittest.TestCase):
    def test_zh(self) -> None:
        self.assertEqual(_make_detector("中国的世界遗产有哪些？").detect_language(), "zh")

    def test_ar(self) -> None:
        self.assertEqual(_make_detector(AR_QUERY).detect_language(), "ar")

    def test_ru(self) -> None:
        self.assertEqual(_make_detector(RU_QUERY).detect_language(), "ru")

    def test_zh_takes_priority_over_other_scripts(self) -> None:
        self.assertEqual(_make_detector("什么是 p-frame 的定义？" + EN_QUERY).detect_language(), "zh")


class FastLangdetectDetectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import fast_langdetect  # noqa: F401
            fast_langdetect.detect("test sentence for model availability")
        except Exception:
            cls.skipTest(cls, "fast-langdetect or its model unavailable")

    def test_id(self) -> None:
        self.assertEqual(_make_detector(ID_QUERY).detect_language(), "id")

    def test_fr(self) -> None:
        self.assertEqual(_make_detector(FR_QUERY).detect_language(), "fr")

    def test_en_default(self) -> None:
        self.assertEqual(_make_detector(EN_QUERY).detect_language(), "en")

    def test_unsupported_language_falls_back_to_en(self) -> None:
        es_query = "¿Cuál es la definición de pragmática en el estudio de la interpretación?"
        self.assertEqual(_make_detector(es_query).detect_language(), "en")

    def test_spanish_is_not_misdetected_as_french(self) -> None:
        es_query = "¿Cuál es la definición de pragmática en el estudio de la interpretación?"
        self.assertNotEqual(_make_detector(es_query).detect_language(), "fr")


if __name__ == "__main__":
    unittest.main()
