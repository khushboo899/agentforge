"""Offline smoke tests for AgentForge's deterministic routing and calculator."""
from __future__ import annotations

import os
import unittest

from app.graph.nodes import (
    analyze_task,
    detect_document_query,
    expand_query,
    extract_calculation,
    get_available_pdfs,
    is_general_query,
)
from app.tools.calculator import calculator


class RoutingTests(unittest.TestCase):
    def test_greeting_is_deterministic(self):
        self.assertTrue(is_general_query("hello"))
        result = analyze_task({"question": "hello"})
        self.assertEqual(result["task_type"], "greeting")
        self.assertFalse(result["requires_rag"])

    def test_document_intent(self):
        self.assertTrue(detect_document_query("Explain this PDF"))

    def test_fx_shorthand_expands_to_function(self):
        self.assertTrue(any("function" in q.lower() for q in expand_query("python fx")))

    def test_calculation_detection(self):
        self.assertEqual(extract_calculation("calculate 24 / 3"), "24 / 3")
        self.assertEqual(extract_calculation("(12 + 8) * 2"), "(12 + 8) * 2")
        self.assertIsNone(extract_calculation("explain multiplication"))

    def test_pdf_discovery_returns_only_existing_pdf_files(self):
        paths = get_available_pdfs()
        self.assertEqual(paths, sorted(paths))
        self.assertTrue(all(os.path.isfile(p) and p.lower().endswith(".pdf") for p in paths))


class CalculatorTests(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(calculator.invoke({"expression": "(25 * 16) / 2"}), "200.0")

    def test_disallows_python_execution(self):
        result = calculator.invoke({"expression": "__import__('os').system('echo unsafe')"})
        self.assertTrue(result.startswith("Calculation error:"))

    def test_division_by_zero_is_reported(self):
        result = calculator.invoke({"expression": "1 / 0"})
        self.assertTrue(result.startswith("Calculation error:"))


if __name__ == "__main__":
    unittest.main()
