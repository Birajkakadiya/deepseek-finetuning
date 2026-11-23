"""Validate the professor submission notebook without requiring a GPU or network."""
import ast
import json
import unittest
from pathlib import Path


class ProfessorNotebookTests(unittest.TestCase):
    def test_code_cells_have_valid_python_syntax(self):
        file = Path(__file__).resolve().parents[1] / "colab" / "Professor_Submission_DeepSeek_Coder_6_7B.ipynb"
        nb = json.loads(file.read_text(encoding="utf-8"))
        self.assertEqual(nb["nbformat"], 4)
        cells = nb["cells"]
        self.assertEqual(len(cells), 10)
        self.assertTrue(all(c["cell_type"] == "code" for c in cells))
        self.assertEqual(len({c["id"] for c in cells}), len(cells))
        # Saved outputs are intentional: the professor can review real archived results
        # without executing training or accessing private model weights.
        self.assertTrue(any(c.get("outputs") for c in cells))
        for i, c in enumerate(cells, start=1):
            with self.subTest(cell=i):
                ast.parse("".join(c["source"]), filename=f"cell_{i}.py")

    def test_optional_training_requires_opt_in(self):
        file = Path(__file__).resolve().parents[1] / "colab" / "Professor_Submission_DeepSeek_Coder_6_7B.ipynb"
        nb = json.loads(file.read_text(encoding="utf-8"))
        source = "
".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn("RUN_NEW_TRAINING = False", source)
        self.assertIn("nvidia/OpenCodeInstruct", source)
        self.assertIn("deepseek-ai/deepseek-coder-6.7b-instruct", source)
        self.assertIn('HF_TOKEN', source)
        self.assertNotIn('RUN_NEW_TRAINING = True', source)


if __name__ == "__main__":
    unittest.main()
