import unittest
from data_utils import valid_record, record_to_conversation, select_disjoint_rows

class DatasetTests(unittest.TestCase):
    def test_schema(self):
        r = {"id": "a", "input": "hello", "output": "print('hello')"}
        self.assertTrue(valid_record(r))
        c = record_to_conversation(r)
        self.assertEqual(c["prompt"][0]["role"], "user")
        self.assertEqual(c["completion"][0]["role"], "assistant")

    def test_score(self):
        self.assertFalse(valid_record({"input": "x", "output": "y", "average_test_score": 0.5}, 0.8))
        self.assertTrue(valid_record({"input": "x", "output": "y", "average_test_score": 0.9}, 0.8))

    def test_disjoint(self):
        rows = [{"id": str(i), "input": str(i), "output": str(i)} for i in range(10)]
        rows.append(rows[0])
        train, eval_rows = select_disjoint_rows(rows, 6, 2)
        self.assertEqual(len(train), 6)
        self.assertFalse({r["id"] for r in train} & {r["id"] for r in eval_rows})

if __name__ == "__main__":
    unittest.main()
