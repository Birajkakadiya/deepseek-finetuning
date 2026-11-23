"""CPU-only tests for Hugging Face publishing: no credentials or network required."""
import json
import struct
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from publish_to_hub import inspect_archive, prepare_upload, BASE_MODEL, DATASET, FILES


class PublishingTests(unittest.TestCase):
    def build_zip(self, location):
        adapter_cfg = {
            "base_model_name_or_path": BASE_MODEL,
            "peft_type": "LORA",
            "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        }
        run = {
            "base_model": BASE_MODEL, "dataset": DATASET,
            "gpu": "Tesla T4", "vram_gib": 14.6,
            "train_rows": 48, "eval_rows": 8,
            "config": {"max_steps": 2, "max_length": 512, "learning_rate": 0.0002},
            "train_metrics": {"train_loss": 0.277488},
            "eval_metrics": {"eval_loss": 0.342528, "eval_mean_token_accuracy": 0.90031},
        }
        trainer = {"global_step": 2}
        header = json.dumps({"layer.lora_A.weight": {
            "dtype": "F32", "shape": [1], "data_offsets": [0, 4]
        }}).encode("utf-8")
        weights = struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0)
        with ZipFile(location, "w") as z:
            for item in FILES:
                payload = weights if item == "adapter_model.safetensors" else "{}"
                if item == "adapter_config.json":
                    payload = json.dumps(adapter_cfg)
                z.writestr("adapter/" + item, payload)
            z.writestr("run_metadata.json", json.dumps(run))
            z.writestr("trainer_state.json", json.dumps(trainer))
        return location

    def test_dryrun_inspection_and_upload_files(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            path = self.build_zip(base / "smoke.zip")
            cfg, run, trainer, count = inspect_archive(path)
            self.assertEqual(trainer["global_step"], 2)
            self.assertEqual(count, 1)
            staged = base / "staging"
            staged.mkdir()
            prepare_upload(path, staged, "example/test-adapter")
            card = (staged / "README.md").read_text()
            self.assertIn("example/test-adapter", card)
            self.assertIn("48", card)
            self.assertIn("NVIDIA", card)
            self.assertTrue((staged / "adapter_model.safetensors").exists())
            self.assertTrue((staged / "training_metrics.json").exists())
            self.assertFalse((staged / "training_args.bin").exists())

    def test_reject_wrong_base_model(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.build_zip(Path(d) / "smoke.zip")
            with ZipFile(path, "a") as z:
                z.writestr("adapter/adapter_config.json", json.dumps(
                    {"base_model_name_or_path": "bad/model", "peft_type": "LORA"}))
            with self.assertRaises(ValueError):
                inspect_archive(path)


if __name__ == "__main__":
    unittest.main()
