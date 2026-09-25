import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def argparse_defaults(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        if not isinstance(node.args[0], ast.Constant):
            continue
        flag = node.args[0].value
        default = next((kw.value for kw in node.keywords if kw.arg == "default"), None)
        if default is not None:
            values[flag] = ast.literal_eval(default)
    return values


class PublicReproducibilityTest(unittest.TestCase):
    def test_public_anonymized_set_is_complete(self):
        data_root = ROOT / "data" / "anonymized_test"
        images = sorted((data_root / "images").glob("*.jpg"))
        frame_dirs = sorted((data_root / "ordered_frames").glob("THYQC_TEST_*"))
        self.assertEqual(len(images), 100)
        self.assertEqual(len(frame_dirs), 100)
        self.assertTrue(all(len(list(path.glob("*.jpg"))) == 4 for path in frame_dirs))
        for name in (
            "test100_labels.csv",
            "test100_manifest.jsonl",
            "test100_video_manifest.jsonl",
        ):
            self.assertTrue((data_root / name).is_file())

    def test_three_seed_weights_are_bundled(self):
        weight_root = ROOT / "weights" / "thyqc"
        for seed in (41, 42, 43):
            backbone = weight_root / f"seed{seed}_backbone_state.pt"
            temporal = weight_root / f"seed{seed}_gt_qdm_state.pt"
            self.assertGreater(backbone.stat().st_size, 1_000_000)
            self.assertGreater(temporal.stat().st_size, 10_000)

    def test_public_evaluation_assets_cover_all_seeds(self):
        evaluator = ROOT / "code" / "evaluate_public.py"
        self.assertTrue(evaluator.is_file())
        for seed in (41, 42, 43):
            features = ROOT / "results" / "features" / f"seed{seed}_public_features.jsonl"
            self.assertTrue(features.is_file())
            rows = [json.loads(line) for line in features.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 100)
            self.assertTrue(all(len(row["token_probs"]) == 5 for row in rows))

    def test_teacher_code_is_present_without_teacher_assets(self):
        teacher_root = ROOT / "code" / "teacher"
        for name in (
            "build_teacher_rationale_sft_dataset.py",
            "train_teacher_rationale_adapter.py",
            "generate_teacher_structured_rationales.py",
            "rationale_semantic_rules.py",
            "audit_teacher_knowledge_packages.py",
        ):
            self.assertTrue((teacher_root / name).is_file())
        self.assertFalse(any(ROOT.glob("**/*teacher*.pt")))
        self.assertFalse(any(ROOT.glob("**/*rag*")))

    def test_stage1_defaults_match_reference_configuration(self):
        defaults = argparse_defaults(ROOT / "code" / "train_g2d_uot_thyqc_20260919.py")
        self.assertEqual(defaults["--lambda-prob"], 0.06)
        self.assertEqual(defaults["--lambda-g2d"], 1.8)
        self.assertEqual(defaults["--semantic-cost-mode"], "fixed_jaccard")
        self.assertEqual(defaults["--uot-loss-mode"], "transport")

    def test_stage2_defaults_match_reference_configuration(self):
        defaults = argparse_defaults(ROOT / "code" / "train_g2d_uot_gt_qdm_seed42_20260919.py")
        self.assertEqual(defaults["--lambda-prob"], 0.06)
        self.assertEqual(defaults["--lambda-g2d"], 1.8)
        self.assertEqual(defaults["--semantic-cost-mode"], "fixed_jaccard")
        self.assertEqual(defaults["--uot-loss-mode"], "transport")

    def test_seed_templates_use_same_objective(self):
        for path in (ROOT / "configs").glob("seed*.json"):
            config = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(config["lambda_prob"], 0.06)
            self.assertEqual(config["lambda_g2d"], 1.8)
            self.assertEqual(config["semantic_cost_mode"], "fixed_jaccard")
            self.assertEqual(config["uot_loss_mode"], "transport")

    def test_readme_uses_paper_repository_style(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("Official implementation", readme)
        self.assertNotIn("This public release contains", readme)
        self.assertNotIn("No raw clinical dataset", readme)
        self.assertIn("2 percentage points", readme)

    def test_two_stage_pipeline_has_cache_builder(self):
        builder = ROOT / "code" / "build_g2d_gt_qdm_prob_cache_20260919.py"
        self.assertTrue(builder.is_file())
        for source in (ROOT / "code").glob("*.py"):
            self.assertNotIn("/root/", source.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
