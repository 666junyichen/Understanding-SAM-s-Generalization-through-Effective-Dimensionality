import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data.py"


class DataModuleStaticTests(unittest.TestCase):
    def test_data_module_defines_required_public_api(self):
        source = DATA_FILE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        function_names = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        class_names = {
            node.name for node in tree.body if isinstance(node, ast.ClassDef)
        }

        self.assertIn("AddGaussianNoise", class_names)
        self.assertIn("get_train_loader", function_names)
        self.assertIn("get_id_test_loader", function_names)
        self.assertIn("get_ood_loaders", function_names)
        self.assertIn("get_ood_transform", function_names)

    def test_config_defines_ood_corruption_parameters(self):
        config_file = PROJECT_ROOT / "config.py"
        source = config_file.read_text(encoding="utf-8")

        self.assertIn("CIFAR10_MEAN", source)
        self.assertIn("CIFAR10_STD", source)
        self.assertIn("NOISE_STD", source)
        self.assertIn("BLUR_KERNEL_SIZE", source)
        self.assertIn("BRIGHTNESS_FACTOR", source)


if __name__ == "__main__":
    unittest.main()
