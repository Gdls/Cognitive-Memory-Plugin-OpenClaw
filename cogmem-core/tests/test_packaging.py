from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from cogmem_core.cli import main
from cogmem_core.openclaw_bundle import bundled_plugin, install_openclaw_plugin, plugin_manifest


class WheelBundleTests(unittest.TestCase):
    def test_bundled_plugin_contains_openclaw_entry(self) -> None:
        root = bundled_plugin()
        names = {item.name for item in root.iterdir()}
        self.assertIn("index.js", names)
        self.assertIn("openclaw.plugin.json", names)
        self.assertIn("tools.js", names)
        self.assertIn("skill", names)

    def test_install_openclaw_plugin_copies_skill_and_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "openclaw-cogmem"
            installed = install_openclaw_plugin(dest)
            self.assertTrue((installed / "index.js").exists())
            self.assertTrue((installed / "tools.js").exists())
            self.assertTrue((installed / "skill" / "cogmem-memory-guide" / "SKILL.md").exists())
            manifest = plugin_manifest(installed)
            self.assertEqual(manifest["install_dir"], str(installed))
            self.assertEqual((manifest.get("plugin") or {}).get("id"), "cogmem")

    def test_cli_plugin_path_prints_default(self) -> None:
        self.assertEqual(main(["plugin-path"]), 0)
        self.assertEqual(main(["--version"]), 0)


if __name__ == "__main__":
    unittest.main()
