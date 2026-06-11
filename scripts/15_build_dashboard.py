"""
Inject the JSON payload into the dashboard template and write final HTML.
"""
import json
from pathlib import Path

ROOT = Path("/home/claude/atlas")
TEMPLATE = ROOT / "scripts/dashboard_template.html"
PAYLOAD  = ROOT / "data/dashboard_payload.json"
OUT      = ROOT / "reports/dashboard_v2.html"

tpl = TEMPLATE.read_text(encoding="utf-8")
payload = PAYLOAD.read_text(encoding="utf-8")
out = tpl.replace("__PAYLOAD_PLACEHOLDER__", payload)
OUT.write_text(out, encoding="utf-8")
size = OUT.stat().st_size / 1024
print(f"Wrote {OUT}  ({size:.0f} KB)")
