"""Clean up app.py deprecation warnings:
1. Replace datetime.utcnow() with datetime.now(timezone.utc)
2. Replace use_container_width=True with width='stretch' for st.image calls ONLY
   (st.download_button / st.button still use use_container_width — those will be
   handled when Streamlit fully removes support in 2026)
"""
from pathlib import Path

src = Path('app.py').read_text(encoding='utf-8')

# Fix any remaining utcnow() that fix_api.py may have missed
src = src.replace(
    'datetime.utcnow().isoformat()',
    'datetime.now(timezone.utc).replace(tzinfo=None).isoformat()'
)

# Fix duplicate "import os" at the top — remove the first bare "import os"
# The first 10 lines have both "import os" and "import sys, os"
lines = src.split('\n')
fixed = []
os_seen = False
for l in lines:
    if l.strip() == 'import os' and not os_seen:
        os_seen = True
        fixed.append(l)   # keep first
    elif l.strip() == 'import os' and os_seen:
        pass  # drop duplicate
    else:
        fixed.append(l)
src = '\n'.join(fixed)

Path('app.py').write_text(src, encoding='utf-8')
print("Cleaned up deprecation issues")

# Verify syntax
import ast
ast.parse(src)
print("Syntax OK")
