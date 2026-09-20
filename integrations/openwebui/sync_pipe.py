"""Install or update the Horizon Relay Pipe in Open WebUI's database before the server starts.

Run from /app/backend. Importing open_webui.config applies database migrations, so this works on
a fresh volume. Valves are preserved; the Pipe is (re)enabled so the image's version is always live.
"""

import asyncio
import sys
from pathlib import Path

FUNCTION_ID = "horizon_relay"
SOURCE = Path(__file__).with_name("horizon_relay_pipe.py")


async def main() -> None:
    import open_webui.config  # noqa: F401  (runs migrations)
    from open_webui.models.functions import FunctionForm, FunctionMeta, Functions
    from open_webui.models.users import Users
    from open_webui.utils.plugin import load_function_module_by_id, replace_imports

    content = replace_imports(SOURCE.read_text())
    _, function_type, frontmatter = await load_function_module_by_id(FUNCTION_ID, content=content)
    meta = FunctionMeta(description=frontmatter.get("description"), manifest=frontmatter)
    name = frontmatter.get("title", "Horizon Relay")

    existing = await Functions.get_function_by_id(FUNCTION_ID)
    if existing is None:
        admin = await Users.get_super_admin_user()
        form = FunctionForm(id=FUNCTION_ID, name=name, content=content, meta=meta)
        if await Functions.insert_new_function(admin.id if admin else None, function_type, form) is None:
            sys.exit("horizon-relay: failed to install Pipe")
        action = "installed"
    elif existing.content != content or not existing.is_active:
        action = "updated"
    else:
        print(f"horizon-relay: Pipe {frontmatter.get('version')} already current")
        return
    await Functions.update_function_by_id(
        FUNCTION_ID, {"name": name, "content": content, "meta": meta.model_dump(), "type": function_type, "is_active": True}
    )
    print(f"horizon-relay: Pipe {frontmatter.get('version')} {action}")


if __name__ == "__main__":
    asyncio.run(main())
