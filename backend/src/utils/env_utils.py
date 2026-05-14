import os


def update_env_file(filepath: str, updates: dict):
    lines = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

    existing_keys = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith("#"):
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                lines[i] = f"{key}={updates[key]}\n"
                existing_keys.add(key)

    for key, value in updates.items():
        if key not in existing_keys:
            lines.append(f"{key}={value}\n")

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)


def read_env_value(filepath: str, key: str, default: str = "") -> str:
    if not os.path.exists(filepath):
        return default
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if "=" in stripped and not stripped.startswith("#"):
                k, v = stripped.split("=", 1)
                if k.strip() == key:
                    return v.strip()
    return default
