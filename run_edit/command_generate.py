import json
import os
from pathlib import Path

def generate_commands_from_launch():
    launch_path = Path(".vscode") / "launch.json"
    commands = {}

    with open(launch_path, "r", encoding="utf-8") as f:
        configs = json.load(f)["configurations"]

    # 获取当前工作目录作为 workspaceFolder
    workspace_root = Path(os.getcwd()).as_posix()

    for config in configs:
        name = config["name"]
        
        program = config["program"]
        if "${file}" in program:
            program = program.replace("${file}", "<需要替换为实际文件路径>")
        else:
            program = str(Path(program).as_posix())  # 统一路径格式

        cwd = config.get("cwd", "")
        if "${workspaceFolder}" in cwd:
            cwd = cwd.replace("${workspaceFolder}", workspace_root)
        elif not cwd:
            cwd = workspace_root
        else:
            cwd = str(Path(cwd).as_posix())

        args = [arg.strip(" ,") for arg in config.get("args", [])]  # 清理多余符号
        args_str = " ".join(args)

        commands[name] = f"!python '{program}' {args_str}"

    return commands

if __name__ == "__main__":
    commands = generate_commands_from_launch()
    
    # 打印生成的命令
    for name, cmd in commands.items():
        print(f"# {name}")
        print(f"{cmd}\n")
        print("-" * 80)
    