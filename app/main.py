import argparse
import json
import os
import subprocess
import sys
from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def execute_tool(name: str, arguments: dict) -> str:
    if name == "Read":
        file_path = arguments.get("file_path", "")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"

    elif name == "Write":
        file_path = arguments.get("file_path", "")
        content = arguments.get("content", "")
        try:
            parent = os.path.dirname(file_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error writing file: {e}"

    elif name == "Bash":
        command = arguments.get("command", "")
        try:
            res = subprocess.run(
                ["/bin/sh", "-c", command], capture_output=True, text=True
            )
            result = ""
            if res.stdout:
                result += res.stdout
            if res.stderr:
                if result:
                    result += "\n"
                result += res.stderr
            return result or "Command executed successfully (no output)"
        except Exception as e:
            return f"Error executing command: {e}"

    else:
        return f"Unknown tool: {name}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read and return the contents of a file",
                "parameters": {
                    "type": "object",
                    "required": ["file_path"],
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to read",
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Write",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "required": ["file_path", "content"],
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path of the file to write to",
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "Execute a shell command",
                "parameters": {
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute",
                        }
                    },
                },
            },
        },
    ]

    messages = [{"role": "user", "content": args.p}]

    while True:
        try:
            chat = client.chat.completions.create(
                model="anthropic/claude-haiku-4.5",
                messages=messages,
                tools=tools,
            )
        except Exception as e:
            print(f"API Error: {e}", file=sys.stderr)
            raise e

        message = chat.choices[0].message
        tool_calls = message.tool_calls

        # Assistantの返答をメッセージ履歴に追加する形式にフォーマット
        assistant_msg = {
            "role": "assistant",
        }
        if message.content:
            assistant_msg["content"] = message.content
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                } for tc in tool_calls
            ]
        
        messages.append(assistant_msg)

        if tool_calls:
            for tool_call in tool_calls:
                name = tool_call.function.name
                
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except Exception:
                    arguments = {}

                result = execute_tool(name, arguments)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": result
                })
            # ツール実行結果を含めて再度LLMに投げるため、ループを継続
            continue

        # ツール呼び出しがない場合：最終出力を標準出力に表示して終了
        print("Logs from your program will appear here!", file=sys.stderr)
        if message.content:
            print(message.content)
        break


if __name__ == "__main__":
    main()
