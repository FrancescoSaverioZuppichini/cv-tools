import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: cv-tools sapiens2 --input IMAGE_OR_VIDEO [options]")
        return
    command = sys.argv.pop(1)
    if command == "sapiens2":
        from cv_tools.sapiens2.cli import main as run
    else:
        raise SystemExit(f"Unknown tool: {command}")
    run()


if __name__ == "__main__":
    main()
