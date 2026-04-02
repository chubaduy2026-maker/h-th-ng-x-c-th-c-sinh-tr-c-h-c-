from __future__ import annotations

try:
    from src.database import reset_all_attendance
except ImportError:
    from database import reset_all_attendance


def main() -> None:
    print("This action will reset is_present=False for all students.")
    confirm = input("Type YES to continue: ").strip()

    if confirm != "YES":
        print("Cancelled.")
        return

    result = reset_all_attendance()
    print(
        "Reset completed: "
        f"matched={result['matched']} | modified={result['modified']}"
    )


if __name__ == "__main__":
    main()
