from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core import get_model, startup_diagnostics


def main() -> None:
    print("Startup diagnostics:")
    for label, value in startup_diagnostics():
        print(f"- {label}: {value}")

    print("Loading model to confirm cold start...")
    model = get_model()
    print(f"Model ready with {len(model.pathologies)} pathologies.")


if __name__ == "__main__":
    main()
