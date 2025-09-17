from __future__ import annotations

from src import inference_config, model_config, run_prompt


def main() -> None:
    run_prompt(model_config, inference_config)


if __name__ == "__main__":
    main()
