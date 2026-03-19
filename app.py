"""Codon Optimization Tool – Streamlit entry point.

Run with:
    streamlit run app.py
"""

from ui.app_controller import StreamlitApp


def main() -> None:
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
