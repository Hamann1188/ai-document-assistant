"""End-to-end check of the web UI in a real browser (manual; calls Claude).

Asks three questions through the UI (about $0.06), checks that answers finish,
citation links open the cited PDF, there are no console errors (CSP violations
show up there) and no horizontal scrolling at phone width. Screenshots go to
.cache/ui-smoke/.

Needs the stack running with the four sample PDFs uploaded, and Microsoft Edge or
Google Chrome installed (Playwright drives the installed browser, nothing to download):

    uv run --with playwright python scripts/ui_smoke.py [base_url] [msedge|chrome]
"""

import re
import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
CHANNEL = sys.argv[2] if len(sys.argv) > 2 else "msedge"
OUT = Path(__file__).resolve().parents[1] / ".cache" / "ui-smoke"
CITATION_HREF = re.compile(r"/documents/[0-9a-f-]{36}/file#page=\d+")


def wait_for_answer(page, turn: int) -> None:
    # The meta line is filled by the `done` event; an alert covers the error paths.
    page.wait_for_function(
        """n => {
            const t = document.querySelectorAll('.turn')[n - 1];
            return t && (t.querySelector('.meta').textContent.trim() || t.querySelector('.alert'));
        }""",
        arg=turn,
        timeout=120_000,
    )


def open_page(browser, problems: list[str], **context_options):
    page = browser.new_page(**context_options)
    page.on(
        "console",
        lambda m: m.type in ("error", "warning") and problems.append(f"console: {m.text}"),
    )
    page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
    page.goto(BASE)
    expect(page.locator(".doc").first).to_be_visible(timeout=15_000)
    return page


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=CHANNEL)
        size = {"width": 1366, "height": 860}

        page = open_page(browser, problems, viewport=size, color_scheme="light")
        page.screenshot(path=OUT / "1-start.png")
        page.locator(".chip", has_text="имплантация").click()
        wait_for_answer(page, 1)
        page.locator("#question").fill("How early must I cancel an appointment?")
        page.keyboard.press("Enter")
        wait_for_answer(page, 2)
        page.screenshot(path=OUT / "2-answers.png")

        hrefs = set(page.locator(".a-body .cite").evaluate_all("els => els.map(e => e.href)"))
        if not hrefs:
            problems.append("no citation markers in the answers")
        for href in hrefs:
            path = href.removeprefix(BASE)
            if not CITATION_HREF.fullmatch(path):
                problems.append(f"unexpected citation link: {path}")
                continue
            response = page.request.get(BASE + path.split("#")[0])
            if response.headers.get("content-type") != "application/pdf":
                problems.append(f"{path}: HTTP {response.status}, not a PDF")
        for alert in page.locator(".turn .alert").all_inner_texts():
            problems.append(f"alert in an answer: {alert}")

        dark = open_page(browser, problems, viewport=size, color_scheme="dark")
        dark.locator(".chip", has_text="braces").click()
        wait_for_answer(dark, 1)
        dark.screenshot(path=OUT / "3-dark-out-of-scope.png")

        phone = open_page(
            browser, problems, viewport={"width": 400, "height": 820}, color_scheme="light"
        )
        phone.screenshot(path=OUT / "4-phone.png")
        if phone.evaluate("document.documentElement.scrollWidth > window.innerWidth"):
            problems.append("horizontal scrolling at 400 px")

        browser.close()

    print(f"screenshots: {OUT}")
    for problem in problems:
        print("PROBLEM:", problem)
    print("FAIL" if problems else "PASS")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
