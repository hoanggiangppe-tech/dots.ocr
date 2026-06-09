"""
dots.ocr - Local Web Interface
Connects to a vLLM server for document/image OCR and layout parsing.
"""

import os
import json
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests as _requests
import gradio as gr
from PIL import Image

from dots_ocr.parser import DotsOCRParser
from dots_ocr.utils import dict_promptmode_to_prompt
from dots_ocr.utils.consts import MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.doc_utils import load_images_from_pdf
from dots_ocr.utils.image_utils import fetch_image

# ── Constants ────────────────────────────────────────────────────────────────

PROMPT_FITZ = {
    "prompt_layout_all_en": True,
    "prompt_layout_only_en": True,
    "prompt_ocr": True,
    "prompt_web_parsing": False,
    "prompt_scene_spotting": False,
    "prompt_image_to_svg": False,
    "prompt_general": False,
}

PROMPT_TEMP = {
    "prompt_layout_all_en": 0.1,
    "prompt_layout_only_en": 0.1,
    "prompt_ocr": 0.1,
    "prompt_web_parsing": 0.1,
    "prompt_scene_spotting": 0.1,
    "prompt_image_to_svg": 0.9,
    "prompt_general": 0.5,
}

PROMPT_LABELS = {
    "prompt_layout_all_en":  "Layout Parsing (Full)",
    "prompt_layout_only_en": "Layout Detection Only",
    "prompt_ocr":            "OCR Text Extraction",
    "prompt_web_parsing":    "Web Page Parsing",
    "prompt_scene_spotting": "Scene Text Spotting",
    "prompt_image_to_svg":   "Image → SVG",
    "prompt_general":        "Free QA / Custom Prompt",
}

DEMO_IMAGES_DIR = "./assets/showcase/origin"
DEMO_IMAGES = []
if os.path.exists(DEMO_IMAGES_DIR):
    exts = {".jpg", ".jpeg", ".png", ".pdf"}
    DEMO_IMAGES = sorted(
        p for p in Path(DEMO_IMAGES_DIR).iterdir()
        if p.suffix.lower() in exts
    )

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_session():
    return {
        "pages": [],
        "page_idx": 0,
        "is_parsed": False,
        "parsed_pages": [],
        "temp_dir": None,
    }


def parse_server_url(server_url):
    """Accept either a full URL (https://xxx.ngrok.io) or plain IP."""
    server_url = server_url.strip().rstrip("/")
    if server_url.startswith("http"):
        p = urlparse(server_url)
        protocol = p.scheme
        ip = p.hostname
        port = p.port or (443 if protocol == "https" else 80)
    else:
        # plain IP or host
        protocol = "http"
        ip = server_url
        port = 8000
    return protocol, ip, port


def build_parser(server_url, model_name, temperature, dpi=150):
    protocol, ip, port = parse_server_url(server_url)
    return DotsOCRParser(
        protocol=protocol,
        ip=ip,
        port=int(port),
        model_name=model_name,
        temperature=float(temperature),
        dpi=int(dpi),
        output_dir=tempfile.mkdtemp(prefix="dotsocr_"),
    )


def load_file(file_path, state):
    if not file_path or not os.path.exists(file_path):
        return None, "0 / 0", state

    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        pages = load_images_from_pdf(file_path, dpi=150)
    elif ext in {".jpg", ".jpeg", ".png"}:
        pages = [Image.open(file_path).convert("RGB")]
    else:
        return None, "Unsupported format", state

    state["pages"] = pages
    state["page_idx"] = 0
    state["is_parsed"] = False
    state["parsed_pages"] = []
    return pages[0], f"1 / {len(pages)}", state


def navigate(direction, state):
    pages = state["pages"]
    if not pages:
        return None, "0 / 0", "", state

    idx = state["page_idx"]
    if direction == "prev":
        idx = max(0, idx - 1)
    else:
        idx = min(len(pages) - 1, idx + 1)
    state["page_idx"] = idx

    img = pages[idx]
    if state["is_parsed"] and idx < len(state["parsed_pages"]):
        r = state["parsed_pages"][idx]
        if r.get("layout_image"):
            img = r["layout_image"]
        json_txt = json.dumps(r.get("cells_data", []), ensure_ascii=False, indent=2)
    else:
        json_txt = ""

    return img, f"{idx+1} / {len(pages)}", json_txt, state


def run_parse(file_input, demo_file, prompt_mode, custom_prompt,
              server_url, model_name, dpi, page_from, page_to, min_px, max_px, state):

    path = file_input or demo_file
    if not path or not os.path.exists(str(path)):
        return (
            None, "⚠️ Please upload a file or select a demo image.",
            "", "", gr.update(visible=False), "0 / 0", "", state,
        )

    # Cleanup previous temp dir
    if state.get("temp_dir") and os.path.exists(state["temp_dir"]):
        shutil.rmtree(state["temp_dir"], ignore_errors=True)

    temp_dir = tempfile.mkdtemp(prefix="dotsocr_")
    state["temp_dir"] = temp_dir

    temperature = PROMPT_TEMP.get(prompt_mode, 0.1)
    fitz_pre    = PROMPT_FITZ.get(prompt_mode, True)

    _original_general = dict_promptmode_to_prompt.get("prompt_general", " ")
    if prompt_mode == "prompt_general" and custom_prompt.strip():
        dict_promptmode_to_prompt["prompt_general"] = custom_prompt.strip()

    try:
        parser = build_parser(server_url, model_name, temperature, dpi)
        parser.min_pixels = int(min_px) if min_px else None
        parser.max_pixels = int(max_px) if max_px else None

        ext = Path(path).suffix.lower()
        fname = Path(path).stem

        if ext == ".pdf":
            # Load page by page to avoid bulk memory usage
            start = max(0, int(page_from) - 1)
            end   = int(page_to) - 1 if page_to else None
            try:
                pages = load_images_from_pdf(path, dpi=int(dpi),
                                             start_page_id=start, end_page_id=end)
            except Exception as mem_exc:
                if "malloc" in str(mem_exc) or "MemoryError" in str(type(mem_exc)):
                    raise MemoryError(str(mem_exc))
                raise

            results = []
            for i, img in enumerate(pages):
                page_no = start + i
                r = parser._parse_single_image(
                    img, prompt_mode, temp_dir,
                    fname, source="pdf", page_idx=page_no,
                )
                r["file_path"] = path
                results.append(r)
        else:
            results = parser.parse_image(path, fname, prompt_mode, temp_dir,
                                         fitz_preprocess=fitz_pre)

        # Rebuild state pages from parse results
        parsed_pages = []
        all_md = []
        all_cells = []

        for r in results:
            pr = {"layout_image": None, "cells_data": None, "md": ""}
            if r.get("layout_image_path") and os.path.exists(r["layout_image_path"]):
                pr["layout_image"] = Image.open(r["layout_image_path"])
            if r.get("layout_info_path") and os.path.exists(r["layout_info_path"]):
                with open(r["layout_info_path"], encoding="utf-8") as f:
                    pr["cells_data"] = json.load(f)
                    all_cells.extend(pr["cells_data"] if isinstance(pr["cells_data"], list) else [pr["cells_data"]])
            md_path = r.get("md_content_path") or r.get("md_content_nohf_path")
            if md_path and os.path.exists(md_path):
                with open(md_path, encoding="utf-8") as f:
                    pr["md"] = f.read()
                    all_md.append(pr["md"])
            parsed_pages.append(pr)

        state["pages"] = [p["layout_image"] or state["pages"][i]
                         for i, p in enumerate(parsed_pages)]
        state["page_idx"] = 0
        state["is_parsed"] = True
        state["parsed_pages"] = parsed_pages

        combined_md = "\n\n---\n\n".join(all_md)

        if not combined_md.strip():
            return (
                None,
                "⚠️ **Server trả về nội dung rỗng.**\n\n"
                "Kiểm tra:\n"
                "1. Model name trong **Server Config** phải là `model`\n"
                "2. vLLM đã sẵn sàng chưa (Cell 4b log `✅ vLLM sẵn sàng`)\n"
                "3. Thử **🔌 Test Connection** — nếu xanh thì server OK\n"
                "4. Chạy lại Cell 4b trên Kaggle nếu server mới restart",
                "", "", gr.update(visible=False), "0 / 0", "", state,
            )

        first = parsed_pages[0]
        first_img = first["layout_image"] or (Image.open(path) if ext != ".pdf" else state["pages"][0])
        first_json = json.dumps(first.get("cells_data") or [], ensure_ascii=False, indent=2)

        page_range_str = f"trang {int(page_from)}–{int(page_to)}" if page_to else f"từ trang {int(page_from)}"
        info = (
            f"**File:** `{Path(path).name}`  \n"
            f"**Pages parsed:** {len(results)} ({page_range_str})  |  "
            f"**Elements:** {len(all_cells)}  \n"
            f"**Model:** `{model_name}` @ `{server_url}`  |  "
            f"**Prompt:** `{prompt_mode}`"
        )

        zip_path = os.path.join(temp_dir, f"results_{uuid.uuid4().hex[:6]}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(temp_dir):
                for fn in files:
                    if not fn.endswith(".zip"):
                        full = os.path.join(root, fn)
                        zf.write(full, os.path.relpath(full, temp_dir))

        return (
            first_img, info, combined_md, combined_md,
            gr.update(value=zip_path, visible=True),
            f"1 / {len(results)}", first_json, state,
        )

    except (MemoryError, Exception) as exc:
        import traceback; traceback.print_exc()
        err = str(exc)
        if "malloc" in err or isinstance(exc, MemoryError):
            msg = (
                "❌ **Lỗi bộ nhớ (MemoryError)** — PDF quá lớn để render.\n\n"
                "**Cách xử lý:**\n"
                "1. Mở **🔧 Advanced** → kéo **PDF render DPI** xuống (thử `100` hoặc `72`)\n"
                "2. Giới hạn số trang trong **Page range** (VD: trang 1–10, rồi 11–20...)\n"
                "3. Bấm Parse lại"
            )
        elif "Connection refused" in err or "connect" in err.lower() or "APIConnection" in err:
            msg = (
                f"⚠️ **Không kết nối được server** tại `{server_url}`\n\n"
                "Hãy chắc chắn Colab đang chạy và URL ngrok còn hiệu lực."
            )
        else:
            msg = f"❌ Error: {err}"
        return (None, msg, "", "", gr.update(visible=False), "0 / 0", "", state)
    finally:
        dict_promptmode_to_prompt["prompt_general"] = _original_general


def clear_all(state):
    if state.get("temp_dir") and os.path.exists(state["temp_dir"]):
        shutil.rmtree(state["temp_dir"], ignore_errors=True)
    return (
        None, None,                              # file_input, demo_file
        None,                                    # preview image
        "Waiting for results...",                # info
        "## Upload a file and click **Parse**.", # md rendered
        "",                                      # md raw
        gr.update(visible=False),                # download btn
        "0 / 0",                                 # page info
        "",                                      # json tab
        make_session(),                          # state
    )


def on_prompt_change(prompt_mode):
    is_free = prompt_mode == "prompt_general"
    text = "" if is_free else dict_promptmode_to_prompt[prompt_mode]
    return gr.update(value=text, interactive=is_free)


def on_demo_select(path, state):
    if not path:
        return None, "0 / 0", state
    return load_file(path, state)


def check_connection(server_url):
    if not server_url or not server_url.strip():
        return _conn_html("gray", "— Chưa nhập URL")
    url = server_url.strip().rstrip("/")
    try:
        r = _requests.get(f"{url}/v1/models", timeout=5)
        if r.status_code == 200:
            models = [m.get("id", "?") for m in r.json().get("data", [])]
            label = ", ".join(models) if models else "connected"
            return _conn_html("green", f"✅ Đã kết nối — model: {label}")
        return _conn_html("orange", f"⚠️ HTTP {r.status_code}")
    except Exception as e:
        msg = str(e)
        if "Connection refused" in msg or "connect" in msg.lower():
            return _conn_html("red", "❌ Server chưa khởi động hoặc URL sai")
        return _conn_html("red", f"❌ {msg[:80]}")


def _conn_html(color, text):
    colors = {"green": "#1a7a2e", "red": "#c0392b", "orange": "#d35400", "gray": "#666"}
    bg = {"green": "#d4edda", "red": "#fde8e8", "orange": "#fef3cd", "gray": "#f0f0f0"}
    c, b = colors.get(color, "#666"), bg.get(color, "#f0f0f0")
    return f'<div style="padding:6px 10px;border-radius:6px;background:{b};color:{c};font-size:13px">{text}</div>'


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
#parse-btn { background: #e63946 !important; border-color: #e63946 !important; }
#parse-btn:hover { background: #c1121f !important; }
#page-nav { text-align: center; font-size: 15px; padding: 6px 18px;
            border: 1px solid #ccc; border-radius: 6px; background: #f8f8f8; }
footer { visibility: hidden; }
"""

with gr.Blocks(title="dots.ocr") as demo:
    state = gr.State(make_session())

    gr.HTML("""
        <div style="text-align:center; padding: 12px 0 4px">
            <h1 style="margin:0; font-size:1.9em">🔍 dots.ocr</h1>
            <p style="margin:4px 0; color:#666"><em>Multilingual Document Layout Parsing</em></p>
        </div>
    """)

    with gr.Row():
        # ── Left panel ────────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### 📂 Input")
            file_input = gr.File(
                label="Upload image / PDF",
                file_types=[".jpg", ".jpeg", ".png", ".pdf"],
                type="filepath",
            )
            demo_file = gr.Dropdown(
                label="Or pick a demo image",
                choices=[""] + [str(p) for p in DEMO_IMAGES],
                value="",
            )

            gr.Markdown("### ⚙️ Settings")
            prompt_mode = gr.Dropdown(
                label="Task / Prompt",
                choices=list(PROMPT_LABELS.keys()),
                value="prompt_layout_all_en",
            )
            custom_prompt = gr.Textbox(
                label="Custom prompt (Free QA mode only)",
                placeholder="Type your question here...",
                lines=3,
                interactive=False,
            )

            with gr.Row():
                parse_btn = gr.Button("🔍 Parse", variant="primary", elem_id="parse-btn", scale=3)
                clear_btn = gr.Button("🗑️ Clear", scale=1)

            with gr.Accordion("🌐 Server Config", open=True):
                server_url = gr.Textbox(
                    label="Server URL",
                    value="http://127.0.0.1:8000",
                    info="Local: http://127.0.0.1:8000  |  Kaggle/ngrok: https://xxxx.ngrok-free.app",
                )
                model_name = gr.Textbox(label="Model name", value="model")
                conn_status = gr.HTML(_conn_html("gray", "— Chưa kiểm tra"))
                test_conn_btn = gr.Button("🔌 Test Connection", size="sm")

            with gr.Accordion("🔧 Advanced", open=False):
                dpi = gr.Slider(
                    label="PDF render DPI",
                    minimum=48, maximum=200, step=1, value=150,
                    info="Giảm DPI nếu bị lỗi bộ nhớ (150 → 100 → 72 → 48)",
                )
                with gr.Row():
                    page_from = gr.Number(label="Trang bắt đầu", value=1, precision=0, minimum=1,
                                          info="Trang đầu cần parse")
                    page_to   = gr.Number(label="Trang kết thúc", value=10, precision=0, minimum=1,
                                          info="Để trống = hết file")
                min_px = gr.Number(label="Min pixels", value=MIN_PIXELS, precision=0)
                max_px = gr.Number(label="Max pixels", value=MAX_PIXELS, precision=0)

        # ── Right panel ───────────────────────────────────────────────────────
        with gr.Column(scale=5):
            with gr.Row(equal_height=True):

                # Preview column
                with gr.Column(scale=1):
                    gr.Markdown("### 🖼️ Preview")
                    preview_img = gr.Image(
                        label="Layout Preview",
                        show_label=False,
                        height=620,
                    )
                    with gr.Row():
                        prev_btn  = gr.Button("◀ Prev", size="sm")
                        page_info = gr.HTML('<span id="page-nav">0 / 0</span>')
                        next_btn  = gr.Button("Next ▶", size="sm")
                    info_md = gr.Markdown("Waiting for results...")

                # Result column
                with gr.Column(scale=1):
                    gr.Markdown("### ✅ Results")
                    with gr.Tabs():
                        with gr.TabItem("Rendered Markdown"):
                            md_rendered = gr.Markdown(
                                "## Upload a file and click **Parse**.",
                                latex_delimiters=[
                                    {"left": "$$", "right": "$$", "display": True},
                                    {"left": "$",  "right": "$",  "display": False},
                                ],
                                height=580,
                            )
                        with gr.TabItem("Raw Markdown"):
                            md_raw = gr.Textbox(
                                label="Raw output",
                                show_label=False,
                                lines=30,
                                max_lines=60,
                            )
                        with gr.TabItem("JSON"):
                            json_out = gr.Textbox(
                                label="Layout JSON",
                                show_label=False,
                                lines=30,
                                max_lines=60,
                            )

            download_btn = gr.DownloadButton("⬇️ Download Results", visible=False)

    # ── Event wiring ──────────────────────────────────────────────────────────

    prompt_mode.change(on_prompt_change, inputs=prompt_mode, outputs=custom_prompt)

    test_conn_btn.click(check_connection, inputs=server_url, outputs=conn_status)
    server_url.change(check_connection, inputs=server_url, outputs=conn_status)
    demo.load(check_connection, inputs=server_url, outputs=conn_status)

    file_input.upload(load_file, inputs=[file_input, state],
                      outputs=[preview_img, page_info, state])

    demo_file.change(on_demo_select, inputs=[demo_file, state],
                     outputs=[preview_img, page_info, state])

    prev_btn.click(lambda s: navigate("prev", s), inputs=state,
                   outputs=[preview_img, page_info, json_out, state])
    next_btn.click(lambda s: navigate("next", s), inputs=state,
                   outputs=[preview_img, page_info, json_out, state])

    parse_btn.click(
        run_parse,
        inputs=[file_input, demo_file, prompt_mode, custom_prompt,
                server_url, model_name, dpi, page_from, page_to, min_px, max_px, state],
        outputs=[preview_img, info_md, md_rendered, md_raw,
                 download_btn, page_info, json_out, state],
    )

    clear_btn.click(
        clear_all, inputs=state,
        outputs=[file_input, demo_file, preview_img, info_md,
                 md_rendered, md_raw, download_btn, page_info, json_out, state],
    )

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    print(f"\n✅  dots.ocr UI ready → http://localhost:{args.port}\n")
    demo.queue().launch(server_name=args.host, server_port=args.port,
                        share=args.share, show_error=True,
                        theme=gr.themes.Ocean(), css=CSS)
