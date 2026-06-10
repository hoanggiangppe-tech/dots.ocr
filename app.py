"""
dots.ocr - Local Web Interface
Connects to a vLLM server for document/image OCR and layout parsing.
"""

import hashlib
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
import fitz as _fitz
from dots_ocr.utils.doc_utils import load_images_from_pdf, fitz_doc_to_image as _fitz_to_img
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

RESULTS_DIR = Path.home() / ".dots_ocr_manager" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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


def _stable_output_dir(file_path: str, prompt_mode: str, start_page: int) -> Path:
    key = hashlib.md5(
        f"{os.path.abspath(file_path)}:{prompt_mode}:{start_page}".encode()
    ).hexdigest()[:10]
    d = RESULTS_DIR / f"{Path(file_path).stem}_{key}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_checkpoint(stable_dir: Path) -> dict:
    cp = stable_dir / "checkpoint.json"
    if cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_page_checkpoint(stable_dir: Path, checkpoint: dict, page_key: str, result: dict):
    serializable = {k: v for k, v in result.items()
                    if isinstance(v, (str, int, float, bool, type(None)))}
    checkpoint[page_key] = serializable
    (stable_dir / "checkpoint.json").write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_file(file_path, state):
    if not file_path or not os.path.exists(file_path):
        return None, "0 / 0", state

    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        with _fitz.open(file_path) as _doc:
            total = _doc.page_count
            first_img = _fitz_to_img(_doc[0], target_dpi=100)
        # Lazy placeholders — pages loaded on demand to avoid OOM on large PDFs
        state["pages"] = [first_img] + [None] * (total - 1)
        state["pdf_path"] = file_path
    elif ext in {".jpg", ".jpeg", ".png"}:
        first_img = Image.open(file_path).convert("RGB")
        state["pages"] = [first_img]
        state["pdf_path"] = None
    else:
        return None, "Unsupported format", state

    state["page_idx"] = 0
    state["is_parsed"] = False
    state["parsed_pages"] = []
    total = len(state["pages"])
    return first_img, f"1 / {total}", state


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
    # Lazy-load from PDF if placeholder (avoids loading all pages at once)
    if img is None and state.get("pdf_path"):
        with _fitz.open(state["pdf_path"]) as _doc:
            img = _fitz_to_img(_doc[idx], target_dpi=100)
        pages[idx] = img  # cache for repeat navigation

    if state["is_parsed"] and idx < len(state["parsed_pages"]):
        r = state["parsed_pages"][idx]
        if r.get("layout_image"):
            img = r["layout_image"]
        json_txt = json.dumps(r.get("cells_data", []), ensure_ascii=False, indent=2)
    else:
        json_txt = ""

    return img, f"{idx+1} / {len(pages)}", json_txt, state


def run_parse(file_input, demo_file, prompt_mode, custom_prompt,
              server_url, model_name, dpi, page_from, page_to, min_px, max_px, state,
              progress=gr.Progress()):
    """Generator: yields partial info_md updates each page, then final full result."""

    _NO_CHANGE = gr.update()  # sentinel: leave component unchanged

    path = file_input or demo_file
    if not path or not os.path.exists(str(path)):
        yield (
            None, "⚠️ Please upload a file or select a demo image.",
            "", "", gr.update(visible=False), "0 / 0", "", state,
        )
        return

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

    stable_dir = None  # set below for PDFs
    results = []       # populated inside try; readable in except for partial display

    try:
        parser = build_parser(server_url, model_name, temperature, dpi)
        parser.min_pixels = int(min_px) if min_px else None
        parser.max_pixels = int(max_px) if max_px else None

        ext = Path(path).suffix.lower()
        fname = Path(path).stem

        if ext == ".pdf":
            start = max(0, int(page_from) - 1)
            with _fitz.open(path) as _count_doc:
                end = (int(page_to) - 1) if page_to else (_count_doc.page_count - 1)
                total_pages = end - start + 1

            # Use stable dir so checkpoint survives app restarts
            stable_dir = _stable_output_dir(path, prompt_mode, start)
            checkpoint = _load_checkpoint(stable_dir)
            skipped_cp = 0
            results = []

            # Open PDF once and load one page at a time — avoids OOM on large PDFs
            with _fitz.open(path) as _pdf:
                for i in range(total_pages):
                    page_no = start + i
                    page_key = str(page_no)

                    # Update progress bar
                    try:
                        progress(i / total_pages, desc=f"Trang {page_no + 1} / {end + 1}")
                    except Exception:
                        pass

                    # Resume from checkpoint only if page result file exists AND has content
                    if page_key in checkpoint:
                        cp_r = checkpoint[page_key]
                        md_p = cp_r.get("md_content_path") or cp_r.get("md_content_nohf_path")
                        if md_p and os.path.exists(md_p) and os.path.getsize(md_p) > 0:
                            results.append(cp_r)
                            skipped_cp += 1
                            yield (
                                _NO_CHANGE,
                                f"⏩ Trang **{page_no + 1} / {end + 1}** — từ cache ✅"
                                f"&nbsp;&nbsp;`{i + 1}/{total_pages} xong`",
                                _NO_CHANGE, _NO_CHANGE, _NO_CHANGE, _NO_CHANGE, _NO_CHANGE, state,
                            )
                            continue

                    # Live status before OCR starts on this page
                    yield (
                        _NO_CHANGE,
                        f"⏳ Đang OCR trang **{page_no + 1} / {end + 1}**…"
                        f"&nbsp;&nbsp;`{i}/{total_pages} hoàn thành`",
                        _NO_CHANGE, _NO_CHANGE, _NO_CHANGE, _NO_CHANGE, _NO_CHANGE, state,
                    )

                    # Load single page on demand
                    img = _fitz_to_img(_pdf[page_no], target_dpi=int(dpi))

                    # Retry up to 3x with backoff (handles transient ngrok errors)
                    last_err = None
                    for attempt in range(3):
                        try:
                            r = parser._parse_single_image(
                                img, prompt_mode, str(stable_dir),
                                fname, source="pdf", page_idx=page_no,
                            )
                            r["file_path"] = path
                            _save_page_checkpoint(stable_dir, checkpoint, page_key, r)
                            results.append(r)
                            last_err = None
                            break
                        except Exception as e:
                            last_err = e
                            if attempt < 2:
                                yield (
                                    _NO_CHANGE,
                                    f"🔄 Trang **{page_no + 1}** lỗi lần {attempt + 1}/3 — thử lại sau {10*(attempt+1)}s…",
                                    _NO_CHANGE, _NO_CHANGE, _NO_CHANGE, _NO_CHANGE, _NO_CHANGE, state,
                                )
                                time.sleep(10 * (attempt + 1))

                    del img  # free page memory immediately

                    if last_err is not None:
                        done = len(results)
                        raise RuntimeError(
                            f"❌ Trang {page_no + 1} thất bại sau 3 lần thử: {last_err}\n\n"
                            f"Đã xử lý **{done}/{total_pages} trang** — kết quả đã lưu.\n"
                            f"▶ Chạy lại với cùng file + cùng cài đặt để **tiếp tục từ trang {page_no + 1}**."
                        )

            try:
                progress(1.0, desc="Hoàn thành!")
            except Exception:
                pass

        else:
            yield (
                _NO_CHANGE,
                "⏳ Đang xử lý ảnh…",
                _NO_CHANGE, _NO_CHANGE, _NO_CHANGE, _NO_CHANGE, _NO_CHANGE, state,
            )
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

        # Prefer layout_image; fall back to lazy placeholder (navigate will load on demand)
        state["pages"] = [p.get("layout_image") for p in parsed_pages]
        state["page_idx"] = 0
        state["is_parsed"] = True
        state["parsed_pages"] = parsed_pages

        combined_md = "\n\n---\n\n".join(all_md)

        if not combined_md.strip():
            yield (
                None,
                "⚠️ **Server trả về nội dung rỗng.**\n\n"
                "Kiểm tra:\n"
                "1. Model name trong **Server Config** phải là `model`\n"
                "2. vLLM đã sẵn sàng chưa (Cell 4b log `✅ vLLM sẵn sàng`)\n"
                "3. Thử **🔌 Test Connection** — nếu xanh thì server OK\n"
                "4. Chạy lại Cell 4b trên Kaggle nếu server mới restart",
                "", "", gr.update(visible=False), "0 / 0", "", state,
            )
            return

        first = parsed_pages[0]
        first_img = first["layout_image"] or (Image.open(path) if ext != ".pdf" else state["pages"][0])
        first_json = json.dumps(first.get("cells_data") or [], ensure_ascii=False, indent=2)

        page_range_str = f"trang {int(page_from)}–{int(page_to)}" if page_to else f"từ trang {int(page_from)}"
        resume_note = (
            f"  \n**Resume:** {skipped_cp} trang từ cache, {len(results) - skipped_cp} trang mới xử lý"
            if ext == ".pdf" and skipped_cp > 0 else ""
        )
        info = (
            f"**File:** `{Path(path).name}`  \n"
            f"**Pages parsed:** {len(results)} ({page_range_str})  |  "
            f"**Elements:** {len(all_cells)}{resume_note}  \n"
            f"**Model:** `{model_name}` @ `{server_url}`  |  "
            f"**Prompt:** `{prompt_mode}`"
        )

        # For PDFs use stable_dir (persistent), for images use temp_dir
        zip_source = str(stable_dir) if ext == ".pdf" else temp_dir
        zip_path = os.path.join(temp_dir, f"results_{uuid.uuid4().hex[:6]}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(zip_source):
                for fn in files:
                    if fn == "checkpoint.json" or fn.endswith(".zip"):
                        continue
                    full = os.path.join(root, fn)
                    zf.write(full, os.path.relpath(full, zip_source))

        yield (
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

        # Collect partial results from pages that were already successfully parsed
        partial_parsed = []
        partial_md_parts = []
        for r in results:
            pr = {"layout_image": None, "cells_data": None, "md": ""}
            if r.get("layout_image_path") and os.path.exists(r["layout_image_path"]):
                try:
                    pr["layout_image"] = Image.open(r["layout_image_path"])
                except Exception:
                    pass
            md_path = r.get("md_content_path") or r.get("md_content_nohf_path")
            if md_path and os.path.exists(md_path):
                try:
                    with open(md_path, encoding="utf-8") as f:
                        pr["md"] = f.read()
                    partial_md_parts.append(pr["md"])
                except Exception:
                    pass
            partial_parsed.append(pr)

        partial_md = "\n\n---\n\n".join(partial_md_parts)

        if partial_md:
            state["pages"] = [p.get("layout_image") for p in partial_parsed]
            state["page_idx"] = 0
            state["is_parsed"] = True
            state["parsed_pages"] = partial_parsed
            first_img = partial_parsed[0].get("layout_image") if partial_parsed else None
            full_msg = (
                f"{msg}\n\n"
                f"---\n\n"
                f"**Kết quả {len(partial_parsed)} trang đã xử lý trước khi bị ngắt:**"
            )
            yield (
                first_img, full_msg, partial_md, partial_md,
                gr.update(visible=False),
                f"1 / {len(partial_parsed)}", "", state,
            )
        else:
            yield (None, msg, "", "", gr.update(visible=False), "0 / 0", "", state)
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
