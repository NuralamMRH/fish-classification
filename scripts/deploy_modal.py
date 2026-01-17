import os
import base64
import json
import modal

# Build an image with system and Python deps needed by OpenCV/Torch/Flask-based pipeline
image = modal.Image.debian_slim().apt_install(
    [
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
    ]
).pip_install(
    [
        "opencv-python>=4.1.2",
        "numpy>=1.15",
        "pycocotools>=2.0.2",
        "pillow>=7.1.1",
        "Shapely",
        "pyyaml>=5.1",
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "flask>=2.0.0",
        "flask-cors>=4.0.0",
        "requests>=2.25.0",
        "mediapipe>=0.10.0",
        "fastapi[standard]>=0.115.0",
    ]
)

# Mount your entire project into the container
PROJECT_ROOT_LOCAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT_REMOTE = "/root/project"
# Add specific subdirectories to avoid .git changes triggering build errors
image = image.add_local_dir(os.path.join(PROJECT_ROOT_LOCAL, "scripts"), remote_path=os.path.join(PROJECT_ROOT_REMOTE, "scripts"))
image = image.add_local_dir(os.path.join(PROJECT_ROOT_LOCAL, "models"), remote_path=os.path.join(PROJECT_ROOT_REMOTE, "models"))
image = image.add_local_dir(os.path.join(PROJECT_ROOT_LOCAL, "static"), remote_path=os.path.join(PROJECT_ROOT_REMOTE, "static"))
app = modal.App("fish-classification-web")

HTML_PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Fish Classification (Modal)</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 40px; }
    .card { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-top: 20px; }
    img { max-width: 100%; border-radius: 8px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .fish { background: white; border-left: 4px solid #4caf50; padding: 12px; border-radius: 8px; margin: 10px 0; }
    label { display: inline-block; margin-right: 12px; }
  </style>
</head>
<body>
  <h1>🐟 Fish Classification (Modal)</h1>
  <p>Upload a fish image; results and images are rendered below.</p>
  <div class="card">
    <form id="form">
      <input type="file" id="file" accept=".png,.jpg,.jpeg,.gif,.bmp" required>
      <label>Pixels/cm <input type="number" step="0.0001" id="ppc" placeholder=""></label>
      <label>Length Type
        <select id="lengthType">
          <option value="AUTO" selected>AUTO</option>
          <option value="TL">TL</option>
          <option value="FL">FL</option>
          <option value="LJFL">LJFL</option>
        </select>
      </label>
      <label>Girth Factor <input type="number" step="0.0001" id="girthFactor" value="3.1416"></label>
      <button type="submit">Analyze</button>
    </form>
  </div>
  <div id="out"></div>

<script>
async function fileToBase64(file) {
  const buf = await file.arrayBuffer();
  const bytes = new Uint8Array(buf);
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

document.getElementById('form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = document.getElementById('file').files[0];
  const ppc = document.getElementById('ppc').value;
  const lengthType = document.getElementById('lengthType').value;
  const girthFactor = document.getElementById('girthFactor').value;
  if (!file) return;

  const b64 = await fileToBase64(file);
  const payload = {
    image_b64: b64,
    filename: file.name,
    pixels_per_cm: ppc || null,
    length_type: lengthType || "AUTO",
    girth_factor: girthFactor || 3.1416
  };
  
  const out = document.getElementById('out');
  out.innerHTML = '<div class="card"><h3>⏳ Processing...</h3></div>';

  try {
      const res = await fetch(window.location.href, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      if (!data.success) {
        out.innerHTML = '<div class="card"><h3>❌ Error</h3><pre>' + (data.error || 'Unknown error') + '</pre></div>';
        return;
      }

      let html = '<div class="card"><h2>🎯 Detection Complete</h2><p>' + data.fish_count + ' fish detected.</p></div>';
      html += '<div class="grid">';
      if (data.annotated_image) {
        html += '<div><h3>📷 Visual Detection Results</h3><img src="/static/' + data.annotated_image + '"></div>';
      }
      html += '<div><h3>📊 Detailed Identification Results</h3>';
      for (const fish of data.fish) {
        html += '<div class="fish">';
        html += '<div><strong>🐠 Fish #' + fish.fish_id + '</strong></div>';
        html += '<div><strong>Species:</strong> ' + (fish.common_name || fish.species) + (fish.scientific_name ? ' (' + fish.scientific_name + ')' : '') + '</div>';
        html += '<div><strong>Accuracy:</strong> ' + (fish.accuracy * 100).toFixed(1) + '%</div>';
        html += '<div><strong>Confidence:</strong> ' + (fish.confidence * 100).toFixed(1) + '%</div>';
        html += '<div><strong>Box:</strong> [' + fish.box.join(', ') + ']</div>';
        if (fish.measurements) {
          html += '<div><strong>Length Type:</strong> ' + fish.measurements.length_type + '</div>';
          html += '<div><strong>Length:</strong> ' + fish.measurements.length_px + ' px' + (fish.measurements.length_cm ? ' (' + fish.measurements.length_cm + ' cm)' : '') + '</div>';
          html += '<div><strong>Girth:</strong> ' + fish.measurements.girth_px + ' px' + (fish.measurements.girth_cm ? ' (' + fish.measurements.girth_cm + ' cm)' : '') + '</div>';
        }
        if (fish.crop_mask_image) {
          html += '<div style="margin-top:8px;"><strong>Masked Crop:</strong><br><img src="/static/' + fish.crop_mask_image + '"></div>';
        }
        html += '</div>';
      }
      html += '</div></div>';
      out.innerHTML = html;
  } catch (err) {
      out.innerHTML = '<div class="card"><h3>❌ Connection Error</h3><pre>' + err + '</pre></div>';
  }
});
</script>
</body>
</html>
"""

@app.function(
    image=image,
    timeout=600,
    min_containers=1,
    max_containers=4,
    gpu="T4",
    memory=4096,
    scaledown_window=120
)
@modal.asgi_app()
def fastapi_app():
    import os
    import base64
    import sys
    from math import pi
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, FileResponse

    sys.path.insert(0, os.path.join(PROJECT_ROOT_REMOTE, "scripts"))
    import web_app

    app = Starlette()

    @app.route("/static/{path:path}", methods=["GET"])
    async def static_files(request: Request):
        path = request.path_params.get("path") or ""
        full_path = os.path.join(web_app.STATIC_FOLDER, path)
        if os.path.exists(full_path):
            return FileResponse(full_path)
        return JSONResponse({"error": "Not found"}, status_code=404)

    @app.route("/api", methods=["POST"])
    async def post_root(request: Request):
        try:
            body = await request.json()
            image_b64 = body.get("image_b64")
            filename = body.get("filename") or "upload.jpg"
            pixels_per_cm = body.get("pixels_per_cm")
            length_type = body.get("length_type") or "AUTO"
            girth_factor = float(body.get("girth_factor") or pi)

            if not image_b64:
                return JSONResponse(status_code=400, content={"success": False, "error": "Missing image_b64"})

            upload_dir = web_app.UPLOAD_FOLDER
            os.makedirs(upload_dir, exist_ok=True)
            tmp_path = os.path.join(upload_dir, filename)
            with open(tmp_path, "wb") as f:
                f.write(base64.b64decode(image_b64))

            results = web_app.process_image_external(
                tmp_path,
                pixels_per_cm=pixels_per_cm if pixels_per_cm else None,
                length_type=length_type,
                girth_factor=girth_factor,
            )

            try:
                os.unlink(tmp_path)
            except Exception:
                pass

            if not results or not results.get("success"):
                return JSONResponse(results or {"success": False, "error": "Unknown processing error"})
            def file_to_b64(rel_path):
                try:
                    full_path = os.path.join(web_app.STATIC_FOLDER, rel_path)
                    if os.path.exists(full_path):
                        with open(full_path, "rb") as f:
                            import base64 as _b
                            return _b.b64encode(f.read()).decode("utf-8")
                except Exception:
                    pass
                return None
            resp = dict(results)
            if results.get("annotated_image"):
                b64 = file_to_b64(results["annotated_image"])
                if b64:
                    resp["annotated_image_b64"] = b64
            fish_out = []
            for fi in results.get("fish", []):
                item = dict(fi)
                cm = fi.get("crop_mask_image")
                if cm:
                    cm_b64 = file_to_b64(cm)
                    if cm_b64:
                        item["crop_mask_image_b64"] = cm_b64
                fish_out.append(item)
            resp["fish"] = fish_out
            return JSONResponse(resp)
        except Exception as e:
            return JSONResponse(status_code=500, content={"success": False, "error": f"Server error: {str(e)}"})

    return app
