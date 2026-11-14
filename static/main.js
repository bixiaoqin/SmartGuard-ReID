// SmartGuard-ReID 前端交互：抓帧 / 点击框选 / 检索展示
let currentKey = null;
let currentBoxes = [];

function setStatus(text, sticky) {
  const el = document.getElementById("status");
  el.textContent = text;
  el.classList.remove("hidden");
  if (!sticky) setTimeout(() => el.classList.add("hidden"), 4000);
}

async function grabFrame() {
  const video = document.getElementById("videoSel").value;
  if (!video) { alert("没有可用的视频源"); return; }
  const seek = parseFloat(document.getElementById("seek").value || 0);
  setStatus("正在扫描视频中带行人的画面，请稍候（CPU 约 10~60 秒）...", true);
  try {
    const resp = await fetch("/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video, seek })
    });
    const data = await resp.json();
    if (data.error) { setStatus("出错: " + data.error); return; }
    currentKey = data.key;
    currentBoxes = data.boxes;
    const img = document.getElementById("frameImg");
    img.src = data.image + "?t=" + Date.now();
    img.onload = () => { img.dataset.w = img.naturalWidth; img.dataset.h = img.naturalHeight; };
    renderBoxButtons();
    document.getElementById("frameArea").classList.remove("hidden");
    setStatus(`第 ${data.frame_no} 帧检测到 ${data.boxes.length} 个行人，点击画面中的目标`);
  } catch (e) {
    setStatus("请求失败: " + e);
  }
}

function renderBoxButtons() {
  const wrap = document.getElementById("boxBtns");
  wrap.innerHTML = "";
  currentBoxes.forEach(b => {
    const btn = document.createElement("button");
    btn.textContent = `选择行人 ${b.index + 1}（${b.conf}）`;
    btn.onclick = () => selectPerson(b.bbox);
    wrap.appendChild(btn);
  });
}

// 点击画面 → 换算成原图坐标 → 找到包含该点的检测框 → 发给后端
function onFrameClick(ev) {
  const img = document.getElementById("frameImg");
  const rect = img.getBoundingClientRect();
  const scaleX = img.dataset.w / rect.width;
  const scaleY = img.dataset.h / rect.height;
  const x = (ev.clientX - rect.left) * scaleX;
  const y = (ev.clientY - rect.top) * scaleY;
  const hit = currentBoxes.find(b => {
    const [x1, y1, x2, y2] = b.bbox;
    return x >= x1 && x <= x2 && y >= y1 && y <= y2;
  });
  if (!hit) { setStatus("点击位置不在任何检测框内，请点在行人的框上"); return; }
  selectPerson(hit.bbox);
}

async function selectPerson(bbox) {
  setStatus("正在跨镜头检索（提取特征 + 比对图库）...", true);
  try {
    const resp = await fetch("/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key: currentKey, bbox })
    });
    const data = await resp.json();
    showResult(data);
  } catch (e) { setStatus("请求失败: " + e); }
}

async function uploadQuery(ev) {
  ev.preventDefault();
  const file = document.getElementById("imgFile").files[0];
  if (!file) return false;
  const form = new FormData();
  form.append("image", file);
  setStatus("正在检索上传图片中的行人...", true);
  try {
    const resp = await fetch("/upload", { method: "POST", body: form });
    const data = await resp.json();
    showResult(data);
  } catch (e) { setStatus("请求失败: " + e); }
  return false;
}

function showResult(data) {
  if (data.error) { setStatus("出错: " + data.error); return; }
  document.getElementById("rankImg").src = data.rank_list + "?t=" + Date.now();
  const tbody = document.querySelector("#resultTable tbody");
  tbody.innerHTML = "";
  data.results.forEach(r => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>#${r.rank}</td><td>${r.similarity}</td><td>${r.file}</td>`;
    tbody.appendChild(tr);
  });
  document.getElementById("resultArea").classList.remove("hidden");
  setStatus(`完成！Top-1 相似度 ${data.results[0].similarity}`);
}
