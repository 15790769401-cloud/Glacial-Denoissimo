const el = document.querySelector('.title');
const text = '基于Transformer的相机图像降噪系统';
let i = 0;
const timer = setInterval(() => {
  el.textContent = text.slice(0, ++i);
  if (i >= text.length) clearInterval(timer);
}, 120);

const text1 = document.querySelector('.text1');
const target1 = '为Canan, Sony, Nikon*定制的降噪大模型';
const text2 = document.querySelector('.text2');
const target2 = '以超棒的自训练算法，打造极佳的降噪效果';
const chars = '为定制的降噪大模型以超棒的自训练算法打造极佳的降噪效果';
let frame1 = 0;
setTimeout(function () {
  const interval = setInterval(function () {
    const process = target1.split('')
    const texta = document.querySelector('.text1');
    texta.style.color = '#fff';
    const content = process.map(function (c, i) {
      return i < frame1 / 2 ? c : chars[Math.floor(Math.random() * chars.length)];
    }).join('');
    text1.textContent = content
    frame1++;
    if (frame1 > target1.length * 2) clearInterval(interval);
}, 40);
}, 3000);
let frame2 = 0;
setTimeout(function () {
  const interval = setInterval(function () {
    const process = target2.split('')
    const content = process.map(function (c, i) {
      return i < frame2 / 2 ? c : chars[Math.floor(Math.random() * chars.length)];
    }).join('');
    const texta = document.querySelector('.text2');
    texta.style.color = '#fff';
    text2.textContent = content
    frame2++;
    if (frame2 > target2.length * 2) clearInterval(interval);
}, 40);
}, 3000);

let sliderPosition = 50;
let isDragging = false;

const photoContainer = document.querySelector('.photo-container');
const sliderComponent = document.querySelector('.slider-component');
const sliderHandle = document.querySelector('#sliderHandle');
const leftImage = document.querySelector('.left-image');
const rightImage = document.querySelector('.right-image');

sliderComponent.addEventListener('mousedown', (e) => {
  e.preventDefault();
  isDragging = true;
  document.addEventListener("mousemove", handleMouseMove);
  document.addEventListener("mouseup", upMouseMove);
  sliderHandle.classList.add('active');
})

function upMouseMove(){
  if (!isDragging) return;
  isDragging = false;
  document.removeEventListener('mousemove', handleMouseMove);
  document.removeEventListener('mouseup', upMouseMove);
  sliderHandle.classList.remove('active');
}

function handleMouseMove(e){
  let photoContainer = document.querySelector(".photo-container");
  let container = photoContainer.getBoundingClientRect();
  if (!isDragging || !container) return;
  let position = [(e.clientX - container.left) / (container.right - container.left)] * 100
  if (0<= position && position <=100){
      updateSliderPosition(position)
    }
  sliderHandle.classList.add('active');
}

function updateSliderPosition(pos) {
  sliderComponent.style.left = `${pos}%`;
  leftImage.style.clipPath = `inset(0 ${100 - pos}% 0 0)`;
  rightImage.style.clipPath = `inset(0 0 0 ${pos}%)`;
}

const container = document.querySelector('.container');
const cardImages = document.querySelectorAll('.choices-card1 .img, .choices-card2 .img, .choices-card3 .img');
const cards = document.querySelectorAll('.choices-card1, .choices-card2, .choices-card3')
const logo1 = document.querySelector('.logo1');
const uploadContainer = document.querySelector('.upload-container');
const hdText2 = document.querySelector('.hd-text2');
const title3 = document.querySelector('.show1 h3');
const btns = document.querySelector('.btns');
const title2 = document.querySelector('.show1 h2');
const textarea = document.querySelector('.textarea');
const show2 = document.querySelector('.show2');
let typewriterStarted = false;

const textAreaData = [
  {
    title: '10档ISO采样，轻松面对暗光',
    desc: '针对暗光环境下的高ISO噪点，系统采用多帧合成与深度学习降噪算法，在保留画面细节的同时有效抑制亮度噪点与色彩噪点，让夜景与弱光拍摄也能获得纯净细腻的画质。'
  },
  {
    title: '智能识别室内光线，还原真实色彩',
    desc: '室内光线复杂多变，系统可智能识别场景并自适应调整降噪强度，还原真实色彩与层次，让人像肤色自然、物体质感清晰，即使灯光昏暗也能轻松出片。'
  },
  {
    title: '长时间曝光降噪，消除热噪与坏点',
    desc: '长时间曝光容易产生热噪点与固定模式噪声，系统通过暗帧校准与AI降噪技术，精准去除热噪与坏点，让星空、车流与夜景长曝画面更加通透纯净。'
  },
  {
    title: '压缩不失真，保持细腻画质',
    desc: '在压缩与传输过程中画质难免受损，系统利用超分辨率与细节修复技术，补偿压缩伪影与细节损失，让照片在缩小体积的同时依然保持清晰锐利的观感。'
  }
];

const sceneBtns = document.querySelectorAll('.btns > *');
const txHeader = document.querySelector('.textarea .tx-header');
const txText = document.querySelector('.textarea .text');

window.addEventListener('scroll', () => {
  const scrollTop = document.documentElement.scrollTop;

  if (scrollTop > 0 && scrollTop < window.innerHeight) {
    const headerInner = document.querySelector('.header-inner');
    const header = document.querySelector('.header');
    headerInner.style.height = `calc(200px - ${100*(scrollTop / window.innerHeight)}px)`;
    header.style.height  = `calc(200px - ${100*(scrollTop / window.innerHeight)}px)`;
    const choicesNon = document.querySelector('.choices-non');
    const progress = scrollTop / (window.innerHeight*0.8);
    const cardOpacity = 1 - Math.min(Math.max((progress - 0.2) / 0.1, 0), 1);
    cardImages.forEach(img => { img.style.opacity = cardOpacity; });
    const logoOpacity = 1 - Math.min(Math.max((progress - 0.4) / 0.1, 0), 1);
    logo1.style.opacity = logoOpacity;
    const subtitleOpacity = 1 - Math.min(Math.max((progress - 0.1) / 0.1, 0), 1);
    hdText2.style.opacity = subtitleOpacity;
    hdText2.style.maxHeight = `${subtitleOpacity * 40}px`;

    if (scrollTop > window.innerHeight*0.4) {
      logo1.style.position = 'fixed';
    }
    else {
      logo1.style.position = 'static';
    }
    choicesNon.style.opacity = cardOpacity;
    const cardScale = Math.round((1 - progress / 2) * 10000) / 10000;
    cards.forEach(card => { card.style.height = `calc(var(--card-size) * ${cardScale})`; });
    const logo2 = document.querySelector('.logo2');
    logo2.style.marginTop = `${-5 - 10 * progress }px`;
    const uploadbtn = document.querySelector('.uploadbtn');
    const uploadOpacity = Math.min(Math.max((progress - 0.6) / 0.1, 0), 1);
    uploadbtn.style.opacity = uploadOpacity;
  }
  if (scrollTop > window.innerHeight*0.6 && !typewriterStarted) {
    title3.classList.add('fade-in-up');
    textarea.classList.add('fade-in-up');
    btns.classList.add('fade-in-up');
    typewriterStarted = true;
    const textb = '轻松为心爱的照片降噪';
    let k = 0;
    title2.textContent = '';
    const timer2 = setInterval(() => {
      title2.textContent = textb.slice(0, ++k);
      if (k >= textb.length) clearInterval(timer2);
    }, 120);
  }else if (scrollTop < window.innerHeight*0.1) {
    typewriterStarted = false;
    title2.textContent = '';
    title3.classList.remove('fade-in-up');
    textarea.classList.remove('fade-in-up');
    btns.classList.remove('fade-in-up');
  }
  if (scrollTop > window.innerHeight *1.1) {
    show2.classList.add('fade-in-up2');
  }
  else {
    show2.classList.remove('fade-in-up2');
  }
})

let currentIndex = 0
let Carousel = null

function renderScene(pos){
  const data = textAreaData[pos];
  if (!data) return;
  txHeader.textContent = data.title;
  txText.textContent = data.desc;
  currentIndex = pos;
  sceneBtns.forEach((btn, i) => {
    btn.classList.toggle('active', i === pos);
  });
}

function Auto(){
  setTimeout(() => {
  if (Carousel) clearInterval(Carousel)
    Carousel = setInterval(() => 
    {
      renderScene((currentIndex + 1) % textAreaData.length)
    }, 3000)
  }, 1000)
}

sceneBtns.forEach((btn, i) => {
  btn.addEventListener('click', () => {
    renderScene(i);
    Auto();
  });
});

renderScene(currentIndex)
Auto()

const fileInput = document.querySelector('.file-input');
const closeBtn = document.querySelector('.close-btn');
const uploadIcon = document.querySelector('.upload-icon');
const actionScene = document.querySelector('.action-scene');
const trackContainer = document.querySelector('.track-container');
const trackFill = document.querySelector('.track-fill');
const boxMover = document.querySelector('.box-mover');
const boxLid = document.querySelector('.box-lid');
const flyingPhoto = document.querySelector('.flying-photo');
const previewImg = document.querySelector('.preview-img');
const resultArea = document.querySelector('.result-area');
const finalImage = document.querySelector('.final-image');
const downloadBtn = document.querySelector('.download-btn');
const loadingLabel = document.querySelector('.loading-label');
const loadingPercent = document.querySelector('.loading-percent');
const loadingDots = document.querySelector('.loading-dots');

let isAnimating = false;
uploadIcon.addEventListener('click', (e) => {
  if (isAnimating || uploadContainer.classList.contains('expanded')) return;
  if (e.target.closest('.download-btn') || e.target.closest('.close-btn')) return;
  fileInput.click();
});

closeBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  resetToInitialState();
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

uploadContainer.addEventListener('dragover', (e) => {
  e.stopPropagation();
  if (!isAnimating) uploadContainer.classList.add('dragover');
});

uploadContainer.addEventListener('dragleave', () => uploadContainer.classList.remove('dragover'));

uploadContainer.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadContainer.classList.remove('dragover');
  if (e.dataTransfer.files.length > 0 && !isAnimating) handleFile(e.dataTransfer.files[0]);
});

window.addEventListener('paste', (e) => {
if (isAnimating) return;
const items = e.clipboardData.items;
for (let i = 0; i < items.length; i++) {
    if (items[i].type.indexOf('image') !== -1) {
        handleFile(items[i].getAsFile());
        break;
    }
  }
})


//哥几个我真累了，动画的实现就交给ai了（努力让这个动画成为唯一使用ai的地方）
// --- 核心动画逻辑 ---
// 进度条 + 加载文字由后端 SSE 的**真实进度**驱动；下面这些状态负责协调「箱子动画」。

let animRunning = false;    // 箱子是否已开始沿轨道奔跑
let animFinished = false;   // 后端是否已返回结果（或失败）
let animFinishing = false;  // 是否已进入收尾（开盖 → 弹出 → 展示结果）
let progressValue = 0;
let dotsTimer = null;

// 更新「加载过程」文字；文案变化时做一次轻微淡入，营造动态感
function setLoadingText(text) {
  if (!loadingLabel) return;
  if (loadingLabel.textContent !== text) {
    loadingLabel.textContent = text;
    loadingLabel.classList.remove('flash');
    void loadingLabel.offsetWidth;   // 强制重排，重启动画
    loadingLabel.classList.add('flash');
  }
}

// 轨道左右各留 50px（见 style.css 的 .track-container）。
// 箱子和进度条填充必须用同一像素坐标系，否则两者的百分比参考容器不同会错位：
//   · track-fill 是 .track-container 子元素，width 相对「轨道宽度」；
//   · box-mover 是 .progress-wrapper 子元素，left 相对「整个容器宽度」（且带 margin-left:-50px）。
// 统一：箱子中心 = 轨道起点(offsetLeft) + 轨道宽度 × 进度%。
const TRACK_PAD = 50;
function applyTrackProgress(pct) {
  const clamped = Math.max(0, Math.min(100, pct));
  const trackW = trackContainer ? trackContainer.clientWidth : 0;
  const pad = trackContainer ? trackContainer.offsetLeft : TRACK_PAD;
  const x = trackW * clamped / 100;               // 轨道内已走完的像素
  trackFill.style.width = x + 'px';               // 填充从轨道起点铺到 x
  boxMover.style.left = (pad + x) + 'px';         // 箱子中心压在填充末端
}

// 更新真实进度：百分比文字实时刷新；轨道/箱子在开跑后才跟着进度走
function setProgress(pct) {
  progressValue = Math.max(0, Math.min(100, pct));
  if (loadingPercent) loadingPercent.textContent = Math.round(progressValue) + '%';
  if (animRunning) {
    applyTrackProgress(progressValue);
  }
}

// 动态省略号：… 循环 0~3 个点，表现「正在加载」
function startDots() {
  stopDots();
  let n = 0;
  dotsTimer = setInterval(() => {
    n = (n + 1) % 4;
    if (loadingDots) loadingDots.textContent = '.'.repeat(n);
  }, 300);
}

function stopDots() {
  if (dotsTimer) { clearInterval(dotsTimer); dotsTimer = null; }
  if (loadingDots) loadingDots.textContent = '';
}

// 后端结束后，等箱子跑起来再收尾（开盖 → 弹出 → 展示结果）
function maybeFinish() {
  if (!animFinished || !animRunning || animFinishing) return;
  animFinishing = true;
  boxMover.classList.remove('running');
  applyTrackProgress(100);
  setLoadingText('降噪完成');
  stopDots();
  setTimeout(() => {
    boxLid.classList.add('open');
    setTimeout(() => {
      flyingPhoto.classList.remove('in-box');
      flyingPhoto.classList.add('pop-out');
      setTimeout(() => {
        actionScene.classList.remove('show');
        resultArea.classList.add('show');
        isAnimating = false;
      }, 1500);
    }, 1000);
  }, 500);
}

// RAW 扩展名 —— 与后端 infer.py 的 RAW_EXTS 保持一致。
// RAW 文件在浏览器里 file.type 常为空或 octet-stream，不能只看 image/*，
// 否则后端明明支持 RAW 双模式，前端却先把 RAW 拦掉了。
const RAW_EXTS = ['.arw', '.cr2', '.cr3', '.nef', '.nrw',
                  '.dng', '.raf', '.rw2', '.orf'];
function isAcceptedImage(file) {
  if (file.type && file.type.startsWith('image/')) return true;
  const name = (file.name || '').toLowerCase();
  return RAW_EXTS.some(ext => name.endsWith(ext));
}

function handleFile(file) {
    if (!isAcceptedImage(file) || isAnimating) return;
    isAnimating = true;
    animRunning = false;
    animFinished = false;
    animFinishing = false;
    progressValue = 0;

    const imageUrl = URL.createObjectURL(file);
    currentUploadName = file.name || '';
    finalImage.src = imageUrl;      // 先用原图占位，后端失败时兜底
    previewImg.src = imageUrl;
    downloadBtn.href = imageUrl;
    downloadBtn.setAttribute('download', file.name || 'downloaded_photo.jpg');

    // 重置进度条 / 加载文字 / 省略号动画
    applyTrackProgress(0);
    setProgress(0);
    setLoadingText('正在上传照片');
    startDots();

    // 1. 容器放大至接近全屏
    uploadContainer.classList.add('expanded');

    setTimeout(() => {
        // 2. 出现动作场景，箱子从高处掉落就位
        //    统一进度条填充与箱子的过渡时长，保证两者始终同步移动
        trackFill.style.transition = 'width 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
        boxMover.style.transition = 'left 0.5s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.4s ease';
        actionScene.classList.add('show');
        boxMover.classList.add('show');

        // 3. 照片飞入纸箱
        setTimeout(() => {
            flyingPhoto.classList.add('in-box');

            // 4. 合盖，箱子开始沿进度条奔跑 —— 此后进度由后端真实驱动
            setTimeout(() => {
                animRunning = true;
                boxMover.classList.add('running');
                setProgress(progressValue);   // 把已攒下的进度立即落位
                maybeFinish();                // 万一后端早已结束，立刻收尾
            }, 300);

        }, 300);

    }, 400);

    // ★ 触发真实降噪：SSE 流式接收进度，实时驱动进度条 + 加载文字
    denoiseImage(file, backend.selectedCamera || '', backend.defaultIso, (msg) => {
      if (msg.label) setLoadingText(msg.label);
      if (typeof msg.progress === 'number') setProgress(msg.progress);
    })
      .then((data) => {
        applyDenoiseResult(imageUrl, data);
        animFinished = true;
        setProgress(100);
        setLoadingText('降噪完成');
        maybeFinish();
      })
      .catch((err) => {
        console.warn('[backend] /api/denoise 失败，退回展示原图：', err);
        setStatus('降噪失败，已展示原图', false);
        animFinished = true;
        setLoadingText('降噪失败，展示原图');
        maybeFinish();
      });
}

function resetToInitialState() {
  uploadContainer.classList.remove('expanded');

  setTimeout(() => {
      isAnimating = false;
      animRunning = false;
      animFinished = false;
      animFinishing = false;
      stopDots();

      trackFill.style.transition = 'none';
      boxMover.style.transition = 'none';
      applyTrackProgress(0);
      boxMover.classList.remove('show', 'running', 'run');
      boxLid.classList.remove('open');
      flyingPhoto.classList.remove('in-box', 'pop-out');
      actionScene.classList.remove('show');
      resultArea.classList.remove('show');
      loadingPercent.textContent = '0%';
      loadingLabel.textContent = '正在上传照片';

      setTimeout(() => {
          trackFill.style.transition = 'width 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
          boxMover.style.transition = 'left 0.5s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.4s ease';

          finalImage.src = '';
          previewImg.src = '';
          downloadBtn.href = '#';
          downloadBtn.setAttribute('download', 'photo.jpg');
          // 恢复顶部 before/after 演示图
          leftImage.src = 'img/left.png';
          rightImage.src = 'img/right.png';

          fileInput.value = '';
      }, 100);
  }, 800);
}

const backgroundData = 
[
  "img/background1.svg",
  "img/background2.svg",
  "img/background3.svg",
]

const uploadbtnData = 
[
  "img/Canon.svg",
  "img/Sony.svg",
  "img/Nikon.svg",
]
const choicesCard1 = document.querySelector('.choices-card1');
const choicesCard2 = document.querySelector('.choices-card2');
const choicesCard3 = document.querySelector('.choices-card3');
const choicesCards = document.querySelectorAll('.choices-card1, .choices-card2, .choices-card3');
const backgroundImg = document.querySelector('.background .img');
const uploadBtn = document.querySelector('.upload-icon');

choicesCard1.addEventListener('click', () => {
  choicesCards.forEach(card => card.classList.remove('active'));
  choicesCard1.classList.add('active');
  backgroundImg.src = backgroundData[0];
  document.documentElement.style.setProperty('--color-orange', '#ea5019');
  document.documentElement.style.setProperty('--color-red', '#cf141b');
  document.documentElement.style.setProperty('--color-pinkorange', '#f5a88c');
  document.documentElement.style.setProperty('--color-pink', '#f7c0c3');
  uploadBtn.src = uploadbtnData[0];
})
choicesCard2.addEventListener('click', () => {
  choicesCards.forEach(card => card.classList.remove('active'));
  choicesCard2.classList.add('active');
  backgroundImg.src = backgroundData[1];
  document.documentElement.style.setProperty('--color-orange', '#5555ff');
  document.documentElement.style.setProperty('--color-red', '#103d7d');
  document.documentElement.style.setProperty('--color-pinkorange', '#e1e8ff');
  document.documentElement.style.setProperty('--color-pink', '#ebe1fd');
  uploadBtn.src = uploadbtnData[1];
})
choicesCard3.addEventListener('click', () => {
  choicesCards.forEach(card => card.classList.remove('active'));
  choicesCard3.classList.add('active');
  backgroundImg.src = backgroundData[2];
  document.documentElement.style.setProperty('--color-orange', '#62a400');
  document.documentElement.style.setProperty('--color-red', '#f4b516');
  document.documentElement.style.setProperty('--color-pinkorange', '#f1f6e6');
  document.documentElement.style.setProperty('--color-pink', '#f1f6e6');
  uploadBtn.src = uploadbtnData[2];
})

const fakeuploadbtn = document.querySelector('.uploadbtn');
fakeuploadbtn.addEventListener('click', () => {
  // 原先点击的是隐藏的 <a>，不会触发文件选择。改成直接触发真正的 file input。
  if (isAnimating) return;
  fileInput.click();
})

function setStatus(text, ok = true) {
  let el = document.getElementById('denoise-status');
  if (!el) return;   // 找不到状态条时静默跳过，避免阻断主流程
  el.textContent = '当前状态：' + text;
  el.style.background = ok ? 'rgba(17,24,39,.85)' : 'rgba(185,28,28,.9)';
}

const API_BASE = location.origin;

const backend = {
  ok: false,             
  cameras: [],           
  noiseTable: {},        
  selectedCamera: '',    
  defaultIso: 0,         
  device: '',
};

const CAMERA_BRANDS = [
  { key: 'canon', card: '.choices-card1' },
  { key: 'sony',  card: '.choices-card2' },
  { key: 'nikon', card: '.choices-card3' },
];

let currentUploadName = '';

async function initBackend() {
  try {
    const res = await fetch(`${API_BASE}/api/info`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const info = await res.json();
    backend.ok = true;
    backend.cameras = info.cameras || [];
    backend.noiseTable = info.noise_table || {};
    backend.device = info.device || '';


    const firstCam = backend.cameras.find(
      c => c && Array.isArray(c.isos) && c.isos.length);
    if (firstCam) {
      backend.defaultIso = firstCam.isos[Math.floor(firstCam.isos.length / 2)] * 10 || 32000;
    }

    CAMERA_BRANDS.forEach((b, i) => {
      const cam = (backend.cameras || []).find(
        c => c && c.name && c.name.toLowerCase().includes(b.key));
      const cardEl = document.querySelector(b.card);
      if (cam && cardEl) {
        cardEl.dataset.camera = cam.name;
        if (i === 0 && !backend.selectedCamera) backend.selectedCamera = cam.name;
      }
    });

    setStatus(`后端已连接 · 设备 ${backend.device.toUpperCase()}`);
  } catch (err) {
    setStatus('后端未连接 · 将直接预览原图（不做降噪）', false);
  }
}

async function denoiseImage(file, camera, iso, onEvent) {
  const fd = new FormData();
  fd.append('file', file, file.name);  
  fd.append('camera', camera || '');
  fd.append('iso', String(iso || 0));
  fd.append('tile', '512');
  fd.append('overlap', '32');

  const res = await fetch(`${API_BASE}/api/denoise`, { method: 'POST', body: fd });
  if (!res.ok) {
    let detail = '';
    try { detail = (await res.json()).detail || ''; } catch (_) {}
    throw new Error(detail || `HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep;
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const rawEvent = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      for (const line of rawEvent.split('\n')) {
        if (line.indexOf('data:') !== 0) continue;
        const payload = line.slice(5).trim();
        if (!payload) continue;
        let msg;
        try { msg = JSON.parse(payload); } catch (_) { continue; }
        if (msg.type === 'done') {
          result = msg;
        } else if (msg.type === 'error') {
          throw new Error(msg.detail || '后端处理失败');
        } else if (typeof onEvent === 'function') {
          onEvent(msg);
        }
      }
    }
  }

  if (!result) throw new Error('连接中断，未收到降噪结果');
  return result;
}

function applyDenoiseResult(originalUrl, data) {
  const outUrl = data.image;
  finalImage.src = outUrl;
  downloadBtn.href = outUrl;

  const base = (currentUploadName || 'photo').replace(/\.[^.]+$/, '');
  downloadBtn.setAttribute('download', `denoised_${base}.png`);

  const cam = data.used_camera || data.camera || '';
  setStatus(`降噪完成 · ${cam} ISO${data.iso / 10} · ${data.elapsed_ms}ms`);
  console.log('[backend] /api/denoise 完成：', data);
}

initBackend();