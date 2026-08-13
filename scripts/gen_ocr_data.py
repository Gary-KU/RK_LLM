"""Generate 100 synthetic OCR training images with Chinese text + labels.jsonl"""
import json, os, random, math
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

OUT_DIR  = "data/ocr"
IMG_DIR  = os.path.join(OUT_DIR, "images")
JSONL    = os.path.join(OUT_DIR, "labels.jsonl")
NUM_IMGS = 100
SEED     = 42
random.seed(SEED)

os.makedirs(IMG_DIR, exist_ok=True)

# --------------- diverse Chinese text content ---------------
TEXTS = [
    # 路牌 / 指示牌
    "中山路 50m", "解放大道 200m", "人民广场 前方 500m",
    "停车场 P", "出口 EXIT", "入口 ENTRANCE", "卫生间 →",
    "禁止停车", "限速 60", "前方施工 请绕行",
    # 招牌 / 门头
    "老张牛肉面", "阿强烧烤", "天天鲜果", "便民超市",
    "幸福药店", "小李理发", "阳光花店", "好再来饭店",
    # 菜单 / 价目表
    "红烧牛肉面 18元", "酸辣粉 10元", "珍珠奶茶 12元",
    "炒饭 15元", "煲仔饭 22元", "叉烧包 8元",
    # 票据 / 标签
    "NO.20260728001", "单价: 25.00 数量: 3 合计: 75.00",
    "生产日期: 2026-07-28", "保质期: 12个月", "净含量: 500ml",
    # 告示 / 通知
    "今日休息 敬请谅解",
    "请随手关门", "节约用水", "已消毒",
    "消防通道 禁止占用", "进入车间 请戴安全帽",
    # 产品信息
    "RK3588 开发板", "USB 3.0 接口", "电源适配器 12V 3A",
    "型号: X152T", "固件版本: v2.1.0",
    # 文件 / 表格
    "检验报告", "合格证", "出厂编号: 2026A00158",
    "品名: 电子元器件", "规格: 0805",
    # 混合中英文 / 数字
    "WiFi 密码: abc12345", "IP: 192.168.1.112", "SSID: Rockchip_5G",
    "CPU 温度: 45°C", "内存使用率: 72%",
    "Model: RK3588", "SN: RK20260728001",
    # 多行文本
    "收件人: 张三\n电话: 13800138000\n地址: 广东省深圳市南山区科技园",
    "注意事项:\n1. 请勿拆卸设备\n2. 工作温度 0-40°C\n3. 额定电压 12V",
    # 常见标识
    "推 PUSH", "拉 PULL", "开 ON", "关 OFF",
    "小心台阶", "当心触电", "非工作人员 禁止入内",
    # 长句子
    "欢迎光临本店 全场八折优惠活动进行中",
    "本设备已通过国家强制性产品认证",
    "如需帮助请拨打客服热线 400-888-0000",
]

# --------------- font fallback chain ---------------
FONT_CANDIDATES = [
    "C:/Windows/Fonts/simsun.ttc",     # 宋体
    "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",     # 黑体
    "C:/Windows/Fonts/STKAITI.TTF",   # 楷体
    "C:/Windows/Fonts/Deng.ttf",       # 等线
]

FONTS = {}
for size in [20, 24, 28, 32, 36, 40, 44, 48, 56, 64]:
    for path in FONT_CANDIDATES:
        try:
            FONTS.setdefault(size, []).append(ImageFont.truetype(path, size))
        except Exception:
            continue

# fallback: default font (won't render Chinese well but won't crash)
for size in [20, 24, 28, 32, 36, 40, 44, 48, 56, 64]:
    if size not in FONTS:
        FONTS[size] = [ImageFont.load_default()]

def random_font(size: int) -> ImageFont.FreeTypeFont:
    size = min(FONTS.keys(), key=lambda s: abs(s - size))
    return random.choice(FONTS[size])

# --------------- image generation ---------------
BG_COLORS = [
    (255, 255, 255), (248, 248, 240), (255, 250, 240),
    (245, 245, 245), (240, 248, 255), (255, 245, 238),
    (250, 235, 215), (245, 255, 250), (253, 245, 230),
]
TEXT_COLORS = [
    (0, 0, 0), (30, 30, 30), (50, 50, 50),
    (10, 10, 40), (40, 10, 10), (0, 40, 0),
    (80, 20, 0), (0, 0, 80), (60, 0, 60),
]

def add_noise(img: Image.Image) -> Image.Image:
    """Add subtle grain to simulate real camera noise."""
    arr = img.copy()
    w, h = arr.size
    pixels = arr.load()
    for _ in range(w * h // 20):
        x, y = random.randint(0, w-1), random.randint(0, h-1)
        r, g, b = pixels[x, y]
        v = random.randint(-15, 15)
        pixels[x, y] = (max(0, min(255, r+v)),
                         max(0, min(255, g+v)),
                         max(0, min(255, b+v)))
    return arr

def rotate_text(draw: ImageDraw.ImageDraw, text: str, font,
                color, cx: int, cy: int, angle: float):
    """Draw text rotated around (cx, cy)."""
    txt_img = Image.new("RGBA", (1200, 200), (0,0,0,0))
    txt_draw = ImageDraw.Draw(txt_img)
    txt_draw.text((0, 0), text, font=font, fill=color)
    rotated = txt_img.rotate(angle, expand=True, resample=Image.BICUBIC,
                             center=(txt_img.width//2, txt_img.height//2))
    return rotated

print(f"Generating {NUM_IMGS} images...")
records = []

for i in range(NUM_IMGS):
    text = random.choice(TEXTS)
    # vary image dimensions
    w = random.randint(500, 1000)
    h = random.randint(80, 350)
    bg = random.choice(BG_COLORS)
    fg = random.choice(TEXT_COLORS)
    font_size = random.randint(22, 52)
    font = random_font(font_size)

    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)

    # Try rotated approach for variety (30% chance)
    if random.random() < 0.3 and "\n" not in text:
        angle = random.uniform(-3, 3)
        cx, cy = w // 2, h // 2
        rotated = rotate_text(draw, text, font, fg, cx, cy, angle)
        rx, ry = cx - rotated.width // 2, cy - rotated.height // 2
        img.paste(rotated, (rx, ry), rotated)
    else:
        # Multi-line or single line
        lines = text.split("\n")
        # Measure total height
        line_heights = [draw.textbbox((0,0), line, font=font)[3] for line in lines]
        total_h = sum(line_heights) + (len(lines)-1) * 6
        start_y = max(10, (h - total_h) // 2)

        for j, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            tw = bbox[2] - bbox[0]
            x = max(5, (w - tw) // 2)
            draw.text((x, start_y), line, font=font, fill=fg)
            start_y += line_heights[j] + 6

    # Post-processing for realism
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.6)))
    if random.random() < 0.4:
        img = add_noise(img)
    if random.random() < 0.3:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7, 1.2))
    if random.random() < 0.3:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.15))

    fname = f"{i:03d}.jpg"
    path = os.path.join(IMG_DIR, fname)
    img.save(path, quality=92)
    records.append({"image": f"images/{fname}", "text": text})

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{NUM_IMGS} ...")

# Write labels.jsonl
with open(JSONL, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nDone! {NUM_IMGS} images → {IMG_DIR}/")
print(f"Labels → {JSONL}")
print(f"Example record: {json.dumps(records[0], ensure_ascii=False)}")
