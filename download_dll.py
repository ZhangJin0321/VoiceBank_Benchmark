import os
import urllib.request

# 64位 Windows 预编译 librnnoise.dll 国内加速镜像节点列表
urls = [
    "https://ghp.ci/https://github.com/j-fuerst/rnnoise-windows/releases/download/v1.0/librnnoise.dll",
    "https://ghproxy.net/https://github.com/j-fuerst/rnnoise-windows/releases/download/v1.0/librnnoise.dll",
    "https://mirror.ghproxy.com/https://github.com/j-fuerst/rnnoise-windows/releases/download/v1.0/librnnoise.dll"
]

target_file = "librnnoise.dll"

print("开始获取 64位 librnnoise.dll 动态库...")

success = False
headers = {'User-Agent': 'Mozilla/5.0'}

for idx, url in enumerate(urls, 1):
    try:
        print(f"尝试通道 [{idx}/{len(urls)}]...")
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response, open(target_file, 'wb') as f:
            f.write(response.read())

        # 验证下载的文件大小（正常 DLL 约 150KB ~ 300KB）
        if os.path.exists(target_file) and os.path.getsize(target_file) > 50000:
            print(f"\n✅ 成功！`librnnoise.dll` 已保存至项目根目录 ({os.path.getsize(target_file) / 1024:.1f} KB)")
            success = True
            break
        else:
            if os.path.exists(target_file):
                os.remove(target_file)
    except Exception as e:
        print(f"通道 {idx} 连接超时，自动切至下一个节点...")

if not success:
    print("\n❌ 自动下载受阻。请手动在浏览器访问以下任意链接下载，并改名为 `librnnoise.dll` 放入项目根目录：")
    for u in urls:
        print(f" - {u}")