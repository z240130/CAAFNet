import requests
from bs4 import BeautifulSoup

# 目标 URL
url = 'https://huggingface.co/timm/pvt_v2_b2_li.in1k/blob/main/pytorch_model.bin'

# 设置 SOCKS5 代理
proxies = {
    'http': 'socks5://127.0.0.1:1079',
    'https': 'socks5://127.0.0.1:1079',
}

try:
    # 发送请求，使用代理
    response = requests.get(url, timeout=10, verify=False, proxies=proxies)
    response.raise_for_status()  # 检查请求是否成功

    # 解析网页内容
    soup = BeautifulSoup(response.text, 'html.parser')

    # 示例：查找特定的内容
    print(soup.prettify())  # 打印整个网页内容（示例）

except requests.exceptions.RequestException as e:
    print(f"请求出错: {e}")
