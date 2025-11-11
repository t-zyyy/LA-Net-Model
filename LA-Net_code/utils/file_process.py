import chardet

def load_lines(file_path):
    """Read file into a list of lines with detected encoding."""
    # 自动检测编码
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']  # 获取检测到的编码

    print(f"Detected encoding: {encoding}")  # 输出检测到的编码

    with open(file_path, 'r', encoding=encoding) as fio:
        lines = fio.read().splitlines()
    return lines
