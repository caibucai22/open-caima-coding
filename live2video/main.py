import os

def search_pattern(pattern, data):
    """使用Boyer-Moore算法搜索字节模式"""
    m = len(pattern)
    n = len(data)
    if m == 0 or n == 0:
        return -1

    # 预处理跳转表
    jump_table = [m] * 256
    for i in range(m - 1):
        jump_table[pattern[i]] = m - 1 - i

    # 搜索模式
    i = m - 1
    while i < n:
        j = m - 1
        while j >= 0 and data[i] == pattern[j]:
            i -= 1
            j -= 1
        if j == -1:
            return i + 1
        i += max(jump_table[data[i]], 1)

    return -1

def live2video(jpg_path):
    # 定义MP4文件头模式
    mp4_patterns = [
        
        b'\x00\x00\x00\x18\x66\x74\x79\x70\x6D\x70\x34\x32',
        b'\x00\x00\x00\x1C\x66\x74\x79\x70\x69\x73\x6F\x6D',
        b'\x00\x00\x00\x1C\x66\x74\x79\x70',
        # b'\x00\x00\x00\x20\x66\x74\x79\x70\x69\x73\x6F\x36\x00\x00\x00\x01\x6D\x70\x34\x32\x69\x73\x6F\x36\x61\x76\x63\x31\x69\x73\x6F\x6D',
    ]

    with open(jpg_path, 'rb') as f:
        file_bytes = f.read()

    index_of_mp4 = -1
    for pattern in mp4_patterns:
        index_of_mp4 = search_pattern(pattern, file_bytes)
        print(index_of_mp4)
        if index_of_mp4 >= 0:
            break
    
    if index_of_mp4 >= 0:
        mp4_path = os.path.splitext(jpg_path)[0] + '.mp4'
        with open(mp4_path, 'wb') as mp4_file:
            mp4_file.write(file_bytes[index_of_mp4:])
        print(f"MP4 file extracted and saved to: {mp4_path}")
    else:
        print("No MP4 file found in the given JPG file")

# 示例用法
live2video(r'C:\Users\Malong\Desktop\open-caima-coding\live2video\image\123.jpg')