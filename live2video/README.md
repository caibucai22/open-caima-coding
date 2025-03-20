# live2video

本项目是一个基于 `Python` 的 Android「动态照片」提取工具，可以将动态照片中的视频提取出来。

在线使用 [地址](tab.caima.tech)。

## 介绍
在 Android 系统中，动态照片是一种特殊的图片格式，它实际上是一个包含了视频的 JPG 图片。

JPG文件由多个段 `(Segments)` 构成，每个段以标记 `marker` 开头，标记由 `0xFF` 后跟一个标记类型字节组成（如 `0xD8`、 `0xDB` 等）。所有数据段按顺序排列，共同定义图像参数和压缩数据。

Android 系统中保存的动态照片的JPG文件结构如下图所示：
<img src="image/display.png" alt="JPG文件结构" style="zoom:5%;" />

## 提取方法

本方案的提取方法是通过搜索 JPG 文件中的 MP4 文件头来定位视频数据，然后将视频数据提取出来。
关键函数有两个：
`search_pattern` 函数使用 Boyer-Moore 算法搜索 MP4 文件头位置并返回其索引

`live2video` 函数则是将 JPG 文件中的 MP4 数据提取出来并保存到新的 MP4 文件中。



### `search_pattern(pattern, data)`
```python
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
```

###  `live2video(jpg_path)` 
```python
def live2video(jpg_path):
    # 定义MP4文件头模式
    mp4_patterns = [
        b'\x00\x00\x00\x18\x66\x74\x79\x70\x6D\x70\x34\x32',
        b'\x00\x00\x00\x1C\x66\x74\x79\x70\x69\x73\x6F\x6D',
        b'\x00\x00\x00\x1C\x66\x74\x79\x70',
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
```

### 使用方法

```python
jpg_path = 'path/to/your/live_photo.jpg'
live2video(jpg_path)
```

## 提取效果
提取效果如下图所示：
| 原始动态照片 | 提取效果 |
| :----------: | :------: |
| ![原始动态照片](image/test.jpg) | ![提取效果](image/test.mp4) |
| ![原始动态照片](image/test2.jpg) | ![提取效果](image/test2.mp4) |

## 致谢
提取mp4视频的核心代码来自于 [GoMoPho](https://github.com/cliveontoast/GoMoPho) 项目。