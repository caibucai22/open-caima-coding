

def read_jpeg_segments(data):
    # JPEG标记定义
    jpeg_markers = {
        0xD8: "SOI (Start of Image)",
        0xD9: "EOI (End of Image)",
        0xE0: "APP0 (JFIF)",
        0xE1: "APP1 (Exif)",
        0xE2: "APP2 (ICC Profile)",
        0xDB: "DQT (Define Quantization Table)",
        0xC0: "SOF0 (Start of Frame, Baseline DCT)",
        0xC1: "SOF1 (Start of Frame, Extended Sequential DCT)",
        0xC2: "SOF2 (Start of Frame, Progressive DCT)",
        0xC4: "DHT (Define Huffman Table)",
        0xDA: "SOS (Start of Scan)",
        0xDD: "DRI (Define Restart Interval)",
        0xFE: "COM (Comment)",
        # 其他APP段
        0xE3: "APP3",
        0xE4: "APP4",
        0xE5: "APP5",
        0xE6: "APP6",
        0xE7: "APP7",
        0xE8: "APP8",
        0xE9: "APP9",
        0xEA: "APP10",
        0xEB: "APP11",
        0xEC: "APP12",
        0xED: "APP13 (Photoshop)",
        0xEE: "APP14 (Adobe)",
        0xEF: "APP15",
        # 其他SOF段
        0xC3: "SOF3 (Lossless)",
        0xC5: "SOF5 (Differential Sequential DCT)",
        0xC6: "SOF6 (Differential Progressive DCT)",
        0xC7: "SOF7 (Differential Lossless)",
        0xC8: "JPG (Reserved)",
        0xC9: "SOF9 (Extended Sequential DCT, Arithmetic coding)",
        0xCA: "SOF10 (Progressive DCT, Arithmetic coding)",
        0xCB: "SOF11 (Lossless, Arithmetic coding)",
        0xCD: "SOF13 (Differential Sequential DCT, Arithmetic coding)",
        0xCE: "SOF14 (Differential Progressive DCT, Arithmetic coding)",
        0xCF: "SOF15 (Differential Lossless, Arithmetic coding)",
    }

    idx = 0
    segments = []
    while idx < len(data):
        if data[idx] == 0xFF:
            marker = data[idx + 1]
            # 跳过连续的 0xFF
            while idx + 1 < len(data) and data[idx + 1] == 0xFF:
                idx += 1
            # 特殊标记处理（无需长度的标记）
            if marker == 0xD8:  # SOI
                segments.append({
                    "marker": "0xFFD8", 
                    "marker_name": jpeg_markers.get(marker, f"Unknown (0x{marker:02X})"),
                    "pos": idx, 
                    "length": 2
                })
                idx += 2
                continue
            if marker == 0xD9:  # EOI
                segments.append({
                    "marker": "0xFFD9", 
                    "marker_name": jpeg_markers.get(marker, f"Unknown (0x{marker:02X})"),
                    "pos": idx, 
                    "length": 2
                })
                idx += 2
                break
            # 其他标记需读取长度
            if idx + 3 < len(data):
                length = (data[idx + 2] << 8) + data[idx + 3]
                marker_code = f"0xFF{marker:02X}"
                segments.append({
                    "marker": marker_code,
                    "marker_name": jpeg_markers.get(marker, f"Unknown (0x{marker:02X})"),
                    "pos": idx,
                    "length": length,
                    "data": data[idx + 4 : idx + 2 + length]
                })
                idx += 2 + length
            else:
                break
        else:
            idx += 1
    return segments


with open(r"C:\Users\Malong\Desktop\open-caima-coding\live2video\image\test.jpg", "rb") as f:
    data = f.read()
segments = read_jpeg_segments(data)
for seg in segments:
    print(f"Marker: {seg['marker']} - {seg['marker_name']}, Length: {seg['length']}")