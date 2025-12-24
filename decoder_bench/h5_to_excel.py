import os
import re
import math
import h5py
import numpy as np
import pandas as pd
from tkinter import Tk
from tkinter import filedialog

# Excel 的硬限制
EXCEL_MAX_ROWS = 1_048_576
EXCEL_MAX_COLS = 16_384

def _safe_sheet_name(name: str) -> str:
    """
    Excel sheet 名限制：
      - 不能含: : \ / ? * [ ]
      - 最长 31 字符
    """
    name = re.sub(r"[:\\/?*\[\]]", "_", name)
    name = name.strip()
    if not name:
        name = "sheet"
    return name[:31]

def _decode_if_bytes(x):
    if isinstance(x, (bytes, np.bytes_)):
        try:
            return x.decode("utf-8", errors="replace")
        except Exception:
            return repr(x)
    return x

def _dataset_to_2d_array(ds: h5py.Dataset):
    """
    尽量把 dataset 转成可写入 Excel 的二维结构：
      - 标量 -> (1,1)
      - 1D -> (n,1)
      - 2D -> 原样
      - >2D -> (dim0, prod(other_dims)) 展平
    同时对 bytes/string 做解码。
    """
    data = ds[()]  # 读取到内存；如果你的文件极大，这一步会比较吃内存

    # 标量
    if np.isscalar(data) or (isinstance(data, np.ndarray) and data.shape == ()):
        v = data.item() if isinstance(data, np.ndarray) else data
        # bool 标量转 0/1
        if isinstance(v, (bool, np.bool_)):
            v = int(v)
        else:
            v = _decode_if_bytes(v)
        return np.array([[v]], dtype=object)

    arr = np.array(data)

    # ===== 关键改动：bool -> 0/1 =====
    if arr.dtype == np.bool_ or arr.dtype.kind == "b":
        arr = arr.astype(np.uint8)
        # 后续无需再做 bytes/string 处理
    else:
        # 处理字符串/bytes
        if arr.dtype.kind in ("S", "O", "U"):
            arr = np.vectorize(_decode_if_bytes, otypes=[object])(arr)

    # 维度整理
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    first = arr.shape[0]
    rest = int(np.prod(arr.shape[1:]))
    return arr.reshape(first, rest)

def _write_2d_array_split(writer: pd.ExcelWriter, sheet_base: str, arr2d: np.ndarray):
    """
    将二维数组写入 Excel，必要时按行/列拆分成多个 sheet。
    """
    rows, cols = arr2d.shape
    # 至少保留一行表头空间（pandas写入会包含header/index，这里我们关闭它们）
    max_rows = EXCEL_MAX_ROWS
    max_cols = EXCEL_MAX_COLS

    row_parts = math.ceil(rows / max_rows) if rows > max_rows else 1
    col_parts = math.ceil(cols / max_cols) if cols > max_cols else 1

    for rp in range(row_parts):
        r0 = rp * max_rows
        r1 = min((rp + 1) * max_rows, rows)
        for cp in range(col_parts):
            c0 = cp * max_cols
            c1 = min((cp + 1) * max_cols, cols)

            part = arr2d[r0:r1, c0:c1]

            # 生成 sheet 名（避免超长）
            suffix = ""
            if row_parts > 1:
                suffix += f"_r{rp+1}"
            if col_parts > 1:
                suffix += f"_c{cp+1}"
            sheet_name = _safe_sheet_name(sheet_base + suffix)

            df = pd.DataFrame(part)
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

def _write_text_block(writer: pd.ExcelWriter, sheet_name: str, text: str):
    lines = text.splitlines() if text is not None else []
    df = pd.DataFrame({"text": lines})
    df.to_excel(writer, sheet_name=_safe_sheet_name(sheet_name), index=False)

def _walk_h5(obj, path=""):
    """
    递归遍历 HDF5，返回 (full_path, h5_object) 列表
    """
    items = []
    if isinstance(obj, h5py.Dataset):
        items.append((path, obj))
    elif isinstance(obj, h5py.Group):
        for k, v in obj.items():
            new_path = f"{path}/{k}" if path else k
            items.extend(_walk_h5(v, new_path))
    return items

def h5_to_excel(h5_path: str):
    out_xlsx = os.path.splitext(h5_path)[0] + ".xlsx"

    with h5py.File(h5_path, "r") as f, pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        # 元信息 sheet
        meta_rows = []
        for full_path, obj in _walk_h5(f):
            if isinstance(obj, h5py.Dataset):
                meta_rows.append({
                    "path": full_path,
                    "shape": str(obj.shape),
                    "dtype": str(obj.dtype),
                })
        pd.DataFrame(meta_rows).to_excel(writer, sheet_name=_safe_sheet_name("_meta"), index=False)

        # 写每个 dataset
        for full_path, ds in _walk_h5(f):
            # sheet base 名用路径替换斜杠
            sheet_base = full_path.replace("/", "__")

            try:
                # 特殊处理：如果看起来是“电路文本”，尽量按行写
                # （常见情况：dataset 是字符串或 bytes）
                if ds.dtype.kind in ("S", "O", "U") and (ds.shape == () or (len(ds.shape) == 1 and ds.shape[0] < 2_000_000)):
                    raw = ds[()]
                    if isinstance(raw, (bytes, np.bytes_)):
                        raw = raw.decode("utf-8", errors="replace")
                    elif isinstance(raw, np.ndarray) and raw.dtype.kind in ("S", "O", "U"):
                        # 1D字符串数组：逐元素写
                        arr = np.array(raw)
                        arr = np.vectorize(_decode_if_bytes, otypes=[object])(arr)
                        df = pd.DataFrame({"text": arr.reshape(-1)})
                        df.to_excel(writer, sheet_name=_safe_sheet_name(sheet_base), index=False)
                        continue
                    if isinstance(raw, str):
                        _write_text_block(writer, sheet_base, raw)
                        continue

                arr2d = _dataset_to_2d_array(ds)
                _write_2d_array_split(writer, sheet_base, arr2d)

            except Exception as e:
                # 如果某个dataset写失败，把错误写进一个sheet，方便定位
                err_sheet = _safe_sheet_name(f"ERR_{sheet_base}")
                pd.DataFrame([{
                    "dataset": full_path,
                    "error": repr(e)
                }]).to_excel(writer, sheet_name=err_sheet, index=False)

    return out_xlsx

def pick_file_and_convert():
    root = Tk()
    root.withdraw()  # 不显示主窗口
    root.attributes("-topmost", True)

    h5_path = filedialog.askopenfilename(
        title="选择要转换的 HDF5(.h5) 文件",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    )
    root.destroy()

    if not h5_path:
        print("未选择文件，已退出。")
        return

    out_xlsx = h5_to_excel(h5_path)
    print(f"转换完成：{out_xlsx}")

if __name__ == "__main__":
    # 依赖：pip install h5py pandas openpyxl
    pick_file_and_convert()
