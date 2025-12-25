import re
import os
import math
import tkinter as tk
from tkinter import filedialog, messagebox


import h5py
import matplotlib.pyplot as plt


def choose_h5_file() -> str:
    root = tk.Tk()
    root.withdraw()
    root.update()
    path = filedialog.askopenfilename(
        title="Select a .h5 file",
        filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")]
    )
    root.destroy()
    return path


def find_dataset_by_name(h5obj, target_name: str):
    """
    Recursively search an HDF5 file/group for a dataset with a given name.
    Returns the dataset object or None.
    """
    if isinstance(h5obj, h5py.Dataset) and h5obj.name.split("/")[-1] == target_name:
        return h5obj
    if isinstance(h5obj, (h5py.File, h5py.Group)):
        for key in h5obj.keys():
            item = h5obj[key]
            if isinstance(item, h5py.Dataset) and key == target_name:
                return item
            if isinstance(item, h5py.Group):
                found = find_dataset_by_name(item, target_name)
                if found is not None:
                    return found
    return None


def load_circuit_text(h5_path: str) -> str:
    with h5py.File(h5_path, "r") as f:
        ds = None
        # common cases
        if "circuit" in f:
            ds = f["circuit"]
        else:
            ds = find_dataset_by_name(f, "circuit")

        if ds is None:
            raise KeyError("Cannot find dataset named 'circuit' in the selected HDF5 file.")

        data = ds[()]
        # handle bytes / numpy scalar
        if isinstance(data, (bytes, bytearray)):
            return data.decode("utf-8", errors="ignore")
        if hasattr(data, "dtype") and getattr(data.dtype, "kind", "") in ("S", "O"):
            # could be numpy bytes scalar
            try:
                return data.astype(str)
            except Exception:
                pass
        return str(data)


def is_int_close(x: float, tol: float = 1e-9) -> bool:
    return abs(x - round(x)) < tol


def is_odd_int(x: float) -> bool:
    if not is_int_close(x):
        return False
    xi = int(round(x))
    return (xi % 2) == 1


def parse_qubit_coords(circuit_text: str):
    """
    Parse lines like: QUBIT_COORDS(1, 1) 1
    Return: dict {qubit_id(int): (x(float), y(float))}
    """
    pattern = re.compile(r"QUBIT_COORDS\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)\s*(\d+)")
    coords = {}
    for m in pattern.finditer(circuit_text):
        x, y, q = float(m.group(1)), float(m.group(2)), int(m.group(3))
        coords[q] = (x, y)
    if not coords:
        raise ValueError("No QUBIT_COORDS found in circuit text.")
    return coords


def parse_detector_xy(circuit_text: str):
    """
    Parse DETECTOR(x,y,t) ... ; return set of (x,y) float pairs.
    """
    # Stim allows DETECTOR with 2 or 3 coords; we handle both.
    pattern3 = re.compile(r"DETECTOR\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)")
    pattern2 = re.compile(r"DETECTOR\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)")

    xy = set()
    for m in pattern3.finditer(circuit_text):
        x, y = float(m.group(1)), float(m.group(2))
        xy.add((x, y))
    # in case some are 2-arg form
    for m in pattern2.finditer(circuit_text):
        x, y = float(m.group(1)), float(m.group(2))
        xy.add((x, y))

    if not xy:
        raise ValueError("No DETECTOR(...) found in circuit text.")
    return xy


def detect_noise_type(circuit_text: str) -> str:
    """
    Decide whether the circuit uses X_ERROR or Z_ERROR.
    """
    has_x = "X_ERROR" in circuit_text
    has_z = "Z_ERROR" in circuit_text
    if has_x and not has_z:
        return "X_ERROR"
    if has_z and not has_x:
        return "Z_ERROR"
    if has_x and has_z:
        # if both exist, choose the dominant (first occurrence) to follow user's rule
        if circuit_text.find("X_ERROR") < circuit_text.find("Z_ERROR"):
            return "X_ERROR"
        return "Z_ERROR"
    return "UNKNOWN"


def build_coord_to_qubit_map(qubit_coords: dict):
    """
    Return: dict {(x,y): qubit_id}
    If duplicates exist (unlikely), keep the first.
    """
    mp = {}
    for q, (x, y) in qubit_coords.items():
        key = (x, y)
        if key not in mp:
            mp[key] = q
    return mp


def plot_layout(qubit_coords: dict, detector_xy: set, noise_type: str, title: str = None):
    coord_to_q = build_coord_to_qubit_map(qubit_coords)

    # data qubits: x,y odd integers
    data_qubits = []
    anc_qubits = []
    for q, (x, y) in qubit_coords.items():
        if is_odd_int(x) and is_odd_int(y):
            data_qubits.append(q)
        else:
            anc_qubits.append(q)

    # ancillas referenced by DETECTOR coords
    detector_ancillas = []
    for (dx, dy) in detector_xy:
        if (dx, dy) in coord_to_q:
            q = coord_to_q[(dx, dy)]
            # only treat as ancilla if it's not a data qubit
            if q in anc_qubits:
                detector_ancillas.append(q)

    detector_ancillas = sorted(set(detector_ancillas))
    remaining_ancillas = sorted(set(anc_qubits) - set(detector_ancillas))

    # Apply your coloring rule
    # If X_ERROR -> DETECTOR ancillas are Z stabilizers (red), remaining are X stabilizers (blue)
    # If Z_ERROR -> DETECTOR ancillas are X stabilizers (blue), remaining are Z stabilizers (red)
    if noise_type == "X_ERROR":
        z_anc = detector_ancillas
        x_anc = remaining_ancillas
    elif noise_type == "Z_ERROR":
        x_anc = detector_ancillas
        z_anc = remaining_ancillas
    else:
        # fallback: still show DETECTOR ancillas distinct
        z_anc = detector_ancillas
        x_anc = remaining_ancillas

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    # plot data qubits (black squares)
    xd = [qubit_coords[q][0] for q in data_qubits]
    yd = [qubit_coords[q][1] for q in data_qubits]
    ax.scatter(xd, yd, marker="s", s=140, edgecolors="black", facecolors="none", label="Data qubits")

    # plot Z-stabilizer ancillas (red circles)
    xz = [qubit_coords[q][0] for q in z_anc]
    yz = [qubit_coords[q][1] for q in z_anc]
    ax.scatter(xz, yz, marker="o", s=130, edgecolors="red", facecolors="none", label="Z-stabilizer ancillas")

    # plot X-stabilizer ancillas (blue circles)
    xx = [qubit_coords[q][0] for q in x_anc]
    yx = [qubit_coords[q][1] for q in x_anc]
    ax.scatter(xx, yx, marker="o", s=130, edgecolors="blue", facecolors="none", label="X-stabilizer ancillas")

    # annotate all qubits with q-id
    def annotate(q_list, dy=0.18):
        for q in q_list:
            x, y = qubit_coords[q]
            ax.text(x, y + dy, str(q), ha="center", va="bottom", fontsize=9)

    annotate(data_qubits, dy=0.18)
    annotate(z_anc, dy=0.18)
    annotate(x_anc, dy=0.18)

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linewidth=0.5)
    ax.set_xlabel("x (QUBIT_COORDS)")
    ax.set_ylabel("y (QUBIT_COORDS)")

    if title is None:
        title = f"Qubit layout from QUBIT_COORDS (noise={noise_type})"
    ax.set_title(title)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),  # (x,y)，x>1 表示在坐标轴右侧外
        borderaxespad=0.0,
        frameon=True
    )
    # 给右侧 legend 留出空间（避免被裁掉）
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()


def main():
    h5_path = choose_h5_file()
    if not h5_path:
        return

    try:
        circuit_text = load_circuit_text(h5_path)
        qubit_coords = parse_qubit_coords(circuit_text)
        detector_xy = parse_detector_xy(circuit_text)
        noise_type = detect_noise_type(circuit_text)

        plot_layout(
            qubit_coords=qubit_coords,
            detector_xy=detector_xy,
            noise_type=noise_type,
            title=os.path.basename(h5_path) + f" (detector-based stabilizer coloring, {noise_type})"
        )
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", str(e))
        root.destroy()


if __name__ == "__main__":
    main()
