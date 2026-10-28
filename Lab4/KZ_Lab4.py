import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1.Загрузка изображения и преобразование цветного к полутоновому
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2

    return cv2, mo, np, plt


@app.cell
def _(plt):
    def show(img, title=""):
        fig, ax = plt.subplots(figsize=(5, 7))
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        plt.show()

    return (show,)


@app.cell
def _(cv2, show):
    image = cv2.imread("image.jpg", cv2.IMREAD_COLOR_RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    show(gray)
    return (gray,)


@app.cell
def _(gray, np, plt):
    def histogram(img):
        return np.bincount(img.ravel(), minlength=256)


    hist = histogram(gray)

    _fig, _ax = plt.subplots(figsize=(12, 3))
    _ax.bar(np.arange(256), hist, width=1)
    plt.show()
    return hist, histogram


@app.cell
def _(cv2, np):
    def quantize(img, levels):
        lut = (np.arange(256) * levels // 256) * 255 // (levels - 1)
        return cv2.LUT(img, lut.astype(np.uint8))

    return (quantize,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2.Квантование по яркости (с ручным заданием количества уровней)
    """)
    return


@app.cell
def _(gray, quantize, show):
    for levels in [2, 4, 8, 16]:
        Q = quantize(gray, levels)
        show(Q, f"levels = {levels}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3. Бинаризация
        1. Бинаризация методом Оцу глобальная
    """)
    return


@app.cell
def _(np):
    def otsu(hist, lo=0, hi=256):
        h = hist[lo:hi].astype(np.float64)
        p = h / h.sum()
        levels = np.arange(lo, hi)

        w = np.cumsum(p)
        mu = np.cumsum(p * levels)
        mu_t = mu[-1]

        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_b = (mu_t * w - mu) ** 2 / (w * (1 - w))
        sigma_b = np.nan_to_num(sigma_b[:-1])

        k = int(np.argmax(sigma_b))
        return lo + k, sigma_b[k]

    return (otsu,)


@app.cell
def _(cv2, gray, hist, otsu):
    t, _ = otsu(hist)
    t_cv, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(t, t_cv)
    return (t,)


@app.cell
def _(gray, np, show, t):
    B = np.where(gray > t, 255, 0).astype(np.uint8)
    show(B, f"global Otsu, t = {t}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2. Бинаризация методом Оцу локальная (с предварительным разбиением на фрагменты - по каждому своя бинаризация)
    """)
    return


@app.cell
def _(histogram, np, otsu):
    def local_otsu(img, n, min_std=10):
        t_global, _ = otsu(histogram(img))
        B = np.zeros_like(img)
        T = np.zeros((n, n), dtype=int)

        rows = np.array_split(np.arange(img.shape[0]), n)
        cols = np.array_split(np.arange(img.shape[1]), n)
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                tile = img[r[0]:r[-1] + 1, c[0]:c[-1] + 1]
                if tile.std() < min_std:
                    t = t_global
                else:
                    t, _ = otsu(histogram(tile))
                T[i, j] = t
                B[r[0]:r[-1] + 1, c[0]:c[-1] + 1] = np.where(tile > t, 255, 0)
        return B, T

    return (local_otsu,)


@app.cell
def _(cv2, np):
    def draw_grid(img, n):
        vis = np.dstack([img] * 3)
        h, w = img.shape
        for k in range(1, n):
            x = round(k * w / n)
            y = round(k * h / n)
            vis = cv2.line(vis, (x, 0), (x, h), (255, 0, 0), 5)
            vis = cv2.line(vis, (0, y), (w, y), (255, 0, 0), 5)
        return vis

    return (draw_grid,)


@app.cell
def _(draw_grid, gray, local_otsu, show):
    for n in [2, 4, 8]:
        _B_local, _T = local_otsu(gray, n)
        show(draw_grid(_B_local, n), f"local Otsu, {n}x{n}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3. Бинаризация методом Оцу иерархическая (выполняется с иерархическим разбиением каждой уже полученной  области, пока возможно разбивать). Квантование с использованием найденных порогов
    """)
    return


@app.cell
def _(cv2, histogram, np, otsu):
    def hierarchical_otsu(hist, lo=0, hi=256, depth=0, max_depth=8, min_std=8):
        h = hist[lo:hi].astype(np.float64)
        if depth >= max_depth or np.count_nonzero(h) < 2:
            return []

        levels = np.arange(lo, hi)
        mean = (h * levels).sum() / h.sum()
        std = np.sqrt((h * (levels - mean) ** 2).sum() / h.sum())
        if std < min_std:
            return []

        t, _ = otsu(hist, lo, hi)
        left = hierarchical_otsu(hist, lo, t + 1, depth + 1, max_depth, min_std)
        right = hierarchical_otsu(hist, t + 1, hi, depth + 1, max_depth, min_std)
        return left + [t] + right


    def quantize_by_thresholds(img, thresholds):
        hist = histogram(img)
        bounds = [0] + [t + 1 for t in thresholds] + [256]

        lut = np.zeros(256, dtype=np.uint8)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            h = hist[lo:hi]
            levels = np.arange(lo, hi)
            lut[lo:hi] = round((h * levels).sum() / h.sum()) if h.sum() else lo
        return cv2.LUT(img, lut)

    return hierarchical_otsu, quantize_by_thresholds


@app.cell
def _(gray, hierarchical_otsu, hist, quantize_by_thresholds, show):
    for depth in [1, 2, 3]:
        ts = hierarchical_otsu(hist, max_depth=depth)
        _Q = quantize_by_thresholds(gray, ts)
        print(depth, ts)
        show(_Q, f"hierarchical Otsu, depth = {depth}, levels = {len(ts) + 1}")
    return


@app.cell
def _(gray, hierarchical_otsu, hist, quantize_by_thresholds, show):
    thresholds = hierarchical_otsu(hist, max_depth=16, min_std=12)
    Q_otsu = quantize_by_thresholds(gray, thresholds)
    print(len(thresholds) + 1, thresholds)

    show(Q_otsu, f"hierarchical Otsu, levels = {len(thresholds) + 1}")
    return (thresholds,)


@app.cell
def _(hist, np, plt, thresholds):
    _fig, _ax = plt.subplots(figsize=(12, 3))
    _ax.bar(np.arange(256), hist, width=1)
    for _t in thresholds:
        _ax.axvline(_t, color="r")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
