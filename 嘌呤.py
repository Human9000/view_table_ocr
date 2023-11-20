import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
import matplotlib

matplotlib.use('TkAgg')
ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", page_num=1, show_log=False)

DEBUG = False
LOG = True


def mean01_dim(img, dim=0, left=0, right=1, mask_zeros=None):
    d = img.shape[dim]
    lr = [int(d * left), int(d * right)]
    _, x, _ = np.split(img, indices_or_sections=lr, axis=dim)
    x = x.mean(axis=dim)
    if mask_zeros is not None:
        x[mask_zeros==0] = x.max()
    x = x - x.min()
    x = x / x.max()
    return x


def min01_dim(img, dim=0, left=0, right=1):
    d = img.shape[dim]
    lr = [int(d * left), int(d * right)]
    _, x, _ = np.split(img, indices_or_sections=lr, axis=dim)
    x = x.min(axis=dim)
    x = x - x.min()
    x = x /(x.max() + 1e-9)
    return x


def show_img(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def split_head_body(img):
    assert len(img.shape) == 2, "img must only have 2 dims"
    # 裁剪到上边
    x = mean01_dim(img, dim=-1, left=0.45, right=0.55)
    index = np.where(x < 0.6)[0]
    w2 = index[-1]
    w0=w1 = w2
    for i in range(len(index)):
        if index[-i] - index[-i - 1] > 100:
            w1 = index[-i - 1]
            break
    for i in range(w1, 0, -1):
        if x[i] > 0.95:
            w0 = i
            break
    assert w1 != w0, "head not found: w1 != w0"
    if DEBUG and True:
        plt.plot(x)
        plt.axvline(x=w0, linestyle='--', color='r')
        plt.axvline(x=w1, linestyle='--', color='r')
        plt.axvline(x=w2, linestyle='--', color='r')
        plt.show()
        plt.imshow(img)
        plt.axhline(y=w0, linestyle='--', color='r')
        plt.axhline(y=w1, linestyle='--', color='r')
        plt.axhline(y=w2, linestyle='--', color='r')
        plt.show()
    return w0, w1, w2


def split_col(head, body):
    w = head.shape[0]
    h = head.shape[1]
    head_m = head.mean(axis=-1)
    head_m = cv.blur(head_m, (3, 3)).astype(np.uint8)
    retVal, _ = cv.threshold(head_m[:, int(h * 0.3):int(h * 0.8)],
                             0,
                             255,
                             cv.THRESH_OTSU)
    head_t = (head_m > retVal).astype(np.uint8)

    # 定义膨胀操作的结构元素
    x = mean01_dim(head_t, dim=0)
    index = np.where(x < 0.98)[0]
    l = np.min(index)
    r = np.max(index)
    l0 = [265, 355,  800, 1000, 1170, 1430, 1650, 1850, 2130]
    rate = (r-l)/2132
    mid = [int(i*rate)+l for i in l0]
    mid = np.array(mid)
    return mid


def split_row(id_rol):
    id_rol = id_rol.mean(-1)
    id_rol = cv.blur(id_rol, (3, 3)).astype(np.uint8)
    retVal, id_rol = cv.threshold(id_rol,
                                  0,
                                  255,
                                  cv.THRESH_OTSU)
    id_rol[:20] = 255
    id_rol[-10:] = 255
    x = min01_dim(cv.erode(id_rol, np.ones((7, 7), np.uint8)),
                   1, left=0.5, right=1)

    if DEBUG:
        plt.plot(x)
        plt.show()
    xdiff = np.diff(x)
    xdiff[xdiff < -0.4] = -0.4
    xdiff[xdiff > 0] = 0
    # 非极小值抑制，r = 70
    mask = np.zeros_like(xdiff)
    for i in range(len(xdiff)):
        l = i-150 if i-150 > 0 else 0
        r = i+100 if i+100 < len(xdiff) else len(xdiff)
        min_ = xdiff[l:r].min()
        if xdiff[i] == min_:
            mask[i] = 1

    index = np.where(mask>0)[0]

    res = [i-10 for id, i in enumerate(index) if abs( i - index[id-1] )> 40]

    if DEBUG:
        plt.imshow(id_rol)
        for i in res:
            plt.axhline(i, color='r')
        plt.show()
    return res


def table_struct(img, page):
    try:
        i0, i1, i2 = split_head_body(img.mean(axis=-1))  # 输入灰度图
        assert (i0 < i1 < 700 < i2 and i1 - i0 > 150 and i2 - i1 > 50), \
            'head is not found: (i0 < i1 < 700 < i2 and i1 - i0 > 150 and i2 - i1 > 50)'
    except Exception as e:
        print("Error:", e)
        if DEBUG:
            plt.imshow(img)
            plt.show()
        return False
    cols = split_col(img[i0:i1], img[i1:i2])
    print("----\t", len(cols))

    rows = split_row(img[i1:i2, cols[0]:cols[1]])
    rows = [i + i1 - 2 for i in rows]
    if LOG:
        fig = plt.figure()
        plt.imshow(img)
        for i in range(len(cols)):
            plt.axvline(x=cols[i], linestyle='--', color='r')
        for i in [i0, i1, i2]:
            plt.axhline(y=i, linestyle='--', color='r')
        for i in rows:
            plt.axhline(y=i, linestyle='--', color='g')
        fig.savefig(f'log/{page}.png')
        cv.imwrite(f'log2/{page}.jpg', img)
        if DEBUG:
            plt.show()
        plt.close(fig)
    return [[i0, i1, i2], cols, rows]


def ocr(img):
    result = ""
    res = ocr_engine(img)
    # print(res[1])
    if len(res[1]) > 0:
        for r in res[1]:
            result += r[0]
    return result


def cv2_imread(filepath):
    cv2_img = cv.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
    return cv2_img


def check_body(img):
    img = 255*(img>180).astype(np.uint8)
    x = min01_dim(img, 1)
    if DEBUG:
        plt.plot(x)
        plt.show()
    if x.mean() > 0.1:
        return True
    else:
        return False


if __name__ == '__main__':
    save_folder = './output'
    root = r"E:\Seafile\07-参考文献\食物数据\第二册"
    for i in os.listdir(root):
        path = root + "\\" + i
        if not i.endswith('.png'):
            continue
        page = int(i[:-4].split('_')[-1])
        print(page, end='\t')
        # if page != 372:
        #     continue
        if page < 371 or page > 384:
            continue
        print(page, end='\t')
        img = cv2_imread(path)  # 输入的图片是调整后方向的图片，高*宽*3
        res = table_struct(img,page)
        # break
        if not res:
            print("Error")
            continue
        [i0, i1, i2], cols, rows = res
        print(cols)
        rows.append(i2)
        n, m = len(cols) - 1, len(rows) - 1
        data_result = []
        for r in range(m):
            if check_body(img[rows[r]:rows[r + 1], cols[2]:cols[-1]]) is False:
                continue
            data_result.append([""] * n)
            for c in range(n):
                item = img[rows[r]:rows[r + 1], cols[c]:cols[c + 1]]
                data_result[-1][c] = ocr(item)
        df = pd.DataFrame(data_result)
        df.to_csv(f"o4/{page}.tsv", index=False, sep='\t')
        print(df)
