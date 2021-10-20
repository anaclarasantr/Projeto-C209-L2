import cv2
import numpy as np
import scipy.fftpack as fftpack
import zlib

def encode_ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

def decode_ycbcr(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

def encode_dct(orig, bx, by):
    new_shape = (
        orig.shape[0] // bx * bx,
        orig.shape[1] // by * by,
        3
    )
    new = orig[
        :new_shape[0],
        :new_shape[1]
    ].reshape((
        new_shape[0] // bx,
        bx,
        new_shape[1] // by,
        by,
        3
    ))
    return fftpack.dctn(new, axes=[1,3], norm='ortho')


def decode_dct(orig, bx, by):
    return fftpack.idctn(orig, axes=[1,3], norm='ortho'
    ).reshape((
        orig.shape[0]*bx,
        orig.shape[2]*by,
        3
    ))

def encode_quant(orig, quant):
    return (orig / quant).astype(np.int)


def decode_quant(orig, quant):
    return (orig * quant).astype(float)

def encode_zip(x):
    return zlib.compress(x.astype(np.int8).tobytes())


def decode_zip(orig, shape):
    return np.frombuffer(zlib.decompress(orig), dtype=np.int8).astype(float).reshape(shape)

if __name__ == '__main__':
    im = cv2.imread("inatel.jpg")
    quants = [5, 10]
    blocks = [(16, 16), (32, 32)]
    for qscale in quants:
        for bx, by in blocks:

            quant = (
                (np.ones((bx, by)) * (qscale * qscale))
                .clip(-100, 100)
                .reshape((1, bx, 1, by, 1))
            )
            ency = encode_ycbcr(im)
            encd = encode_dct(ency, bx, by)
            encq = encode_quant(encd, quant)
            encz = encode_zip(encq)
            decz = decode_zip(encz, encq.shape)
            decq = decode_quant(encq, quant)
            decd = decode_dct(decq, bx, by)
            decy = decode_ycbcr(decd)
            cv2.imwrite("inatel_quant_{}_block_{}x{}.jpeg".format(qscale, bx, by), decy.astype(np.uint8))