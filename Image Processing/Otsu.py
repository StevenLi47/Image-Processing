import numpy as np
import cv2
import time

def multi_otsu(image, mask = None, num_thresh = 1):

    def calc_var(iter = 0, lower = 0, var = 0):
        upper = 129 - num_thresh + iter
        thresh_mean = np.divide(np.cumsum(inten_sum[lower:upper]), np.cumsum(hist[lower:upper]))
        b_var = np.multiply(np.cumsum(prob[lower:upper]), np.square(thresh_mean - total_mean))
        max_var = var
        max_thresh = []
        for val in range(lower + 1, upper):
            cur_var = b_var[val - lower - 1]
            if cur_var != 0:
                if iter != num_thresh - 1:
                    final_var, final_thresh = calc_var(iter = iter + 1, lower = val, var = var + cur_var)
                    if final_var > max_var:
                        max_var = final_var
                        max_thresh = final_thresh + [val * 2]
                else:
                    final_var = var + cur_var + end_thresh[val]
                    if final_var > max_var:
                        max_var = final_var
                        max_thresh = [val * 2]

        return max_var, max_thresh

    hist = cv2.calcHist([image], [0], mask, [128], [0, 256]).flatten()
    total_pixels = np.sum(hist)
    prob = hist / total_pixels
    inten_sum = np.multiply(np.arange(0, 256, 2), hist)
    total_mean = np.mean(image)
    end_thresh = np.multiply(np.cumsum(prob[::-1])[::-1], np.square(np.divide(np.cumsum(inten_sum[::-1])[::-1], np.cumsum(hist[::-1])[::-1]) - total_mean))

    return calc_var()


image = cv2.imread('cat6.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

start_time = time.time()
variance, thresh_val = multi_otsu(image, num_thresh=4)
final_time = time.time() - start_time
print(final_time)
print(thresh_val)

canvas = np.zeros_like(image)

inten_val = [50, 80, 100, 255]

for t, val in zip(thresh_val[::-1], inten_val):
    ret, thresh_im = cv2.threshold(image, t, val, cv2.THRESH_BINARY)
    canvas = cv2.bitwise_or(canvas, thresh_im)

cv2.imshow('test', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()