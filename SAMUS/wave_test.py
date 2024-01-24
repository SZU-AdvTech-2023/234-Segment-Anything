import numpy as np
from matplotlib import pyplot as plt
import pywt
import PIL

img = PIL.Image.open("/data/gjx/project/dataset/CAMUS_256/tvt/train/img/patient0001_2CH0000.png")
# img = np.array(img)[:, :, 0]
plt.imshow(img, cmap='gray')
plt.show()
LLY, (LHY, HLY, HHY) = pywt.dwt2(img, 'haar')
plt.subplot(2, 2, 1)
plt.imshow(LLY, cmap="Greys")
plt.subplot(2, 2, 2)
plt.imshow(LHY, cmap="Greys")
plt.subplot(2, 2, 3)
plt.imshow(HLY, cmap="Greys")
plt.subplot(2, 2, 4)
plt.imshow(HHY, cmap="Greys")
plt.show()
