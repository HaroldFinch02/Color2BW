import cv2
import glob

count = 1

path = 'train_data/originalImages/*.*'

filenames = glob.glob(path)

for filename in filenames:
    print(filename)
    image = cv2.imread(filename)

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (128, 128))
    gray_img = cv2.resize(gray_img, (128, 128))

    done = cv2.imwrite(
        "train_data/gray_images/gray_" + str(count) + ".jpg", gray_img)
    print(done)
    cv2.imwrite("train_data/color_images/color_" +
                str(count) + ".jpg", image)

    count += 1
