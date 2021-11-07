from PIL import Image, ImageFilter


def imageprepare(img_data):
    read_image = Image.open(img_data).convert('L')
    w_image = float(read_image.size[0])
    h_image = float(read_image.size[1])
    updated_image = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if w_image > h_image:
        # dimension check whether it is bigger
        # Width is bigger. Width becomes 20 pixels.
        nh_img = int(round((20.0 / w_image * h_image), 0))  # resize height according to ratio width
        if (nh_img == 0):  # rare case but minimum is 1 pixel
            nh_img = 1
            # resize and sharpen
        img = read_image.resize((20, nh_img), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nh_img) / 2), 0))  # position: horizontal
        updated_image.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nw_img = int(round((20.0 / h_image * w_image), 0))  # resize width according to ratio height
        if (nw_img == 0):  # rare case but minimum is 1 pixel
            nw_img = 1
            # resize and sharpen
        img = read_image.resize((nw_img, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nw_img) / 2), 0))  # caculate vertical pozition
        updated_image.paste(img, (wleft, 4))  # paste resized image on white canvas

    p_image = list(updated_image.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    pixel_image = [(255 - x) * 1.0 / 255.0 for x in p_image]
    return pixel_image