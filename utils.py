

# It doesn't match torchvision.transforms.functional.center_crop
# but strangely using the latter produces bad result after denoising
def manual_center_crop(im):

    width, height = im.size
    d = min(width, height)
    left = (width - d) / 2
    upper = (height - d) / 2
    right = (width + d) / 2
    lower = (height + d) / 2

    return im.crop((left, upper, right, lower))