import cv2
from mid_end.vision import create_tile_frames, merge_tile_frames

image = cv2.imread(r"input\598x620.png")
size = (128, 8, 2)
tiles, width, height = create_tile_frames(image, size)
print(f"{(height,width)}")
temp_height = image.shape[0] * 1
temp_width = image.shape[1] * 1
pad_width = width * 1
pad_height = height * 1
size = (128 * 1, 8 * 1, 2 * 1)
merge_image = merge_tile_frames(
    tiles, temp_width, temp_height, pad_width, pad_height, size
)
cv2.imshow("image", merge_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
