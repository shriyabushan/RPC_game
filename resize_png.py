import cv2

# List of your images
files = ["Resources/1.png", "Resources/2.png", "Resources/3.png"]

# Set the new size you want (e.g. 200x200)
new_size = (300, 300)

for file in files:
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)  # read with alpha
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(file, resized)
    print(f"{file} resized to {new_size}")
