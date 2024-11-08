from PIL import Image
import glob


def create_collage(images, grid_rows, grid_cols, padding=10):

    img = Image.open(images[0])
    img_width, img_height = img.size


    # Compute collage dimensions based on grid size provided
    collage_width = grid_cols * img_width + (grid_cols + 1) * padding
    collage_height = grid_rows * img_height + (grid_rows + 1) * padding


    # create empty collage
    collage = Image.new('RGB', (collage_width, collage_height), 'white')

    # Add the images onto the empty collage we created!
    for i, img_path in enumerate(images):
        img = Image.open(img_path).resize((img_width, img_height), Image.ANTIALIAS)
        x = (i % grid_cols) * (img_width + padding) + padding
        y = (i // grid_cols) * (img_height + padding) + padding
        collage.paste(img, (x,y))

    return collage

image_path = '/root/home/data_jpg'
images = glob.glob(f"{image_path}/*.jpeg")
images = sorted(images)[:6]

collage_2x3 = create_collage(images, 2, 3)
collage_2x3.save('collage_2x3.jpg')

collage_3x2 = create_collage(images, 3, 2)
collage_3x2.save('collage_3x2.jpg')

collage_1x6 = create_collage(images, 1, 6)
collage_1x6.save('collage_1x6.jpg')

collage_6x1 = create_collage(images, 6, 1)
collage_6x1.save('collage_6x1.jpg')

