from pathlib import Path
from rembg import remove, new_session

session = new_session(model_name='birefnet-portrait')

# First remove background and save the image without background
for file in Path('input').glob('*.jpg'):
    input_path = str(file)
    output_path_no_bg = str(
        Path('output') / (file.stem + "_no_bg.jpg"))

    with open(input_path, 'rb') as i:
        with open(output_path_no_bg, 'wb') as o:
            input = i.read()
            output = remove(input, session=session)
            o.write(output)

# Then create a mask from the image without background
for file in Path('output').glob('*_no_bg.jpg'):
    input_path_no_bg = str(file)
    output_path_mask = str(
        Path('output') / (file.stem + "_mask.png"))

    with open(input_path_no_bg, 'rb') as i:
        with open(output_path_mask, 'wb') as o:
            input = i.read()
            output = remove(input, session=session, only_mask=True)
            o.write(output)
