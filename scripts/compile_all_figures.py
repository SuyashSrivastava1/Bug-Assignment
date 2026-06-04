from PIL import Image
from pathlib import Path

fig_dir = Path('documentation/figures')
image_files = sorted(list(fig_dir.glob('*.png')))

if not image_files:
    print("No PNG files found to compile.")
else:
    images = []
    first_image = None
    
    for img_path in image_files:
        img = Image.open(img_path)
        img = img.convert('RGB')
        if first_image is None:
            first_image = img
        else:
            images.append(img)
            
    output_pdf = fig_dir / 'all_figures_compiled.pdf'
    
    first_image.save(
        output_pdf,
        "PDF",
        resolution=100.0,
        save_all=True,
        append_images=images
    )
    print(f"Successfully compiled {len(image_files)} figures into {output_pdf}")
