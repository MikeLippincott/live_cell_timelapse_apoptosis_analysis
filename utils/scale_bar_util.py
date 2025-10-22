import PIL
from PIL import ImageDraw


def add_scale_bar(
    image: PIL.Image.Image,
    pixel_size_um: float,
    scale_bar_length_um: int = 50,  # um
    scale_bar_height_px: int = 5,  # pixels
    print_text: bool = False,
    padding: int = 10,
) -> PIL.Image.Image:
    """
    Add a scale bar to the image.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image to add the scale bar to.
        Can be single channel or composite.
    pixel_size_um : float
        Pixel size in micrometers
    scale_bar_length_um : int, optional
        Length of the scale bar in micrometers, by default 50
    scale_bar_height_px : int, optional
        Height of the scale bar in pixels, by default 5
    print_text : bool, optional
        Whether to print the scale bar length as text on the image, by default False
    padding : int, optional
        Padding around the scale bar in pixels, by default 10
        Padding is the distance from the edge of the image to the scale bar.

    Returns
    -------
    PIL.Image.Image
        Image with the scale bar added.

    Raises
    ------
    ValueError
        If the scale bar length is not positive.
    ValueError
        If the scale bar height is not positive.
    ValueError
        If the pixel size is not positive.
    """
    # check valid inputs
    if scale_bar_length_um <= 0:
        raise ValueError("Scale bar length must be positive.")
    if scale_bar_height_px <= 0:
        raise ValueError("Scale bar height must be positive.")
    if pixel_size_um <= 0:
        raise ValueError("Pixel size must be positive.")
    # Create a copy of the image to draw on
    image_with_scale_bar = image.copy()
    draw = ImageDraw.Draw(image_with_scale_bar)

    # Calculate the scale bar length in pixels
    scale_bar_length_px = int(scale_bar_length_um / pixel_size_um)

    # 0,0 is top left for images
    # so the bottom right is the max coordinates (or the size of the image)
    x_max_pixel = image.size[0]
    y_max_pixel = image.size[1]

    # Draw the scale bar
    draw.rectangle(
        [
            x_max_pixel - padding - scale_bar_length_px,  # x position start
            y_max_pixel - padding - scale_bar_height_px,  # y position start
            x_max_pixel - padding,  # x position end
            y_max_pixel - padding,  # y position end
        ],
        fill="white",
    )

    # Optionally, add text
    if print_text:
        draw.text((padding, padding), f"{scale_bar_length_um} um", fill="white")

    return image_with_scale_bar
