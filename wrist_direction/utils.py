def label_from_signed_area(area):
    """
    area > 0 => CCW (label 1)
    area < 0 => CW (label 0)
    """
    return 1 if area > 0 else 0
