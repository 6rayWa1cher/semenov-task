import random


def generate_position_within_borders(from_pos, size, begin_padding, end_padding):
    return from_pos + random.randrange(begin_padding, size - end_padding)
