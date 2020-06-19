import pygame.freetype  # Import the freetype module.

"""
Author: Mohamed Afify
Date: May 2020

- This file displays a sodoku and shows the added numbers
- Can be used when testing the solving algorithm on grids (like the ones in grids.py)
    before using a camera.
- Uses pygame 
"""

def get_zeros(to_draw):
    list_z = []
    for row in range(0, 9):
        for col in range(0, 9):
            if to_draw[row][col] == 0:
                list_z.append([row, col])
    return list_z


def draw_sudoku(to_draw, zeros_indices, save=False):

    window_width, window_height = 450, 450
    black = (0, 0, 0)
    white = (200, 200, 200)
    gray = (50, 50, 50)
    blue = (62, 144, 225)
    rect_color = blue
    num_color = gray

    pygame.init()
    screen = pygame.display.set_mode((window_width, window_height))
    font = pygame.freetype.Font("Comfortaa.ttf", 24)
    background_image = pygame.image.load("sudoku-01.bmp").convert()

    margin = 50
    i = 0
    to_center = 18

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if i == 0:
            screen.blit(background_image, [0, 0])
            for row in to_draw:
                j = 0
                for num in row:
                    c = int(i/50)
                    r = int(j/50)
                    if [c, r] in zeros_indices:
                        rect = pygame.Rect(r * margin, c * margin, margin, margin)
                        pygame.draw.rect(screen, rect_color, rect, 0)
                    if num == 0:
                        font.render_to(screen, (j+to_center, i+to_center), ' ', num_color)
                    else:
                        font.render_to(screen, (j+to_center, i+to_center), str(num), num_color)
                    j += margin
                i += margin
            if save:
                pygame.image.save(screen, "sudoku.jpeg")
        pygame.display.update()
    pygame.quit()


