import threading
import pygame
import sys
import cv2
import numpy as np
import time


class TextBox:
    def __init__(self):
        self.width = 200
        self.height = 30
        self.x = 950
        self.y = 100
        self.text = ""  # 文本框内容
        self.__surface = pygame.Surface((self.width, self.height))
        self.font = pygame.font.Font(None, 32)  # 使用pygame自带字体

    def draw_text(self, dest_surf):
        text_surf = self.font.render(self.text, True, (255, 255, 255))
        dest_surf.blit(self.__surface, (self.x, self.y))
        dest_surf.blit(text_surf, (self.x, self.y + (self.height - text_surf.get_height())),
                       (0, 0, self.width, self.height))

    def key_down(self, event):
        unicode = event.unicode
        key = event.key

        # 退位键
        if key == 8:
            self.text = self.text[:-1]
            return

        if unicode != "":
            char = unicode
        else:
            char = chr(key)
        self.text += char


class Button():
    def __init__(self, screen, msg):
        self.screen = screen
        self.width = 200
        self.height = 50
        self.button_color = (193, 210, 240)
        self.text_color = (0, 0, 0)
        self.font = pygame.font.Font(None, 36)
        self.rect = pygame.Rect(950, 200, self.width, self.height)
        self.msg_image = self.font.render(msg, True, self.text_color, self.button_color)
        self.msg_image_rect = (980, 210)

    def draw_button(self):
        self.screen.fill(self.button_color, self.rect)
        self.screen.blit(self.msg_image, self.msg_image_rect)


class Label():
    def __init__(self, w, h, screen, msg):
        self.screen = screen
        self.font = pygame.font.Font(None, 36)
        self.msg = msg
        self.obj = self.font.render(msg, True, (255, 10, 10))
        self.rect = (w, h)

    def draw_text(self):
        self.screen.blit(self.obj, self.rect)

    def change_msg(self, msg):
        self.msg = msg


def calc():
    in_image = cv2.imread("in.jpg").astype(np.float64)
    in_height, in_width = in_image.shape[: 2]
    global now_height
    global now_width
    global out_height
    global out_width
    global pic_state
    print("height:", in_height, " width:", in_width)

    kernel_x = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
    kernel_y_left = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
    kernel_y_right = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)
    out_image = np.copy(in_image)
    delta_height = int(out_height - in_height)
    delta_width = int(out_width - in_width)
    # remove width
    if delta_width < 0:
        print("remove width")
        delta_width = -delta_width
        for num_pixel in range(delta_width):
            (B_image, G_image, R_image) = cv2.split(out_image)
            B_energy = np.absolute(cv2.Scharr(B_image, -1, 1, 0)) + np.absolute(cv2.Scharr(B_image, -1, 0, 1))
            G_energy = np.absolute(cv2.Scharr(G_image, -1, 1, 0)) + np.absolute(cv2.Scharr(G_image, -1, 0, 1))
            R_energy = np.absolute(cv2.Scharr(R_image, -1, 1, 0)) + np.absolute(cv2.Scharr(R_image, -1, 0, 1))
            energy_map = B_energy + G_energy + R_energy

            now_height, now_width = energy_map.shape
            matrix_x = np.absolute(cv2.filter2D(B_image, -1, kernel=kernel_x)) + \
                       np.absolute(cv2.filter2D(G_image, -1, kernel=kernel_x)) + \
                       np.absolute(cv2.filter2D(R_image, -1, kernel=kernel_x))
            matrix_y_left = np.absolute(cv2.filter2D(B_image, -1, kernel=kernel_y_left)) + \
                            np.absolute(cv2.filter2D(G_image, -1, kernel=kernel_y_left)) + \
                            np.absolute(cv2.filter2D(R_image, -1, kernel=kernel_y_left))
            matrix_y_right = np.absolute(cv2.filter2D(B_image, -1, kernel=kernel_y_right)) + \
                             np.absolute(cv2.filter2D(G_image, -1, kernel=kernel_y_right)) + \
                             np.absolute(cv2.filter2D(R_image, -1, kernel=kernel_y_right))

            cumulative_map = np.copy(energy_map)
            for row in range(1, now_height):
                for col in range(now_width):
                    if col == 0:
                        e_right = cumulative_map[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[
                            row - 1, col + 1]
                        e_up = cumulative_map[row - 1, col] + matrix_x[row - 1, col]
                        cumulative_map[row, col] = energy_map[row, col] + min(e_right, e_up)
                    elif col == now_width - 1:
                        e_left = cumulative_map[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[
                            row - 1, col - 1]
                        e_up = cumulative_map[row - 1, col] + matrix_x[row - 1, col]
                        cumulative_map[row, col] = energy_map[row, col] + min(e_left, e_up)
                    else:
                        e_left = cumulative_map[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[
                            row - 1, col - 1]
                        e_right = cumulative_map[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[
                            row - 1, col + 1]
                        e_up = cumulative_map[row - 1, col] + matrix_x[row - 1, col]
                        cumulative_map[row, col] = energy_map[row, col] + min(e_left, e_right, e_up)

            seam_idx = np.zeros((now_height,), dtype=np.uint32)
            seam_idx[-1] = np.argmin(cumulative_map[-1])
            for row in range(now_height - 2, -1, -1):
                prv_x = seam_idx[row + 1]
                if prv_x == 0:
                    seam_idx[row] = np.argmin(cumulative_map[row, : 2])
                else:
                    seam_idx[row] = np.argmin(
                        cumulative_map[row, prv_x - 1: min(prv_x + 2, now_width - 1)]) + prv_x - 1

            show_tmp_image = np.copy(out_image)
            for row in range(now_height):
                col = seam_idx[row]
                show_tmp_image[row, col, 0] = 0
                show_tmp_image[row, col, 1] = 0
                show_tmp_image[row, col, 2] = 255
            cv2.imwrite("tmp.jpg", show_tmp_image.astype(np.uint8))
            pic_state = 1

            output = np.zeros((now_height, now_width - 1, 3))
            for row in range(now_height):
                col = seam_idx[row]
                output[row, :, 0] = np.delete(out_image[row, :, 0], [col])
                output[row, :, 1] = np.delete(out_image[row, :, 1], [col])
                output[row, :, 2] = np.delete(out_image[row, :, 2], [col])
            out_image = output
        cv2.imwrite("out.jpg", out_image.astype(np.uint8))
    # insert width
    elif delta_width > 0:
        print("insert width")
        seams_record = []
        temp_image = np.copy(out_image)
        for num_pixel in range(delta_width):
            (B_image, G_image, R_image) = cv2.split(out_image)
            B_energy = np.absolute(cv2.Scharr(B_image, -1, 1, 0)) + np.absolute(cv2.Scharr(B_image, -1, 0, 1))
            G_energy = np.absolute(cv2.Scharr(G_image, -1, 1, 0)) + np.absolute(cv2.Scharr(G_image, -1, 0, 1))
            R_energy = np.absolute(cv2.Scharr(R_image, -1, 1, 0)) + np.absolute(cv2.Scharr(R_image, -1, 0, 1))
            energy_map = B_energy + G_energy + R_energy
            now_height, now_width = energy_map.shape
            cumulative_map = np.copy(energy_map)
            for row in range(1, now_height):
                for col in range(now_width):
                    cumulative_map[row, col] = \
                        energy_map[row, col] + np.amin(
                            cumulative_map[row - 1, max(col - 1, 0): min(col + 2, now_width - 1)])
            seam_idx = np.zeros((now_height,), dtype=np.uint32)
            seam_idx[-1] = np.argmin(cumulative_map[-1])
            for row in range(now_height - 2, -1, -1):
                prv_x = seam_idx[row + 1]
                if prv_x == 0:
                    seam_idx[row] = np.argmin(cumulative_map[row, : 2])
                else:
                    seam_idx[row] = np.argmin(
                        cumulative_map[row, prv_x - 1: min(prv_x + 2, now_width - 1)]) + prv_x - 1
            seams_record.append(seam_idx)

            show_tmp_image = np.copy(out_image)
            for row in range(now_height):
                col = seam_idx[row]
                show_tmp_image[row, col, 0] = 0
                show_tmp_image[row, col, 1] = 0
                show_tmp_image[row, col, 2] = 255
            cv2.imwrite("tmp.jpg", show_tmp_image.astype(np.uint8))
            pic_state = 1

            output = np.zeros((now_height, now_width - 1, 3))
            for row in range(now_height):
                col = seam_idx[row]
                output[row, :, 0] = np.delete(out_image[row, :, 0], [col])
                output[row, :, 1] = np.delete(out_image[row, :, 1], [col])
                output[row, :, 2] = np.delete(out_image[row, :, 2], [col])
            out_image = output
        out_image = np.copy(temp_image)
        for num_pixel in range(len(seams_record)):
            seam = seams_record.pop(0)

            now_height, now_width = out_image.shape[: 2]
            output = np.zeros((now_height, now_width + 1, 3))
            for row in range(now_height):
                col = seam[row]
                for ch in range(3):
                    if col == 0:
                        p = np.average(out_image[row, col: col + 2, ch])
                        output[row, col, ch] = out_image[row, col, ch]
                        output[row, col + 1, ch] = p
                        output[row, col + 1:, ch] = out_image[row, col:, ch]
                    else:
                        p = np.average(out_image[row, col - 1: col + 1, ch])
                        output[row, : col, ch] = out_image[row, : col, ch]
                        output[row, col, ch] = p
                        output[row, col + 1:, ch] = out_image[row, col:, ch]
            out_image = np.copy(output)

            new_seams_record = []
            for this_seam in seams_record:
                this_seam[np.where(this_seam >= seam)] += 2
                new_seams_record.append(this_seam)
            seams_record = new_seams_record
            cv2.imwrite("tmp.jpg", out_image.astype(np.uint8))
        cv2.imwrite("out.jpg", out_image.astype(np.uint8))
    else:
        print("not weight")

    if delta_height < 0:
        print("remove height")
        now_height, now_width, now_ch = out_image.shape
        trans_image = np.zeros((now_width, now_height, now_ch))
        image_flip = np.fliplr(out_image)
        for c in range(now_ch):
            for row in range(now_height):
                trans_image[:, row, c] = image_flip[row, :, c]

        out_image = trans_image
        delta_height = -delta_height
        for num_pixel in range(delta_height):
            (B_image, G_image, R_image) = cv2.split(out_image)
            B_energy = np.absolute(cv2.Scharr(B_image, -1, 1, 0)) + np.absolute(cv2.Scharr(B_image, -1, 0, 1))
            G_energy = np.absolute(cv2.Scharr(G_image, -1, 1, 0)) + np.absolute(cv2.Scharr(G_image, -1, 0, 1))
            R_energy = np.absolute(cv2.Scharr(R_image, -1, 1, 0)) + np.absolute(cv2.Scharr(R_image, -1, 0, 1))
            energy_map = B_energy + G_energy + R_energy

            trans_height, trans_width = energy_map.shape
            now_height = trans_width
            now_width = trans_height
            matrix_x = np.absolute(cv2.filter2D(B_image, -1, kernel=kernel_x)) + \
                       np.absolute(cv2.filter2D(G_image, -1, kernel=kernel_x)) + \
                       np.absolute(cv2.filter2D(R_image, -1, kernel=kernel_x))
            matrix_y_left = np.absolute(cv2.filter2D(B_image, -1, kernel=kernel_y_left)) + \
                            np.absolute(cv2.filter2D(G_image, -1, kernel=kernel_y_left)) + \
                            np.absolute(cv2.filter2D(R_image, -1, kernel=kernel_y_left))
            matrix_y_right = np.absolute(cv2.filter2D(B_image, -1, kernel=kernel_y_right)) + \
                             np.absolute(cv2.filter2D(G_image, -1, kernel=kernel_y_right)) + \
                             np.absolute(cv2.filter2D(R_image, -1, kernel=kernel_y_right))

            cumulative_map = np.copy(energy_map)
            for row in range(1, trans_height):
                for col in range(trans_width):
                    if col == 0:
                        e_right = cumulative_map[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[
                            row - 1, col + 1]
                        e_up = cumulative_map[row - 1, col] + matrix_x[row - 1, col]
                        cumulative_map[row, col] = energy_map[row, col] + min(e_right, e_up)
                    elif col == trans_width - 1:
                        e_left = cumulative_map[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[
                            row - 1, col - 1]
                        e_up = cumulative_map[row - 1, col] + matrix_x[row - 1, col]
                        cumulative_map[row, col] = energy_map[row, col] + min(e_left, e_up)
                    else:
                        e_left = cumulative_map[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[
                            row - 1, col - 1]
                        e_right = cumulative_map[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[
                            row - 1, col + 1]
                        e_up = cumulative_map[row - 1, col] + matrix_x[row - 1, col]
                        cumulative_map[row, col] = energy_map[row, col] + min(e_left, e_right, e_up)

            seam_idx = np.zeros((trans_height,), dtype=np.uint32)
            seam_idx[-1] = np.argmin(cumulative_map[-1])
            for row in range(trans_height - 2, -1, -1):
                prv_x = seam_idx[row + 1]
                if prv_x == 0:
                    seam_idx[row] = np.argmin(cumulative_map[row, : 2])
                else:
                    seam_idx[row] = np.argmin(
                        cumulative_map[row, prv_x - 1: min(prv_x + 2, trans_width - 1)]) + prv_x - 1

            show_tmp_image = np.copy(out_image)
            for row in range(trans_height):
                col = seam_idx[row]
                show_tmp_image[row, col, 0] = 0
                show_tmp_image[row, col, 1] = 0
                show_tmp_image[row, col, 2] = 255
            trans_image = np.zeros((trans_width, trans_height, now_ch))
            for c in range(now_ch):
                for row in range(trans_height):
                    trans_image[:, trans_height - 1 - row, c] = show_tmp_image[row, :, c]
            cv2.imwrite("tmp.jpg", trans_image.astype(np.uint8))
            pic_state = 1

            output = np.zeros((trans_height, trans_width - 1, 3))
            for row in range(trans_height):
                col = seam_idx[row]
                output[row, :, 0] = np.delete(out_image[row, :, 0], [col])
                output[row, :, 1] = np.delete(out_image[row, :, 1], [col])
                output[row, :, 2] = np.delete(out_image[row, :, 2], [col])
            out_image = output

        trans_height, trans_width, trans_ch = out_image.shape
        trans_image = np.zeros((trans_width, trans_height, trans_ch))
        for c in range(trans_ch):
            for row in range(trans_height):
                trans_image[:, trans_height - 1 - row, c] = out_image[row, :, c]
        out_image = trans_image
        cv2.imwrite("out.jpg", out_image.astype(np.uint8))
    elif delta_height > 0:
        print("insert height")
        seams_record = []
        now_height, now_width, now_ch = out_image.shape
        trans_image = np.zeros((now_width, now_height, now_ch))
        image_flip = np.fliplr(out_image)
        for c in range(now_ch):
            for row in range(now_height):
                trans_image[:, row, c] = image_flip[row, :, c]
        out_image = trans_image
        temp_image = np.copy(out_image)

        for num_pixel in range(delta_height):
            (B_image, G_image, R_image) = cv2.split(out_image)
            B_energy = np.absolute(cv2.Scharr(B_image, -1, 1, 0)) + np.absolute(cv2.Scharr(B_image, -1, 0, 1))
            G_energy = np.absolute(cv2.Scharr(G_image, -1, 1, 0)) + np.absolute(cv2.Scharr(G_image, -1, 0, 1))
            R_energy = np.absolute(cv2.Scharr(R_image, -1, 1, 0)) + np.absolute(cv2.Scharr(R_image, -1, 0, 1))
            energy_map = B_energy + G_energy + R_energy

            trans_height, trans_width = energy_map.shape
            now_height = trans_width
            now_width = trans_height

            cumulative_map = np.copy(energy_map)
            for row in range(1, trans_height):
                for col in range(trans_width):
                    cumulative_map[row, col] = \
                        energy_map[row, col] + np.amin(
                            cumulative_map[row - 1, max(col - 1, 0): min(col + 2, trans_width - 1)])
            seam_idx = np.zeros((trans_height,), dtype=np.uint32)
            seam_idx[-1] = np.argmin(cumulative_map[-1])
            for row in range(trans_height - 2, -1, -1):
                prv_x = seam_idx[row + 1]
                if prv_x == 0:
                    seam_idx[row] = np.argmin(cumulative_map[row, : 2])
                else:
                    seam_idx[row] = np.argmin(
                        cumulative_map[row, prv_x - 1: min(prv_x + 2, trans_width - 1)]) + prv_x - 1
            seams_record.append(seam_idx)

            show_tmp_image = np.copy(out_image)
            for row in range(trans_height):
                col = seam_idx[row]
                show_tmp_image[row, col, 0] = 0
                show_tmp_image[row, col, 1] = 0
                show_tmp_image[row, col, 2] = 255

            trans_image = np.zeros((trans_width, trans_height, now_ch))
            for c in range(now_ch):
                for row in range(trans_height):
                    trans_image[:, trans_height - 1 - row, c] = show_tmp_image[row, :, c]
            cv2.imwrite("tmp.jpg", trans_image.astype(np.uint8))
            pic_state = 1

            output = np.zeros((trans_height, trans_width - 1, 3))
            for row in range(trans_height):
                col = seam_idx[row]
                output[row, :, 0] = np.delete(out_image[row, :, 0], [col])
                output[row, :, 1] = np.delete(out_image[row, :, 1], [col])
                output[row, :, 2] = np.delete(out_image[row, :, 2], [col])
            out_image = output
        out_image = np.copy(temp_image)
        for num_pixel in range(len(seams_record)):
            seam = seams_record.pop(0)

            trans_height, trans_width = out_image.shape[: 2]
            output = np.zeros((trans_height, trans_width + 1, 3))
            for row in range(trans_height):
                col = seam[row]
                for ch in range(3):
                    if col == 0:
                        p = np.average(out_image[row, col: col + 2, ch])
                        output[row, col, ch] = out_image[row, col, ch]
                        output[row, col + 1, ch] = p
                        output[row, col + 1:, ch] = out_image[row, col:, ch]
                    else:
                        p = np.average(out_image[row, col - 1: col + 1, ch])
                        output[row, : col, ch] = out_image[row, : col, ch]
                        output[row, col, ch] = p
                        output[row, col + 1:, ch] = out_image[row, col:, ch]
            out_image = np.copy(output)

            new_seams_record = []
            for this_seam in seams_record:
                this_seam[np.where(this_seam >= seam)] += 2
                new_seams_record.append(this_seam)
            seams_record = new_seams_record

            trans_height, trans_width, trans_ch = out_image.shape
            trans_image = np.zeros((trans_width, trans_height, now_ch))
            for c in range(now_ch):
                for row in range(trans_height):
                    trans_image[:, trans_height - 1 - row, c] = out_image[row, :, c]
            cv2.imwrite("tmp.jpg", trans_image.astype(np.uint8))
            pic_state = 1
        trans_height, trans_width, trans_ch = out_image.shape
        trans_image = np.zeros((trans_width, trans_height, trans_ch))
        for c in range(trans_ch):
            for row in range(trans_height):
                trans_image[:, trans_height - 1 - row, c] = out_image[row, :, c]
        out_image = trans_image
        cv2.imwrite("out.jpg", out_image.astype(np.uint8))


def main():
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    screen.fill((255, 255, 255))
    pygame.display.set_caption("处理图像")
    fps = 2
    fclock = pygame.time.Clock()

    global out_height
    global out_width
    global now_height
    global now_width

    text_box = TextBox()
    text_box.draw_text(screen)
    button = Button(screen, "start resize")
    button.draw_button()
    pic = pygame.image.load("in.jpg")
    out_height = pic.get_size()[1]
    out_width = pic.get_size()[0]
    now_height = pic.get_size()[1]
    now_width = pic.get_size()[0]

    screen.blit(pic, (5, 5))
    label_raw = Label(920, 20, screen, "raw size: " + str(pic.get_size()[1]) + "-" + str(pic.get_size()[0]))
    label_raw.draw_text()
    label_label_one = Label(920, 45, screen, "please input new size")
    label_label_thr = Label(950, 45, screen, "please wait")
    label_label_one.draw_text()
    label_label_two = Label(950, 70, screen, "width-height")
    label_label_two.draw_text()
    thread_state = 0  # 0准备 1运行 2死亡
    global pic_state
    t = threading.Thread(target=calc)
    t.setDaemon(True)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                text_box.key_down(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if button.rect.collidepoint(mouse_x, mouse_y) and thread_state == 0:
                    new_size = text_box.text.split('-')
                    if len(new_size) == 1:
                        break
                    out_height = int(new_size[0])
                    out_width = int(new_size[1])
                    thread_state = 1

                    t.start()
                if button.rect.collidepoint(mouse_x, mouse_y) and thread_state == 2:
                    thread_state = 0
                    pic_state = 0
                    t = threading.Thread(target=calc)

        if thread_state == 1:
            screen.fill((255, 255, 255))
            if not t.is_alive():
                thread_state = 2
            if pic_state == 0:
                pic = pygame.image.load("in.jpg")
                screen.blit(pic, (5, 5))
                label_label_thr.draw_text()
                label_label_for = Label(930, 110, screen, "output:" + str(out_height) + '-' + str(out_width))
                label_label_for.draw_text()
                label_label_fv = Label(950, 210, screen, "now:" + str(now_height) + '-' + str(now_width))
                label_label_fv.draw_text()
                label_raw.draw_text()
                pygame.display.update()
            else:
                try:
                    pic = pygame.image.load("tmp.jpg")
                except:
                    continue
                else:
                    screen.blit(pic, (5, 5))
                    fclock.tick(fps)
                    label_label_thr.draw_text()
                    label_label_for = Label(930, 110, screen, "output:" + str(out_height) + '-' + str(out_width))
                    label_label_for.draw_text()
                    label_label_fv = Label(950, 210, screen, "now:" + str(now_height) + '-' + str(now_width))
                    label_label_fv.draw_text()
                    label_raw.draw_text()
                    pygame.display.update()

        elif thread_state == 0:
            screen.fill((255, 255, 255))
            pic = pygame.image.load("in.jpg")
            screen.blit(pic, (5, 5))
            label_raw = Label(920, 20, screen,
                              "raw size: " + str(pic.get_size()[1]) + "-" + str(pic.get_size()[0]))
            label_raw.draw_text()
            label_label_one = Label(920, 45, screen, "please input new size")
            label_label_one.draw_text()
            label_label_two = Label(950, 70, screen, "width-height")
            label_label_two.draw_text()
            button.draw_button()
            text_box.draw_text(screen)
            label_raw.draw_text()
            pygame.display.update()

        elif thread_state == 2:
            screen.fill((255, 255, 255))
            button_two = Button(screen, "again!")
            button_two.draw_button()
            label_label_over = Label(930, 110, screen, "over:" + str(out_height) + '-' + str(out_width))
            label_label_over.draw_text()
            pic = pygame.image.load("out.jpg")
            screen.blit(pic, (5, 5))
            pygame.display.update()


if __name__ == '__main__':
    out_height = 0
    out_width = 0
    now_height = 0
    now_width = 0
    pic_state = 0
    main()
