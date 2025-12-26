import os
import sys
import pygame

# 设置窗口居中（必须在 pygame.display.init 之前设置）
os.environ['SDL_VIDEO_CENTERED'] = '1'

# Ensure top-level modules are importable when running from this subpackage
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import importlib  # lazy import game modules to avoid import-time side-effects (display creation)

# Simple button util
class MenuButton:
    def __init__(self, rect, text, font, color=(200,200,200), hover=(170,170,170)):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.color = color
        self.hover_color = hover
        self.pressed = False
    def draw(self, surf, mouse_pos):
        is_hover = self.rect.collidepoint(mouse_pos)
        col = self.hover_color if is_hover else self.color
        pygame.draw.rect(surf, col, self.rect)
        pygame.draw.rect(surf, (0,0,0), self.rect, 2)
        txt = self.font.render(self.text, True, (0,0,0))
        surf.blit(txt, txt.get_rect(center=self.rect.center))
    def clicked(self, mouse_pos, mouse_down):
        return self.rect.collidepoint(mouse_pos) and mouse_down


def run_menu():
    # 初始化仅 display 与 font，避免在启动菜单时初始化 mixer 导致黑屏或延迟
    pygame.display.init()
    pygame.font.init()
    SCREEN_W, SCREEN_H = 800, 600
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("CALC WARS - Menu")
    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont('Arial', 56, bold=True)
    btn_font = pygame.font.SysFont('Arial', 24)

    title_surf = title_font.render('CALC WARS', True, (30,30,120))

    # Buttons
    btn_classic = MenuButton((SCREEN_W//2 - 150, 220, 300, 60), 'Classic (2P)', btn_font)
    btn_ai = MenuButton((SCREEN_W//2 - 150, 300, 300, 60), 'AI Battle', btn_font)
    btn_quit = MenuButton((SCREEN_W//2 - 150, 460, 300, 50), 'Quit', btn_font, color=(240,100,100), hover=(200,60,60))

    # Difficulty selection overlay
    show_difficulty = False
    diff_buttons = []
    # Map difficulties to model files (adjust as needed)
    model_map = {
        'Easy': 'ppo_1001_1.pth',
        'Medium': 'ppo_144005_1.pth',
        'Hard': 'ppo_369500_1.pth',
    }
    for i, name in enumerate(['Easy', 'Medium', 'Hard']):
        diff_buttons.append(MenuButton((SCREEN_W//2 - 130, 220 + i*80, 260, 56), name, btn_font))
    btn_back = MenuButton((SCREEN_W//2 - 130, 220 + 3*80, 260, 48), 'Back', btn_font, color=(220,220,220), hover=(200,200,200))

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        mouse_down = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_down = True

        screen.fill((220,220,240))
        # Title
        screen.blit(title_surf, title_surf.get_rect(center=(SCREEN_W//2, 120)))

        if not show_difficulty:
            btn_classic.draw(screen, mouse_pos)
            btn_ai.draw(screen, mouse_pos)
            btn_quit.draw(screen, mouse_pos)
            if btn_classic.clicked(mouse_pos, mouse_down):
                # Launch classic game (lazy import to avoid import-time display creation)
                # show quick loading feedback
                loading = btn_font.render('Launching Classic...', True, (10,10,10))
                screen.fill((220,220,240))
                screen.blit(loading, loading.get_rect(center=(SCREEN_W//2, SCREEN_H//2)))
                pygame.display.flip()
                try:
                    mod = importlib.import_module('CALC_WARS_classic')
                    ClassicGame = getattr(mod, 'Game')
                    game = ClassicGame()
                    game.run()
                except Exception as e:
                    print(f"[Menu] Error running Classic game: {e}")
                finally:
                    # Stop any playing audio and fully re-init display + font to return to menu
                    try:
                        pygame.mixer.stop()
                        pygame.mixer.music.stop()
                    except Exception:
                        pass
                    try:
                        # 不完全退出 display，只为了重新设定模式
                        pygame.display.quit()
                    except Exception:
                        pass
                    pygame.display.init()
                    pygame.font.init()
                    # small pause to let subsystems settle
                    pygame.time.delay(100)
                    # 重新设置居中和窗口大小
                    os.environ['SDL_VIDEO_CENTERED'] = '1'
                    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
                    pygame.display.set_caption("CALC WARS - Menu")
                    pygame.event.clear()
                    pygame.display.flip()
            if btn_ai.clicked(mouse_pos, mouse_down):
                show_difficulty = True
            if btn_quit.clicked(mouse_pos, mouse_down):
                running = False
        else:
            # Difficulty selection view
            header = btn_font.render('Select AI Difficulty', True, (10,10,10))
            screen.blit(header, header.get_rect(center=(SCREEN_W//2, 180)))
            for b in diff_buttons:
                b.draw(screen, mouse_pos)
            btn_back.draw(screen, mouse_pos)
            for b in diff_buttons:
                if b.clicked(mouse_pos, mouse_down):
                    # Launch AI with model
                    chosen = b.text
                    model_file = model_map.get(chosen)

                    # show a quick loading message before importing heavy modules
                    loading = btn_font.render('Loading AI, please wait...', True, (10,10,10))
                    screen.fill((220,220,240))
                    screen.blit(loading, loading.get_rect(center=(SCREEN_W//2, SCREEN_H//2)))
                    pygame.display.flip()

                    try:
                        mod = importlib.import_module('CALC_WARS_AI')
                        AIGame = getattr(mod, 'Game')
                        # Resolve model full path relative to the CALC_WARS project root
                        if not os.path.isabs(model_file):
                            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
                            model_path = os.path.join(project_root, model_file)
                        else:
                            model_path = model_file
                        ai_game = AIGame(model_path=model_path)
                        ai_game.run()
                    except Exception as e:
                        print(f"[Menu] Error running AI game: {e}")
                    finally:
                        # Stop audio and fully re-init display + font to return to menu
                        try:
                            pygame.mixer.stop()
                            pygame.mixer.music.stop()
                        except Exception:
                            pass
                        try:
                            pygame.display.quit()
                        except Exception:
                            pass
                        pygame.display.init()
                        pygame.font.init()
                        pygame.time.delay(100)
                        # 重新设置居中和窗口大小
                        os.environ['SDL_VIDEO_CENTERED'] = '1'
                        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
                        pygame.display.set_caption("CALC WARS - Menu")
                        pygame.event.clear()
                        pygame.display.flip()
                    show_difficulty = False
            if btn_back.clicked(mouse_pos, mouse_down):
                show_difficulty = False

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == '__main__':
    run_menu()