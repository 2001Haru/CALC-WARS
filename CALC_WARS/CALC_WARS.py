import os
import sys
import pygame
import importlib 

# 设置窗口居中
os.environ['SDL_VIDEO_CENTERED'] = '1'

'''CALC WARS 主菜单实现文件'''
# 请注意，为了保证加载文件不黑屏影响游戏，我们使用lazy import的方式导入游戏模块

# 代码健壮性保证导入文件可用路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class MenuButton:
    '''菜单按钮类'''
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
    
def font_setting(size, bold=False):
    # 保证mac系统兼容性
    font_names = ['Times New Roman', 'times new roman', "times", "serif", 'Arial']
    return pygame.font.SysFont(font_names, size, bold=bold)

def run_menu():
    # 初始化仅 display 与 font，避免在启动菜单时初始化 mixer 导致黑屏或延迟
    pygame.display.init()
    pygame.font.init()
    SCREEN_W, SCREEN_H = 800, 600
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("CALC WARS - Menu")
    clock = pygame.time.Clock()

    title_font = font_setting(70, bold=True)
    btn_font = font_setting(30)

    title_surf = title_font.render('CALC WARS', True, (30,30,120))

    # 主菜单按钮
    btn_classic = MenuButton((SCREEN_W//2 - 150, 220, 300, 60), 'Classic (2P)', btn_font)
    btn_ai = MenuButton((SCREEN_W//2 - 150, 300, 300, 60), 'AI Battle', btn_font)
    btn_quit = MenuButton((SCREEN_W//2 - 150, 460, 300, 50), 'Quit', btn_font, color=(240,100,100), hover=(200,60,60))

    # 难度选择菜单
    show_difficulty = False
    diff_buttons = []
    # AI 难度对应模型文件
    model_map = {
        'Rookie': 'ppo_67003_1.pth',
        'Veteran': 'ppo_144005_1.pth',
        'Marshal': 'ppo_463534_1.pth',
    }
    for i, name in enumerate(['Rookie', 'Veteran', 'Marshal']):
        diff_buttons.append(MenuButton((SCREEN_W//2 - 130, 220 + i*80, 260, 56), name, btn_font))
    btn_back = MenuButton((SCREEN_W//2 - 130, 220 + 3*80, 260, 48), 'Back', btn_font, color=(220,220,220), hover=(200,200,200))

    # 模式选择菜单
    show_mode_select = False
    mode_for = None  # 'classic' or 'ai'
    mode_buttons = [
        MenuButton((SCREEN_W//2 - 130, 220, 260, 56), 'Single', btn_font),
        MenuButton((SCREEN_W//2 - 130, 300, 260, 56), 'BO5', btn_font)
    ]
    btn_mode_back = MenuButton((SCREEN_W//2 - 130, 380, 260, 48), 'Back', btn_font, color=(220,220,220), hover=(200,200,200))
    selected_model_file = None

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

        # 主菜单 / 难度 / 模式 选择流程
        if show_mode_select:
            header = btn_font.render('Select Mode', True, (10,10,10))
            screen.blit(header, header.get_rect(center=(SCREEN_W//2, 180)))
            for b in mode_buttons:
                b.draw(screen, mouse_pos)
            btn_mode_back.draw(screen, mouse_pos)

            for b in mode_buttons:
                if b.clicked(mouse_pos, mouse_down):
                    chosen_mode = b.text
                    # 根据 mode_for 启动对应游戏模式
                    if mode_for == 'classic':
                        # 启动 Classic 模式
                        loading = btn_font.render(f'Launching Classic ({chosen_mode})...', True, (10,10,10))
                        screen.fill((220,220,240))
                        screen.blit(loading, loading.get_rect(center=(SCREEN_W//2, SCREEN_H//2)))
                        pygame.display.flip()
                        try:
                            mod = importlib.import_module('CALC_WARS_classic')
                            ClassicGame = getattr(mod, 'Game')
                            game = ClassicGame(match_mode='bo5' if chosen_mode == 'BO5' else 'single')
                            game.run()
                        except Exception as e:
                            print(f"[Menu] Error running Classic game: {e}")
                        finally:
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
                            os.environ['SDL_VIDEO_CENTERED'] = '1'
                            screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
                            pygame.display.set_caption("CALC WARS - Menu")
                            pygame.event.clear()
                            pygame.display.flip()
                        show_mode_select = False
                        mode_for = None

                    elif mode_for == 'ai':
                        if not selected_model_file:
                            # 模型未选择，回到难度选择
                            show_mode_select = False
                            show_difficulty = True
                            mode_for = None
                            break

                        loading = btn_font.render(f'Launching AI ({chosen_mode})...', True, (10,10,10))
                        screen.fill((220,220,240))
                        screen.blit(loading, loading.get_rect(center=(SCREEN_W//2, SCREEN_H//2)))
                        pygame.display.flip()
                        try:
                            mod = importlib.import_module('CALC_WARS_AI')
                            AIGame = getattr(mod, 'Game')
                            ai_game = AIGame(model_path=selected_model_file, match_mode='bo5' if chosen_mode == 'BO5' else 'single')
                            ai_game.run()
                        except Exception as e:
                            print(f"[Menu] Error running AI game: {e}")
                        finally:
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
                            os.environ['SDL_VIDEO_CENTERED'] = '1'
                            screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
                            pygame.display.set_caption("CALC WARS - Menu")
                            pygame.event.clear()
                            pygame.display.flip()

                        show_mode_select = False
                        mode_for = None
                        selected_model_file = None

            if btn_mode_back.clicked(mouse_pos, mouse_down):
                # 返回到对应上一步
                if mode_for == 'ai':
                    show_mode_select = False
                    show_difficulty = True
                else:
                    show_mode_select = False
                    mode_for = None

        elif show_difficulty:
            # 难度选择视图
            header = btn_font.render('Select AI Difficulty', True, (10,10,10))
            screen.blit(header, header.get_rect(center=(SCREEN_W//2, 180)))
            for b in diff_buttons:
                b.draw(screen, mouse_pos)
            btn_back.draw(screen, mouse_pos)
            for b in diff_buttons:
                if b.clicked(mouse_pos, mouse_down):
                    # 选择了 AI 难度，接着进入单局/BO5 选择
                    chosen = b.text
                    model_file = model_map.get(chosen)
                    # 解析为绝对路径以便随后传递给 Game
                    if not os.path.isabs(model_file):
                        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
                        model_path = os.path.join(project_root,'Model_checkpoints', model_file)
                    else:
                        model_path = model_file

                    selected_model_file = model_path
                    show_mode_select = True
                    mode_for = 'ai'
                    show_difficulty = False
            if btn_back.clicked(mouse_pos, mouse_down):
                show_difficulty = False

        else:
            # 主菜单视图
            btn_classic.draw(screen, mouse_pos)
            btn_ai.draw(screen, mouse_pos)
            btn_quit.draw(screen, mouse_pos)
            if btn_classic.clicked(mouse_pos, mouse_down):
                # 切换到模式选择（单局/BO5）
                show_mode_select = True
                mode_for = 'classic'
            if btn_ai.clicked(mouse_pos, mouse_down):
                show_difficulty = True
            if btn_quit.clicked(mouse_pos, mouse_down):
                running = False

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == '__main__':
    run_menu()