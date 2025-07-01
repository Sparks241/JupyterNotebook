import pygame
import random
import sys
import matplotlib.pyplot as plt
import io
import os
import datetime
import pymysql

# 获取当前文件的目录
current_path=os.path.abspath(__file__)

# 设定工作路径
current_directory=os.path.dirname(current_path)
log_path=current_directory+"\log\slot_machine_log.txt"
font_path_simhei=current_directory+"\\font\simhei.ttf"
history_path=current_directory+"\log\log_image.png"
# 用户提示
print("欢迎来到老虎机游戏，请根据提示修改你的日志路径")
print("当前的文件目录为",current_directory)
print("当前的本地日志路径",log_path)

pygame.init()
font = pygame.font.Font(font_path_simhei, 24)

# 屏幕设置
WIDTH, HEIGHT = 600, 400    # 600*400像素
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("🎰 老虎机")     # 设置游戏窗口的标题

# 定义用于中奖历史的列表
history = []
show_history = False  # 当前是否在查看记录页面

# 设定符号及概率
symbols = ["DD", "7", "BBB", "BB", "B", "C", "0"]
probabilities = [0.03, 0.03, 0.06, 0.10, 0.25, 0.01, 0.52]

# 初始金额
balance = 1000  # 初始总金额
bet = 10        # 初始下注金额

# 当前转轮符号
current_reels = ["0", "0", "0"]
result_text = ""

# 添加本地日志写入函数，需要修改日志文件的路径
def write_log(bet, multiplier, winnings, balance, reels):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"{now} | 下注: {bet} | 符号: {reels} | 倍率: {multiplier} | 奖金: {winnings} | 余额: {balance}\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_line)

# 连接云服务器的函数
def connect_db():
    conn = pymysql.connect(
        host='106.52.105.241',
        user='slotuser',
        password='slotuser',
        database='slot_logs_from_pygame',
        charset='utf8mb4'
    )
    return conn

# 插入日志函数
def write_log_to_db(bet, multiplier, winnings, balance, reels):
    conn = None
    try:
        conn = connect_db()
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        reels_str = ",".join(reels)
        sql = "INSERT INTO slot_logs(timestamp, bet, reels, multiplier, winnings, balance) VALUES (%s, %s, %s, %s, %s, %s)"
        cursor.execute(sql, (timestamp, bet, reels_str, multiplier, winnings, balance))
        conn.commit()
    except Exception as e:
        print("数据库写入失败:", e)
        print("SQL语句:", sql)
        print("参数:", (timestamp, bet, reels_str, multiplier, winnings, balance))
    finally:
        if conn:
            conn.close()
# 奖金倍率表
def get_multiplier(reel):
    r1, r2, r3 = reel
    if reel == ["DD", "DD", "DD"]:
        return 100
    elif reel == ["7", "7", "7"]:
        return 80
    elif reel == ["BBB", "BBB", "BBB"]:
        return 40
    elif reel == ["BB", "BB", "BB"]:
        return 25
    elif reel == ["B", "B", "B"]:
        return 10
    elif reel == ["C", "C", "C"]:
        return 10
    elif all(r in ["BBB", "BB", "B"] for r in reel) and len(set(reel)) > 1:
        return 5
    elif (r1 == "C" and r2 == "C") or (r1 == "C" and r3 == "C") or (r2 == "C" and r3 == "C"):
        return 5
    elif "C" in reel:
        return 2
    else:
        return 0

# 钻石加倍
def dd_multiplier(reel):
    return 2 ** reel.count("DD")

# 随机转轮
def spin_reel():
    return random.choices(symbols, weights=probabilities, k=3)

# 渲染文字
def draw_text(text, x, y, color=(255, 255, 255)):
    img = font.render(text, True, color)
    screen.blit(img, (x, y))

def draw_history_plot():
    if not history:
        draw_text("暂无中奖记录", 200, 180, (255, 255, 255))
        return

    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    ax.plot(history, marker='o', color='green')
    ax.set_title("中奖金额记录")
    ax.set_xlabel("次数")
    ax.set_ylabel("奖金")
    ax.grid(True)

    # 保存为图像对象
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='PNG')
    buf.seek(0)
    plt.close(fig)

    # 加载为 pygame 图像
    plot_img = pygame.image.load(buf)
    plot_img = pygame.transform.scale(plot_img, (580, 300))
    screen.blit(plot_img, (10, 50))
    pygame.image.save(plot_img,history_path)

# 主循环
clock = pygame.time.Clock()
running = True
while running:
    screen.fill((0, 0, 50))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_TAB:
                show_history = not show_history  # 切换页面
            if not show_history:
                if event.key == pygame.K_UP:
                    bet += 10
                elif event.key == pygame.K_DOWN:
                    bet = max(10, bet - 10)
                elif event.key == pygame.K_SPACE and balance >= bet:
                    current_reels = spin_reel()
                    base_multiplier = get_multiplier(current_reels)
                    final_multiplier = base_multiplier * dd_multiplier(current_reels)
                    winnings = bet * final_multiplier
                    balance = balance - bet + winnings
                    history.append(winnings)
                    if final_multiplier > 0:
                        result_text = f"中奖！倍率: {final_multiplier} 倍，奖金：{winnings}"
                    else:
                        result_text = "未中奖"
                    write_log(bet, final_multiplier, winnings, balance, current_reels)
                    write_log_to_db(bet, final_multiplier, winnings, balance, current_reels)

    # 显示内容（必须放在 while 循环内）
    if show_history:
        draw_text("中奖记录（按 Tab 返回游戏）", 20, 20)
        draw_history_plot()
    else:
        draw_text(f"余额：{balance}", 20, 20)
        draw_text(f"下注金额（↑↓调整）：{bet}", 20, 60)
        draw_text(f"按空格键开始旋转", 20, 100)

        for i, symbol in enumerate(current_reels):
            draw_text(symbol, 250 + i * 60, 180, (255, 255, 0))

        draw_text(result_text, 20, 320, (0, 255, 0))

    pygame.display.flip()
    clock.tick(30)


pygame.quit()
sys.exit()
