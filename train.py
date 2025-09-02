from collections import OrderedDict
from tqdm import tqdm
from dataset.cad_dataset import get_dataloader
from config import ConfigAE
from utils import cycle
from trainer import TrainerAE


def main():
    # create experiment cfg containing all hyperparameters
    cfg = ConfigAE(is_train=True)

    # create network and training agent
    tr_agent = TrainerAE(cfg)

    # load from checkpoint if provided
    if eval(cfg.is_load_weight):
        tr_agent.load_ckpt(cfg.ckpt)

    # create dataloader
    train_loader = get_dataloader('train', cfg)
    val_loader = get_dataloader('validation', cfg)
    val_loader_all = get_dataloader('validation', cfg)
    val_loader = cycle(val_loader)

    # start training
    clock = tr_agent.clock

    for e in range(clock.epoch, cfg.nr_epochs):
        # begin iteration
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # train step
            outputs, losses = tr_agent.train_func(data)

            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

            # validation step
            if clock.step % cfg.val_frequency == 0:
                data = next(val_loader)
                outputs, losses = tr_agent.val_func(data)

            clock.tick()

            tr_agent.update_learning_rate()

        if clock.epoch % 5 == 0:
            tr_agent.evaluate(val_loader_all)

        clock.tock()

        if clock.epoch % cfg.save_frequency == 0:
            tr_agent.save_ckpt()

        # if clock.epoch % 10 == 0:
        tr_agent.save_ckpt('latest')


def test_occ_qt():
    import sys
    from OCC.Display.SimpleGui import init_display

    print("==== PyQt5 + pyocc-core OpenGL 测试 ====")
    try:
        display, start_display, add_menu, add_function_to_menu = init_display("qt-pyqt5")
        print("[OK] 成功创建 Qt-PyQt5 Display3d 窗口")
        print("    如果弹出窗口正常显示，说明 OpenGL 初始化成功。")
        print("    请关闭窗口继续。")
        start_display()
    except Exception as e:
        print("[ERROR] 无法创建 OpenGL 窗口")
        print("错误信息：", e)
        sys.exit(1)


if __name__ == '__main__':
    main()
    # test_occ_qt()




