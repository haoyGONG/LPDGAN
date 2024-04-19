import os
from data.LPBlur_dataset import create_dataset
from models.LPDGAN import create_model
from util.visualizer import save_images
from util import html


def print_test_losses(losses, results_dir, name):
    message = ''
    for k, v in losses.items():
        message += '%s: %.6f ' % (k, v)

    print(message)
    with open(os.path.join(results_dir, name, 'loss_log.txt'), "a") as log_file:
        log_file.write('%s\n' % message)

def test(opt):
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.mode, opt.epoch))
    if opt.load_iter > 0:
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.mode, opt.epoch))

    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()
        losses = model.get_current_losses()
        print_test_losses(losses, opt.results_dir, opt.name)
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()
