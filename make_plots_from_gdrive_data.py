import pickle
import os
import logging
import sys

import matplotlib.pyplot as plt

import pythonutils.path
import pythonutils.gdrive as gdrive
import pythonutils.plotting as plotting

# logging_level = 25
logging_level = logging.INFO
logging_format = "%(asctime)s %(process)s %(thread)s: %(message)s"
logging.basicConfig(level=logging_level,
                    format=logging_format,
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    username = 'robotmatoba'
    do_pgf_plotting = True

    gdrive.authenticate_automatically(username)

    cifar10 = gdrive.find_items(name='cifar10', parent_fid=None, skip_trashed=True)
    assert 1 == len(cifar10)
    parent_fid = cifar10[0][1]

    filename = 'metrics_multiindex.pkl'
    found_items = gdrive.find_items(name=filename,
                             parent_fid=parent_fid,
                             skip_trashed=True)
    remote_fid = found_items[0][1]

    paths = pythonutils.path.get_paths()
    local_filedir = paths['work']
    local_path = os.path.join(local_filedir, filename)
    gdrive.download_file_to_folder(remote_fid, local_path)
    print("Downloaded to {}".format(local_path))

    with open(local_path, 'rb') as fo:
        metrics_multiindex = pickle.load(fo)

    if do_pgf_plotting:
        fig_format = 'pgf'
        plotting.initialise_pgf_plots()
    else:
        fig_format = plotting.get_default_extension()

    plot_metrics = sorted(set(metrics_multiindex.columns.get_level_values(1)))
    scale = .5
    attachments = []
    for idx, metric_name in enumerate(plot_metrics):
        fig = plt.figure(figsize=(12 * scale, 4 * scale))
        to_boxplot = metrics_multiindex.xs(metric_name, axis=1, level=1)
        new_columns = [int(x[1:]) for x in to_boxplot.columns]
        # to_boxplot.rename(mapper=lambda x: str(x[1:]), index='columns', inplace=True)
        # to_boxplot.rename(mapper=lambda x: int(x[1:]), index='columns')
        # to_boxplot.rename(new_columns, axis='columns')
        to_boxplot.columns = new_columns

        to_boxplot.boxplot()
        # plt.title(metric_name)
        ident = "boxplot_{}".format(metric_name)

        fig_path = plotting.smart_save_fig(fig, ident=ident, fig_format=fig_format)
        logger.info("Figure saved to {}".format(fig_path))
        plt.close(fig)
        attachments.append(fig_path)

