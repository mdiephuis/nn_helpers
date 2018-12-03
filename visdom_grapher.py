from visdom import Visdom
import numpy as np
import time


class VisdomGrapher:

    def __init__(self, env_name, server, port=8097):
        self.env_name = env_name
        self.vis = Visdom(server=server, port=port, env=env_name)
        startup_sec = 1
        while not self.vis.check_connection() and startup_sec > 0:
            time.sleep(0.1)
            startup_sec -= 0.1

        # optional, time out connection

    def add_scalar(self, plot_name, idtag, y, x, opts={}):
        '''
        Update vidomplot by win_title with a scalar value.
        If it doesn't exist, create a new plot
        - win_title: name and title of plot
        - y: y coord
        - x: x coord
        - options_dict, example {'legend': 'NAME', 'ytickmin': 0, 'ytickmax': 1}
        '''

        # todo:numpy check for y and x

        # check if graph exists
        exists = self.vis.win_exists(idtag)

        # update existing window
        if exists:
            self.vis.line(Y=np.array([y]), X=np.array([x]), win=idtag,
                          update='append', opts=opts)
        else:
            self.vis.line(Y=np.array([y]), X=np.array([x]), win=idtag,
                          opts={'title': plot_name,
                                'xlabel': 'epoch'})

    def add_histogram(self, x, plot_name, idtag, opts={'numbins': 25}):
        if len(list(x.shape)) > 1:
            x = x.ravel()
        opts = {**opts, **{'title': plot_name}}
        self.vis.histogram(x, win=idtag, opts=opts)

    def add_image(self, plot_name, idtag, image):
        '''
        Update visdomplot by win_title with a scalar value.
        If it doesn't exist, create a new plot by default
        '''
        self.vis.images(image, win=idtag, opts=dict(title=plot_name))
