import queue
import threading

device_type='cpu'
class BaseGenerator(threading.Thread):
    def __init__(self, max_prefetch=1, device=device_type):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(max_prefetch)
        self.device = device
        self.stop_signal = False
        self.start()

    def run(self):
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self
    def len(self):
        return self.queue.qsize()


class BackgroundGenerator(BaseGenerator):
    def __init__(self, dataloader, max_prefetch=1, device=device_type):
        self.dataloader = dataloader
        super().__init__(max_prefetch, device)

    def run(self):
        for i, graph in enumerate(self.dataloader):
            if type(self.dataloader.loader_len)==int:
                stop_signal = i == self.dataloader.loader_len - 1
            else:
                stop_signal = i == self.dataloader.loader_len() - 1
            self.queue.put((graph.to(self.device, non_blocking=True), stop_signal))
        self.queue.put(None)
