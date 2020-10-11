class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self,max_count=100):
        self.reset(max_count)
        
    def reset(self,max_count):
        self.val = 0
        self.avg = 0
        self.data_container = []
        self.max_count = max_count

    def update(self, val):
        self.val = val
        if(len(self.data_container) < self.max_count):
            self.data_container.append(val)
            self.avg = sum(self.data_container) * 1.0 / len(self.data_container)
        else:
            self.data_container.pop(0)
            self.data_container.append(val)
            self.avg = sum(self.data_container) * 1.0 / self.max_count

