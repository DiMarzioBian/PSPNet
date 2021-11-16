def set_optimizer_lr(optimizer, lr):
    """Sets the learning rate to a specific one"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def update_optimizer_lr(optimizer):
    """Update the learning rate"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5
