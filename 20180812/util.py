def get_num_gen(gen):
    return sum(1 for x in gen)

def is_leaf(model):
    return get_num_gen(model.children()) == 0


def init_learning(model):
    for child in model.children():
        if hasattr(child, 'phase'):
            # turn_off_learning(child)
			child.weight.requires_grad = False
        elif is_leaf(child):
            if hasattr(child, 'weight'):
                child.weight.requires_grad = True
        else:
            init_learning(child)


def turn_off_learning(model):
    if is_leaf(model):
        if hasattr(model, 'weight'):
            model.weight.requires_grad = False
        return

    for child in model.children():
        if is_leaf(child):
            if hasattr(child, 'weight'):
                child.weight.requires_grad = False
        else:
            turn_off_learning(child)


def switching_learning(model):
    if is_leaf(model):
        if hasattr(model, 'weight'):
            if model.weight.requires_grad:
                model.weight.requires_grad = False
            else:
                model.weight.requires_grad = True
        return

    for child in model.children():
        if is_leaf(child):
            if hasattr(child, 'weight'):
                if child.weight.requires_grad:
                    child.weight.requires_grad = False
                else:
                    child.weight.requires_grad = True
        else:
            switching_learning(child)