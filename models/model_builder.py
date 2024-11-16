from . import (dualcoop)
from . import (positivecoop)
from . import (negativecoop)
from . import (baseline)

def build_model(cfg, args, classnames):
    """
    Args:
        args: all options defined in opts.py and num_classes
        test_mode:
    Returns:
        network model
        architecture name
    """
    functions = {
        "dualcoop": dualcoop,
        "positivecoop": positivecoop,
        "negativecoop": negativecoop,
        "baseline": baseline
        
        }

    
    method_name = args.method_name

    
    model = functions[method_name](cfg, classnames)
    print("args.method_name", args.method_name)
    
    network_name = model.network_name if hasattr(model, 'network_name') else cfg.MODEL.BACKBONE.NAME
    arch_name = "{dataset}-{arch_name}".format(
        dataset=cfg.DATASET.NAME, arch_name=network_name)
    # add setting info only in training



    arch_name += "{}".format('-' + args.prefix if args.prefix else "")
    if not args.evaluate:
        arch_name += "-{}-{}-bs{}-e{}-p{}-lr{}".format(args.method_name,cfg.OPTIM.LR_SCHEDULER, cfg.DATALOADER.TRAIN_X.BATCH_SIZE,  cfg.OPTIM.MAX_EPOCH, args.partial_portion, args.lr)
    return model, arch_name
