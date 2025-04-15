import os
import os.path as path
import json
import random
import sys
import argparse

sys.path.append(os.getcwd()+'/EasyEdit')

try:
    from EasyEdit.easyeditor import (
        FTHyperParams,
        IKEHyperParams,
        KNHyperParams,
        MEMITHyperParams,
        ROMEHyperParams,
        LoRAHyperParams,
        MENDHyperParams,
        SERACHparams,
        WISEHyperParams,
        ZZZHyperParams
        )

    from EasyEdit.easyeditor import BaseEditor
    from EasyEdit.easyeditor.models.ike import encode_ike_facts
    from sentence_transformers import SentenceTransformer
    from EasyEdit.easyeditor import KnowEditDataset

except ImportError:
    from easyeditor import (
        FTHyperParams,
        IKEHyperParams,
        KNHyperParams,
        MEMITHyperParams,
        ROMEHyperParams,
        LoRAHyperParams,
        MENDHyperParams,
        SERACHparams,
        WISEHyperParams,
        ZZZHyperParams
        )

    from easyeditor import BaseEditor
    from easyeditor.models.ike import encode_ike_facts
    from sentence_transformers import SentenceTransformer
    from easyeditor import KnowEditDataset

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
     
