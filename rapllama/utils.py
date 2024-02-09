import os
from pathlib import Path
import torch
import pandas as pd

from ludwig.api import LudwigModel



def get_model_llama():
    model_dir = Path.cwd() / "assets" / "rapllama"
    #model_dir="./../assets/rapllama"
    print(model_dir)
    ludwig_model = LudwigModel.load(model_dir)
    print(ludwig_model)
    return ludwig_model


def prediction_with_llama(model,instruction="",elements_keys=""):
    data=pd.DataFrame([
        {
            "instruction": instruction,
            'input':elements_keys
        }
    ])
    print(data)
    predictions, l = model.predict(dataset=data)
    print(l)
    return predictions['output_response'][0][0]


