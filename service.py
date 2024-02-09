from __future__ import annotations
import bentoml
import pandas as pd

from rapllama.utils import get_model_llama

EXAMPLE_INPUT = "Breaking News: In an astonishing turn of events, the small \
town of Willow Creek has been taken by storm as local resident Jerry Thompson's cat, \
Whiskers, performed what witnesses are calling a 'miraculous and gravity-defying leap.' \
Eyewitnesses report that Whiskers, an otherwise unremarkable tabby cat, jumped \
a record-breaking 20 feet into the air to catch a fly. The event, which took \
place in Thompson's backyard, is now being investigated by scientists for potential \
breaches in the laws of physics. Local authorities are considering a town festival \
to celebrate what is being hailed as 'The Leap of the Century."


@bentoml.service(
    traffic={
        "timeout": 300,
    },
    resources={
        "gpu": 1,

    },
)
class Rapgenllama:
    def __init__(self) -> None:
        self.model =get_model_llama()

    @bentoml.api
    def generate_rap(self, instruction:str="Transformer en parole de rap ",text: str = EXAMPLE_INPUT) -> str:
        data = pd.DataFrame([
            {
                "instruction": instruction,
                'input': text
            }
        ])

        predictions, l = self.model.predict(dataset=data)

        return predictions['output_response'][0][0]