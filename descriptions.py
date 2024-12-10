import math
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Optional

import pandas as pd
import tiktoken
import numpy as np

class Description(ABC):
    gpt_examples_base = "data/gpt_examples"
    describe_base = "data/describe"

    @property
    @abstractmethod
    def gpt_examples_path(self) -> str:
        """
        Path to excel files containing examples of user and assistant messages for the GPT to learn from.
        """
        return r"C:\Users\degef.antunes\Desktop\JoseAmerico\Soccermatics ProCourse\Project 3 v_2\Forward_general_user_assistant.xlsx"

    @property
    @abstractmethod
    def describe_paths(self) -> Union[str, List[str]]:
        """
        Returns the path(s) to Excel files containing questions and answers for the GPT to learn from.
        """
        return r"C:\Users\degef.antunes\Desktop\JoseAmerico\Soccermatics ProCourse\Project 3 v_2\Forward_players_user_assistant.xlsx"
