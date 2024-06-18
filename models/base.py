from abc import abstractmethod
class LearningDiffusionModel:
    def __init__(self) -> None:
        pass

    def load_dataset(self, dataset_name: str):
        pass
    
    @abstractmethod
    def create_state(self, rng):
        pass

    @abstractmethod
    def train_step(self, sample):
        pass

    def get_potential(self, state):
        return lambda x: 0.
    
    def get_beta(self, state):
        return 0.
    
    def get_interaction(self, state):
        return lambda x: 0.