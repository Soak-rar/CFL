import FACL_RES
import FACL_STC

class Process:
    def __init__(self, Config_name, Data_name):
        self.Config_name = Config_name
        self.Data_name = Data_name
    def __call__(self, *args, **kwargs):
        for arg in args:
            arg(self.Config_name, self.Data_name)

if __name__ == '__main__':
    p = Process("Config_1", "Data_1")
    p(FACL_RES.main)