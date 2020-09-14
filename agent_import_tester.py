from pfrl.agent import AttributeSavingMixin

directory = "results\\10000_finish"

attribute_saver = AttributeSavingMixin()
agent = attribute_saver.load(directory)