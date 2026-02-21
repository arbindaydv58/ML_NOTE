class Memory:

    def __init__(self,max_turns=6):
        self.max=max_turns
        self.turns=[]

    def add(self,user,bot):
        self.turns.append((user,bot))
        self.turns=self.turns[-self.max:]

    def format(self):
        text=""
        for u,b in self.turns:
            text+=f"User: {u}\nAssistant: {b}\n"
        return text