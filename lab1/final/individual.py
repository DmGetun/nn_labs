class Individual(list):
    
    def __init__(self, *args):
        super().__init__(*args)
        
    @property
    def estimate(self):
        return self._estimate
    
    @estimate.setter
    def estimate(self, estimate):
        self._estimate = estimate
    
    @property
    def percent(self):
        return self._percent
    
    @percent.setter
    def percent(self, percent):
        self._percent = percent
        
    @property
    def order(self):
        return self._order
    
    @order.setter
    def order(self, order):
        self._order = order
        
    @property
    def rank(self):
        return self._rank
    
    @rank.setter
    def rank(self,rank):
        self._rank = rank
        
    @property
    def percent_rank(self):
        return self._percent_rank
    
    @percent_rank.setter
    def percent_rank(self, percent_rank):
        self._percent_rank = percent_rank
        
    def direct_mutation(self, start, stop):
            min = self.index(start)
            self[0] , self[min] = self[min] , self[0]
                        
            max = self.index(stop)
            self[-1] , self[max] = self[max] , self[-1]