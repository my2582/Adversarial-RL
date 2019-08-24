class Agent(object):
    def __init__(self, ID, initial_funds, stocks):
        self.ID = ID
        self.total_funds = initial_funds
        self.effective_funds = initial_funds
        self.portfolio = {k:[v,0,0] for k,v in stocks.items()}
        self.order_no = 1
        
    def get_portfolio(self):
        return {k:v[1:] for k,v in self.portfolio.items()}
    
    def get_value(self, stock):
        current_price = self.portfolio[stock][0].get_price()
        if current_price is not None:
            return self.portfolio[stock][0].get_price()*self.portfolio[stock][1]
        
    def make_add_order(self, stock, buy_sell='buy',qty=1,price=None):
        if price == None:
            price = self.portfolio[stock][0].get_price() #use the market price if price not provided
        
        if price != None:
            return {'order_id': 'T'+str(self.ID)+'_'+str(self.order_no), 'timestamp': time.clock(), 'type': 'add', 'quantity': qty, 'side': buy_sell, 'price': price}
        else:
            print("Error: Price is None")
            
    def place_order(self, stock, order):
        if order['side'] == 'buy':
            if self.effective_funds >= order['price'] * order['quantity']:
                self.effective_funds -= order['price'] * order['quantity']
                return self.portfolio[stock][0].process_order(order)
                self.order_no +=1
            else:
                print("Not enough effective funds to place order")
        elif order['side'] == 'sell':
            if self.portfolio[stock][2] >= order['quantity']:
                self.portfolio[stock][2] -= order['quantity']
                return self.portfolio[stock][0].process_order(order)
                self.order_no +=1
            else:
                print("Not enough effective qty to place order. Available effective qty = ", self.portfolio[stock][2])      