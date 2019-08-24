class Exchange(object):
    
    def __init__(self,num_agents,initial_money,IPO):
        self.stocks = {"S"+str(i):Orderbook() for i in range(1,len(IPO)+1)}
        self.agents = {"T"+str(i):Agent(i,initial_money, self.stocks) for i in range(1,num_agents+1)}
        
        self.agents["T-1"] = Agent(-1,0, self.stocks)
        
        #initializing the IPO agent's portfolio with stocks
        for stock in self.stocks.keys():
            self.agents["T-1"].portfolio[stock][1] = IPO[stock][1]
            self.agents["T-1"].portfolio[stock][2] = IPO[stock][1]
            self.place_add_order("T-1", stock, buy_sell='sell',qty=IPO[stock][1],price=IPO[stock][0])
            
        #print(self.get_agents_status())
        
    def get_agents_status(self):
        d = {}
        for agent,data in self.agents.items():
            d[agent] = (data.total_funds,data.effective_funds,data.get_portfolio())   
            
        return d
    
    def get_total_funds(self,agent):
        return self.agents[agent].total_funds
    
    def get_effective_funds(self,agent):
        return self.agents[agent].effective_funds
    
    def get_portfolio(self,agent):
        return self.agents[agent].get_portfolio()
    
    def place_add_order(self,agent, stock, buy_sell='buy',qty=1,price=None):
        o = self.agents[agent].make_add_order(stock, buy_sell,qty,price)
        trades = self.agents[agent].place_order(stock, o)
        #print(trades)
        
        if trades != None:
            for trade in trades:
                io = self.stocks[stock].order_history[trade['incoming_order_id']]
                ro = self.stocks[stock].order_history[trade['resting_order_id']]
                io_t = io['order_id'].split('_')[0]
                ro_t = ro['order_id'].split('_')[0]
                
                self.do_bookkeeping(io_t, stock, trade, io)
                self.do_bookkeeping(ro_t, stock, trade, ro)
                
    def place_delta_add_order(self,agent,new_portfolio):
        current_portfolio = e.get_portfolio(agent)
        assert(len(new_portfolio)==len(current_portfolio)) 
        
        for stock, qty in current_portfolio.items():
            diff = new_portfolio[stock] - qty[1] 
            if diff > 0: #need to place buy orders
                self.place_add_order(agent, stock, buy_sell='buy',qty=diff)
            elif diff <0: #need to place sell orders
                self.place_add_order(agent, stock, buy_sell='sell',qty=-diff)
                    
    def do_bookkeeping(self, agent, stock, trade,orignal_order):
        if orignal_order['side'] == 'buy' and orignal_order['type'] == 'add':
            self.agents[agent].effective_funds += trade['quantity'] * orignal_order['price']
            self.agents[agent].effective_funds -= trade['quantity'] * trade['price']
            self.agents[agent].total_funds -= trade['quantity'] * trade['price']
            self.agents[agent].portfolio[stock][1] += trade['quantity']
            self.agents[agent].portfolio[stock][2] += trade['quantity']
        elif orignal_order['side'] == 'sell' and orignal_order['type'] == 'add':
            self.agents[agent].effective_funds += trade['quantity'] * trade['price']
            self.agents[agent].total_funds += trade['quantity'] * trade['price']
            self.agents[agent].portfolio[stock][1] -= trade['quantity']
            
    def get_order_book(self,stock):
        return (self.stocks[stock]._bid_book,self.stocks[stock]._ask_book) 