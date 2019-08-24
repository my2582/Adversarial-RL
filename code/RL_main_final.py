#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 15:52:23 2018

@author: msyeom
"""

import time
import pprint
import random
import bisect
import pandas as pd
import math
import copy

class Orderbook(object):
    '''
    Orderbook tracks, processes and matches orders.
    
    Orderbook is a set of linked lists and dictionaries containing trades, bids and asks.
    One dictionary contains a history of all orders;
    two other dictionaries contain priced bid and ask orders with linked lists for access;
    one dictionary contains trades matched with orders on the book.
    Orderbook also provides methods for storing and retrieving orders and maintaining a 
    history of the book.
    Public attributes: order_history, confirm_modify_collector, confirm_trade_collector,
    trade_book and traded.
    Public methods: add_order_to_book(), process_order(), order_history_to_h5(), trade_book_to_h5(),
    sip_to_h5() and report_top_of_book()
    '''
    
    def __init__(self, initial_stock_price):
        '''
        Initialize the Orderbook with a set of empty lists and dicts and other defaults
        
        order_history is a list of all incoming orders (dicts) in the order received
        _bid_book_prices and _ask_book_prices are linked (sorted) lists of bid and ask prices
        which serve as pointers to:
        _bid_book and _ask_book: dicts of current order book state and OrderedDicts of orders
        the OrderedDicts maintain time priority for each order at a given price.
        confirm_modify_collector and confirm_trade_collector are lists that carry information (dicts) from the
        order processor and/or matching engine to the traders
        trade_book is a list if trades in sequence
        _order_index identifies the sequence of orders in event time
        '''
        self.order_history = dict()
        self._bid_book = {}
        self._bid_book_prices = []
        self._ask_book = {}
        self._ask_book_prices = []
        self.confirm_modify_collector = []
        self.confirm_trade_collector = []
        self._sip_collector = []
        self.trade_book = []
        self._order_index = 0
        self.traded = False
        self._last_settled_price = initial_stock_price
        self._stock_price_history = [initial_stock_price]

    def _add_order_to_history(self, order):
        '''Add an order (dict) to order_history'''
        hist_order = {'order_id': order['order_id'], 'timestamp': order['timestamp'], 'type': order['type'], 
                      'quantity': order['quantity'], 'side': order['side'], 'price': order['price'], 'clock':order['clock']}
        self._order_index += 1
        hist_order['exid'] = self._order_index
        self.order_history[order['order_id']] = hist_order
    
    def add_order_to_book(self, order):
        '''
        Use insort to maintain on ordered list of prices which serve as pointers
        to the orders.
        '''
        book_order = {'order_id': order['order_id'], 'timestamp': order['timestamp'], 'type': order['type'], 
                      'quantity': order['quantity'], 'side': order['side'], 'price': order['price'], 'clock': order['clock']}
        if order['side'] == 'buy':
            book_prices = self._bid_book_prices
            book = self._bid_book
        else:
            book_prices = self._ask_book_prices
            book = self._ask_book 
        if order['price'] in book_prices:
            book[order['price']]['num_orders'] += 1
            book[order['price']]['size'] += order['quantity']
            book[order['price']]['order_ids'].append(order['order_id'])
            book[order['price']]['orders'][order['order_id']] = book_order
        else:
            bisect.insort(book_prices, order['price'])
            book[order['price']] = {'num_orders': 1, 'size': order['quantity'], 'order_ids': [order['order_id']],
                                    'orders': {order['order_id']: book_order}}
            
    def _remove_order(self, order_side, order_price, order_id):
        '''Pop the order_id; if  order_id exists, updates the book.'''
        if order_side == 'buy':
            book_prices = self._bid_book_prices
            book = self._bid_book
        else:
            book_prices = self._ask_book_prices
            book = self._ask_book
        is_order = book[order_price]['orders'].pop(order_id, None)
        if is_order:
            book[order_price]['num_orders'] -= 1
            book[order_price]['size'] -= is_order['quantity']
            book[order_price]['order_ids'].remove(is_order['order_id'])
            if book[order_price]['num_orders'] == 0:
                book_prices.remove(order_price)
                    
    def _modify_order(self, order_side, order_quantity, order_id, order_price):
        '''Modify order quantity; if quantity is 0, removes the order.'''
        book = self._bid_book if order_side == 'buy' else self._ask_book        
        if order_quantity < book[order_price]['orders'][order_id]['quantity']:
            book[order_price]['size'] -= order_quantity
            book[order_price]['orders'][order_id]['quantity'] -= order_quantity
        else:
            self._remove_order(order_side, order_price, order_id)
            
    def _add_trade_to_book(self, resting_order_id, resting_timestamp, incoming_order_id, timestamp, price, quantity, side):
        '''Add trades (dicts) to the trade_book list.'''
        trade = {'resting_order_id': resting_order_id, 'resting_timestamp': resting_timestamp, 
                                'incoming_order_id': incoming_order_id, 'timestamp': timestamp, 'price': price,
                                'quantity': quantity, 'side': side}
        self.trade_book.append(trade)
        return trade

    def _confirm_trade(self, timestamp, order_side, order_quantity, order_id, order_price):
        '''Add trade confirmation to confirm_trade_collector list.'''
        trader = order_id.partition('_')[0]
        self.confirm_trade_collector.append({'timestamp': timestamp, 'trader': trader, 'order_id': order_id, 
                                             'quantity': order_quantity, 'side': order_side, 'price': order_price})
    
    def _confirm_modify(self, timestamp, order_side, order_quantity, order_id):
        '''Add modify confirmation to confirm_modify_collector list.'''
        trader = order_id.partition('_')[0]
        self.confirm_modify_collector.append({'timestamp': timestamp, 'trader': trader, 'order_id': order_id, 
                                              'quantity': order_quantity, 'side': order_side})
                  
    def process_order(self, order):
        '''Check for a trade (match); if so call _match_trade, otherwise modify book(s).'''
        self.confirm_modify_collector.clear()
        self.traded = False
        self._add_order_to_history(order)
        if order['type'] == 'add':
            if order['side'] == 'buy':
                if len(self._ask_book_prices)>0:
                    if order['price'] >= self._ask_book_prices[0]:
                        return self._match_trade(order)
                    else:
                        self.add_order_to_book(order)     
                else:
                    self.add_order_to_book(order)
            else: #order['side'] == 'sell'
                if len(self._bid_book_prices)>0:              
                    if order['price'] <= self._bid_book_prices[-1]:
                        return self._match_trade(order)
                    else:
                        self.add_order_to_book(order) 
                else:
                    self.add_order_to_book(order)
        else:
            book_prices = self._bid_book_prices if order['side'] == 'buy' else self._ask_book_prices
            if order['price'] in book_prices:
                book = self._bid_book if order['side'] == 'buy' else self._ask_book
                if order['order_id'] in book[order['price']]['orders']:
                    self._confirm_modify(order['timestamp'], order['side'], order['quantity'], order['order_id'])
                    if order['type'] == 'cancel':
                        self._remove_order(order['side'], order['price'], order['order_id'])
                    else: #order['type'] == 'modify'
                        self._modify_order(order['side'], order['quantity'], order['order_id'], order['price'])
    
    def _match_trade(self, order):
        '''Match orders to generate trades, update books.'''
        matched_trades = []
        self.traded = True
        self.confirm_trade_collector.clear()
        if order['side'] == 'buy':
            book_prices = self._ask_book_prices
            book = self._ask_book
            remainder = order['quantity']
            while remainder > 0:
                if book_prices:
                    price = book_prices[0]
                    if order['price'] >= price:
                        book_order_id = book[price]['order_ids'][0]
                        book_order = book[price]['orders'][book_order_id]
                        self._last_settled_price = book_order['price']
                        self._stock_price_history.append(self._last_settled_price)
                        if remainder >= book_order['quantity']:
                            self._confirm_trade(order['timestamp'], book_order['side'], book_order['quantity'], book_order['order_id'], book_order['price'])
                            t = self._add_trade_to_book(book_order['order_id'], book_order['timestamp'], order['order_id'], order['timestamp'], book_order['price'], 
                                                    book_order['quantity'], order['side'])
                            matched_trades.append(t)
                            self._remove_order(book_order['side'], book_order['price'], book_order['order_id'])
                            remainder -= book_order['quantity']
                        else:
                            self._confirm_trade(order['timestamp'], book_order['side'], remainder, book_order['order_id'], book_order['price'])
                            t = self._add_trade_to_book(book_order['order_id'], book_order['timestamp'], order['order_id'], order['timestamp'], book_order['price'],
                                                    remainder, order['side'])
                            matched_trades.append(t)
                            self._modify_order(book_order['side'], remainder, book_order['order_id'], book_order['price'])
                            break
                    else:
                        order['quantity'] = remainder
                        self.add_order_to_book(order)
                        break
                else:
                    order['quantity'] = remainder
                    self.add_order_to_book(order)
                    break
                    #print('Ask Market Collapse with order {0}'.format(order))
                    #break
        else: #order['side'] =='sell'
            book_prices = self._bid_book_prices
            book = self._bid_book
            remainder = order['quantity']
            while remainder > 0:
                if book_prices:
                    price = book_prices[-1]
                    if order['price'] <= price:
                        book_order_id = book[price]['order_ids'][0]
                        book_order = book[price]['orders'][book_order_id]
                        self._last_settled_price = book_order['price']
                        self._stock_price_history.append(self._last_settled_price)
                        if remainder >= book_order['quantity']:
                            self._confirm_trade(order['timestamp'], book_order['side'], book_order['quantity'], book_order['order_id'], book_order['price'])
                            t = self._add_trade_to_book(book_order['order_id'], book_order['timestamp'], order['order_id'], order['timestamp'], book_order['price'],
                                                    book_order['quantity'], order['side'])
                            matched_trades.append(t)
                            self._remove_order(book_order['side'], book_order['price'], book_order['order_id'])
                            remainder -= book_order['quantity']
                        else:
                            self._confirm_trade(order['timestamp'], book_order['side'], remainder, book_order['order_id'], book_order['price'])
                            t = self._add_trade_to_book(book_order['order_id'], book_order['timestamp'], order['order_id'], order['timestamp'], book_order['price'],
                                                    remainder, order['side'])
                            matched_trades.append(t)
                            self._modify_order(book_order['side'], remainder, book_order['order_id'], book_order['price'])
                            break
                    else:
                        order['quantity'] = remainder
                        self.add_order_to_book(order)
                        break
                else:
                    order['quantity'] = remainder
                    self.add_order_to_book(order)
                    break
                    
        return matched_trades
        
    def order_history_to_h5(self, filename):
        '''Append order history to an h5 file, clear the order_history'''
        temp_df = pd.DataFrame(self.order_history)
        temp_df.to_hdf(filename, 'orders', append=True, format='table', complevel=5, complib='blosc', 
                       min_itemsize={'order_id': 12}) 
        self.order_history.clear()
        
    def trade_book_to_h5(self, filename):
        '''Append trade_book to an h5 file, clear the trade_book'''
        temp_df = pd.DataFrame(self.trade_book)
        temp_df.to_hdf(filename, 'trades', append=True, format='table', complevel=5, complib='blosc', 
                       min_itemsize={'resting_order_id': 12, 'incoming_order_id': 12}) 
        self.trade_book.clear()
        
    def sip_to_h5(self, filename):
        '''Append _sip_collector to an h5 file, clear the _sip_collector'''
        temp_df = pd.DataFrame(self._sip_collector)
        temp_df.to_hdf(filename, 'tob', append=True, format='table', complevel=5, complib='blosc')
        self._sip_collector.clear()
    
    def report_top_of_book(self, now_time):
        '''Update the top-of-book prices and sizes'''
        best_bid_price = self._bid_book_prices[-1]
        best_bid_size = self._bid_book[best_bid_price]['size']   
        best_ask_price = self._ask_book_prices[0]
        best_ask_size = self._ask_book[best_ask_price]['size']
        tob = {'timestamp': now_time, 'best_bid': best_bid_price, 'best_ask': best_ask_price, 'bid_size': best_bid_size, 'ask_size': best_ask_size}
        self._sip_collector.append(tob)
        return tob
    
    def describe(self):
        pp = pprint.PrettyPrinter(indent=1)
        print("Order History")
        pp.pprint(self.order_history)
        print("\nBid Book")
        pp.pprint(self._bid_book)
        print("\nBid Book Prices")
        pp.pprint(self._bid_book_prices)
        print("\nAsk Book")
        pp.pprint(self._ask_book)
        print("\nAsk Book Prices")
        pp.pprint(self._ask_book_prices)
        print("\nConfirm Modify Collector")
        pp.pprint(self.confirm_modify_collector)
        print("\nConfirm Trade Collector")
        pp.pprint(self.confirm_trade_collector)
        print("\nSip Collector")
        print(self._sip_collector)
        print("\nTrade Book")
        pp.pprint(self.trade_book)
        print("\nOrder Index")
        pp.pprint(self._order_index)
        print("\nTraded")
        pp.pprint(self.traded)
        
    def get_price(self):
        if self._last_settled_price is not None:
            return self._last_settled_price
        elif len(self._ask_book_prices)>0:
            return self._ask_book_prices[0]
    
    def get_price_history(self):
        return self._stock_price_history 
    

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
        
    def make_add_order(self, stock,clock, buy_sell='buy',qty=1,price=None):
        if price == None:
            price = self.portfolio[stock][0].get_price() #use the market price if price not provided
        
        if price != None:
            return {'order_id': 'T'+str(self.ID)+'_'+str(self.order_no), 'timestamp': time.clock(), 'type': 'add', 'quantity': qty, 'side': buy_sell, 'price': price, 'clock':clock}
        else:
            print("Error: Price is None")
            
    def place_order(self, stock, order):
        if order['side'] == 'buy':
            if self.effective_funds >= order['price'] * order['quantity']:
                self.effective_funds -= order['price'] * order['quantity']
                self.order_no +=1
                return self.portfolio[stock][0].process_order(order), 0
            else:
                return None, order['price'] * order['quantity'] - self.effective_funds
                print("Not enough effective funds to place order")
        elif order['side'] == 'sell':
            if self.portfolio[stock][2] >= order['quantity']:
                self.portfolio[stock][2] -= order['quantity']
                self.order_no +=1
                return self.portfolio[stock][0].process_order(order), 0
            else:
                return None, 0
                print("Not enough effective qty to place order. Available effective qty = ", self.portfolio[stock][2])     
  

class Exchange(object):
    
    def __init__(self,num_agents,initial_money,IPO):

        self.clock = 0
        self.order_life = 2
        self.stocks = {"S"+str(i):Orderbook(IPO[list(IPO.keys())[i-1]][0]) for i in range(1,len(IPO)+1)}
        self.agents = {"T"+str(i):Agent(i,initial_money, self.stocks) for i in range(1,num_agents+1)}
        self.transaction_cost = [[] for i in range(num_agents)]
        
        self.agents["T-1"] = Agent(-1,0, self.stocks)
        
        #initializing the IPO agent's portfolio with stocks
        for stock in self.stocks.keys():
            self.agents["T-1"].portfolio[stock][1] = IPO[stock][1]
            self.agents["T-1"].portfolio[stock][2] = IPO[stock][1]
            self.place_add_order("T-1", stock, buy_sell='sell',qty=IPO[stock][1],price=IPO[stock][0])
            self.place_add_order("T-1", stock, buy_sell='buy',qty=IPO[stock][1],price=IPO[stock][0])
            
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
    
    def get_portfolio_value(self, agent): 
        tot_q = [tot_quantity for tot_quantity, eff_quantity in self.get_portfolio(agent).values()]
        stock_value = sum([tot_q[i]*self.stocks[price]._last_settled_price for i, price in enumerate(self.stocks.keys())])
        return stock_value + self.get_total_funds(agent)
    
    def place_add_order(self,agent, stock, buy_sell='buy',qty=1,price=None):
        o = self.agents[agent].make_add_order(stock, self.clock, buy_sell,qty,price)
        #print(o)
        trades, fs = self.agents[agent].place_order(stock, o)
        #print(trades)
        
        if trades != None:
            for trade in trades:
                io = self.stocks[stock].order_history[trade['incoming_order_id']]
                ro = self.stocks[stock].order_history[trade['resting_order_id']]
                io_t = io['order_id'].split('_')[0]
                ro_t = ro['order_id'].split('_')[0]
                
                self.do_bookkeeping(io_t, stock, trade, io)
                self.do_bookkeeping(ro_t, stock, trade, ro)
                
        return fs
    
    # agent -> agent's name
    def place_delta_add_order(self,agent,new_portfolio):
        self.clock += 1
        current_portfolio = self.get_portfolio(agent)
        print ("[" + agent + "]: current portfolio: ", current_portfolio)
        
        ### ms: new_portfolio is an array of arrays. Not sure why. So I just store the first element of it.
        ### Print(w2): [[0.16674656 0.16680138 0.16674682 0.16636996 0.16690944 0.1664259 ]]
        w2 = new_portfolio[0] 
        
        
        #assert len(new_portfolio)==len(current_portfolio), "len_current_portfolio: "+str(len(current_portfolio))+" len_new_portfolio "+str(len(new_portfolio)) 
        ### ms: {'S1': [stock price, quantity]}
        
        ### ms: to be deleted
        #new_portfolio = current_portfolio.copy()
        ###
        
        ### ms: Assumption: #######################
        ### ms:   current_portfolio.items() returns items ALWAYS in the same order, from S1, S2, to Sn.
        ### ms.   If this does not hold, we neet to replace Dict with OrderedDict, or to find an alternative way.
        i = 0
        buy_orders = set() #keep buy orders for later
        
        for stock, price_qty in current_portfolio.items():
            ### ms: How to do place order using 'different weights' -
            ### ms: First,
            ### ms:   we calculate a difference between two weights, one stock from new portfolio
            ### ms:   and the same stock from the current portfolio.
            ### ms:
            ###### ms' thoughts: well, I have not found any unfavorable case without cash weight in our portfolio.
            ######               However, something inconvenient always occurs without cash weight in the right place to use.
            
            ### ms: Get a current portfolio value.
            cur_pf_value = self.get_portfolio_value(agent)
            cur_stock_price = self.stocks[stock]._last_settled_price
            cur_stock_qty = price_qty[0]     # [0] = total quantity.   [1] = effective quantity
            cur_stock_amount = cur_stock_price * cur_stock_qty
            cur_stock_wt = cur_stock_amount/cur_pf_value
            
            
            
            diff_wt = w2[i+1] - cur_stock_wt
            
            ### ms: to be deleted
            #diff = new_portfolio[stock] - qty[1] 
            ###
        
            bs_amount = diff_wt * cur_pf_value
            
            self.agents[agent].total_funds -= abs(bs_amount) * 0.0001
#            self.transaction_cost[i] += abs(bs_amount) * 0.005
            
            
            new_qty = 0
            
            from random import randint
            
            if bs_amount >= 0:
                bs = 'buy'
                # buy at up to 2% more expensive
                cur_stock_price = cur_stock_price + randint(0,10)/10 
                new_qty = bs_amount/cur_stock_price
                new_qty = int(new_qty)  # We buy consertively in quantity to prevent overdraft (running out of cash)
            else:
                bs = 'sell'
                # sell at up to 2% cheaper
                cur_stock_price = cur_stock_price - randint(0,10)/10
                new_qty = bs_amount/cur_stock_price
                # We sell consertively to meet balance and to prevent overdraft by securing enough cash.
                new_qty = min(abs(math.floor(new_qty)), cur_stock_qty)
            
            if bs == 'sell':
                self.place_add_order(agent, stock, buy_sell=bs ,qty=new_qty, price=cur_stock_price)
            else:
                buy_orders.add((agent, stock, bs ,new_qty, cur_stock_price))   
        
        
#            bs_amount = diff_wt * cur_pf_value
#            new_qty = bs_amount/cur_stock_price
#            
#            if bs_amount >= 0:
#                bs = 'buy'
#                new_qty = int(new_qty)  # We buy consertively in quantity to prevent overdraft (running out of cash)
#            else:
#                bs = 'sell'
#                
#                # We sell consertively to meet balance and to prevent overdraft by securing enough cash.
#                new_qty = min(abs(math.floor(new_qty)), cur_stock_qty)
#            
#            if bs == 'sell':
#                self.place_add_order(agent, stock, buy_sell=bs ,qty=new_qty)
#            else:
#                buy_orders.add((agent, stock, bs ,new_qty))

            i = i + 1
#            print("Order#", i, " by ", agent, "is to ", bs, " ", new_qty, " shares of ", stock)

        fund_shortage = 0
        for buy_order in buy_orders:
            fund_shortage += self.place_add_order(*buy_order)
        
        self.remove_outdated_orders()
    
        return fund_shortage

                    
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
    
    def remove_outdated_orders(self):   
        for s, stock in self.stocks.items():
            bid_book_copy = copy.deepcopy(stock._bid_book)
            for price in bid_book_copy.keys():
                orders = bid_book_copy[price]
                for order_id in orders['orders'].keys():
                    order = stock._bid_book[price]['orders'][order_id]
                    #print((self.clock - order['clock']))
                    if (self.clock - order['clock']) > self.order_life:
                        stock._remove_order(order['side'], order['price'], order['order_id'])
                        agent = order_id.split('_')[0]
                        if order['side'] == 'buy' and order['type'] == 'add':
                            #restore effective funds
                            self.agents[agent].effective_funds += order['quantity'] * order['price']
                        elif order['side'] == 'sell' and order['type'] == 'add':
                            #restore effective qty
                            self.agents[agent].portfolio[s][2] += order['quantity']

        for s, stock in self.stocks.items():
            ask_book_copy = copy.deepcopy(stock._ask_book)
            for price in ask_book_copy.keys():
                orders = ask_book_copy[price]
                for order_id in orders['orders'].keys():
                    order = stock._ask_book[price]['orders'][order_id]
                    #print((self.clock - order['clock']))
                    if (self.clock - order['clock']) > self.order_life:
                        stock._remove_order(order['side'], order['price'], order['order_id'])
                        agent = order_id.split('_')[0]
                        if order['side'] == 'buy' and order['type'] == 'add':
                            #restore effective funds
                            self.agents[agent].effective_funds += order['quantity'] * order['price']
                        elif order['side'] == 'sell' and order['type'] == 'add':
                            #restore effective qty
                            self.agents[agent].portfolio[s][2] += order['quantity']
    
    

from argparse import ArgumentParser
import json
import pandas as pd
import numpy as np
import math
from decimal import Decimal
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt_cash
import matplotlib.pyplot as plt_s1
import matplotlib.pyplot as plt_s2
import matplotlib.pyplot as plt_s3
import matplotlib.pyplot as plt_s4
import matplotlib.pyplot as plt_s5


plt.style.use('ggplot')
plt_cash.style.use('ggplot')
plt_s1.style.use('ggplot')
plt_s2.style.use('ggplot')
plt_s3.style.use('ggplot')
plt_s4.style.use('ggplot')
plt_s5.style.use('ggplot')


from agents.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
from agents.pg import PG
import datetime
import os
import seaborn as sns
sns.set_style("darkgrid")


eps=10e-8
epochs=0
M=0
PATH_prefix=''



class StockTrader():
    def __init__(self):
        self.reset()

    def reset(self):
        self.wealth = 10e3
        self.total_reward = 0
        self.ep_ave_max_q = 0
        self.loss = 0
        self.actor_loss=0

        self.wealth_history = []
        self.r_history = []
        self.w_history = []
        self.p_history = []
        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(M))

    def update_summary(self,loss,r,q_value,actor_loss,w,p):
        self.loss += loss
        self.actor_loss+=actor_loss
        self.total_reward+=r
        self.ep_ave_max_q += q_value
        self.r_history.append(r)
        self.wealth = self.wealth * math.exp(r)
        self.wealth_history.append(self.wealth)
        self.w_history.extend([','.join([str(Decimal(str(w0)).quantize(Decimal('0.00'))) for w0 in w.tolist()[0]])])
        self.p_history.extend([','.join([str(Decimal(str(p0)).quantize(Decimal('0.000'))) for p0 in p.tolist()])])

    def write(self,codes,agent):
        global PATH_prefix
        wealth_history = pd.Series(self.wealth_history)
        r_history = pd.Series(self.r_history)
        w_history = pd.Series(self.w_history)
        p_history = pd.Series(self.p_history)
        history = pd.concat([wealth_history, r_history, w_history, p_history], axis=1)
        history.to_csv(PATH_prefix+agent + '-'.join(codes) + '-' + str(math.exp(np.sum(self.r_history)) * 100) + '.csv')

    def print_result(self,epoch,agent,noise_flag):
        print ("Total reward is: ", self.total_reward)
        self.total_reward=math.exp(self.total_reward) * 100
        print('*-----Episode: {:d}, Reward:{:.6f}%-----*'.format(epoch, self.total_reward))
#        agent.write_summary(self.total_reward)
#        agent.save_model()
        for i in range(len(agent)):
            agent[i].write_summary(self.total_reward)
            agent[i].save_model()

    def plot_result(self):
        pd.Series(self.wealth_history).plot()
        plt.show()

    def action_processor(self,a,ratio):
        a = np.clip(a + self.noise() * ratio, 0, 1)
        a = a / (a.sum() + eps)
        return a

def parse_info(info):
    return info['reward'],info['continue'],info[ 'next state'],info['weight vector'],info['price'],info['risk']





def maxdrawdown(arr):
    i = np.argmax((np.maximum.accumulate(arr) - arr)/np.maximum.accumulate(arr)) # end of the period
    j = np.argmax(arr[:i]) # start of period
    return (1-arr[i]/arr[j])

def backtest(agent,env):
    global PATH_prefix
    print("starting to backtest......")
    from agents.UCRP import UCRP
    from agents.Winner import WINNER
    from agents.Losser import LOSSER


    agents=[]
    agents.extend(agent)
    agents.append(WINNER())
    agents.append(UCRP())
    agents.append(LOSSER())
    labels=['PG','Winner','UCRP','Losser']

    wealths_result=[]
    rs_result=[]
    for i,agent in enumerate(agents):
        stocktrader = StockTrader()
        info = env.step(None, None,'False')
        r, contin, s, w1, p, risk = parse_info(info)
        contin = 1
        wealth=10000
        wealths = [wealth]
        rs=[1]
        while contin:
            w2 = agent.predict(s, w1)
            env_info = env.step(w1, w2,'False')
            r, contin, s_next, w1, p, risk = parse_info(env_info)
            wealth=wealth*math.exp(r)
            rs.append(math.exp(r)-1)
            wealths.append(wealth)
            s=s_next
            stocktrader.update_summary(0, r, 0, 0, w2, p)

        stocktrader.write(map(lambda x: str(x), env.get_codes()),labels[i])
        print('finish one agent')
        wealths_result.append(wealths)
        rs_result.append(rs)

    print('资产名称','   ','平均日收益率','   ','夏普率','   ','最大回撤')
    plt.figure(figsize=(8, 6), dpi=100)
    for i in range(len(agents)):
        plt.plot(wealths_result[i],label=labels[i])
        mrr=float(np.mean(rs_result[i])*100)
        sharpe=float(np.mean(rs_result[i])/np.std(rs_result[i])*np.sqrt(252))
        maxdrawdown=float(max(1-min(wealths_result[i])/np.maximum.accumulate(wealths_result[i])))
        print(labels[i],'   ',round(mrr,3),'%','   ',round(sharpe,3),'  ',round(maxdrawdown,3))
    plt.legend()
    plt.savefig(PATH_prefix+'backtest.png')
    plt.show()

def parse_config(config,mode):
    codes = config["session"]["codes"]
    start_date = config["session"]["start_date"]
    end_date = config["session"]["end_date"]
    features = config["session"]["features"]
    agent_config = config["session"]["agents"]
    market = config["session"]["market_types"]
    noise_flag, record_flag, plot_flag=config["session"]["noise_flag"],config["session"]["record_flag"],config["session"]["plot_flag"]
    predictor, framework, window_length = agent_config
    reload_flag, trainable=config["session"]['reload_flag'],config["session"]['trainable']
    method=config["session"]['method']

    global epochs
    epochs = int(config["session"]["epochs"])

    if mode=='test':
        record_flag='True'
        noise_flag='False'
        plot_flag='True'
        reload_flag='True'
        trainable='False'
        method='model_free'

    print("*--------------------Training Status-------------------*")
    print("Date from",start_date,' to ',end_date)
    print('Features:',features)
    print("Agent:Noise(",noise_flag,')---Recoed(',noise_flag,')---Plot(',plot_flag,')')
    print("Market Type:",market)
    print("Predictor:",predictor,"  Framework:", framework,"  Window_length:",window_length)
    print("Epochs:",epochs)
    print("Trainable:",trainable)
    print("Reloaded Model:",reload_flag)
    print("Method",method)
    print("Noise_flag",noise_flag)
    print("Record_flag",record_flag)
    print("Plot_flag",plot_flag)


    return codes,start_date,end_date,features,agent_config,market,predictor, framework, window_length,noise_flag, record_flag, plot_flag,reload_flag,trainable,method



def randomly(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return iter(shuffled)

def get_state(env, init_stock_price, price_history, N):
    # Before reshaping this arreay,
    # its initial shape is (# of stocks + 1, # of features, the size of lookback windows)
    state_before_reshaping = np.full( (env.M, N, env.L), 1.0 )
    
    for i in range(env.M):    # Suposed to be in 'range(env.M)', but for now..
        # Cash's relative price never changes. So, set it to 1 at every time step.
        if i == 0:
            state_before_reshaping[i][0] = np.full( (1,10), 1.0 )
            state_before_reshaping[i][1] = np.full( (1,10), 1.0 )
        else:
            # For stocks, we initially set the current prices (to each feature at the moment)
            # for the last 'L' time steps.
            for j in range(N):
                for k in range(env.L):
                    state_before_reshaping[i][j][k] = price_history[i-1][k]/init_stock_price[i-1]

    return state_before_reshaping


def get_init_state(env, e, init_stock_info, num_stocks, N):
 #    current_price = [e.stocks[i[0]]._last_settled_price for i in init_stock_info]
    
    ### ms:
    # Get the current prices for stocks as a list in an order S1, S2, ...,..
    current_price = [init_stock_info[i][1] for i in range(len(init_stock_info))]

    # Before reshaping this arreay,
    # its initial shape is (# of stocks + 1, # of features, the size of lookback windows)
    current_stock_price_list = np.full( (env.M, N, env.L), 1.0 )
    for i in range(len(current_price)+1):    # Suposed to be in 'range(env.M)', but for now..
        # Cash's relative price never changes. So, set it to 1 at every time step.
        if i == 0:
            current_stock_price_list[i][0] = np.full( (1,10), 1.0 )
            current_stock_price_list[i][1] = np.full( (1,10), 1.0 )
        else:
            # For stocks, we initially set the current prices (to each feature at the moment)
            # for the last 'L' time steps.
            for j in range(N):
                current_stock_price_list[i][j] = np.full( (1,10), 1)
        
    return current_stock_price_list


def session(config,args):
    global PATH_prefix
    codes, start_date, end_date, features, agent_config, market,predictor, framework, window_length,noise_flag, record_flag, plot_flag,reload_flag,trainable,method=parse_config(config,args)
    print ("codes, start_date, end_date, features, agent_config, market,predictor, framework, window_length,noise_flag, record_flag, plot_flag,reload_flag,trainable,method")
    print (codes, start_date, end_date, features, agent_config, market,predictor, framework, window_length,noise_flag, record_flag, plot_flag,reload_flag,trainable,method)
    
    num_stocks = 5
    init_stock_info = [ ['S1', 10],
                    ['S2', 20],
                    ['S3', 25],
                    ['S4', 40],
                    ['S5', 50]]
    init_money = 10000  
    num_rl_agents=3
    num_random_agents = 1
    num_mean_agents=1
    num_agents = num_random_agents+num_rl_agents+num_mean_agents



#   IPO for test case #1, #2    
#    IPO = {"S1":(10,500),"S2":(20,250),"S3":(25,200), "S4":(40,125), "S5":(50,100)} #price, qty
    
    
#   IPO for test case #3
    IPO = {"S1":(40,50000),"S2":(45,25000),"S3":(50,20000), "S4":(55,12500), "S5":(60,10000)} #price, qty
    e = Exchange(num_agents, init_money, IPO)
        
    init_stock_price = [value[0] for key, value in IPO.items()]

    ################################################################################
    ### Test case #1 : T1 has S1 of 100 shares at $10. -> Stock $1,000; Cash $9,000
    ###               T2 has S2 of  50 shares at $20. -> Stock $1,000; Cash $9,000
    # T1 : Stock name / Quantity(Q) / Price(P) / Cost(at bought price) / Total Cost
    #  1) S1 / 100 / 10 / 10 / 1,000 (Will be traded)
    #  2) S2 /   0 / 20 / 19 / 0     (Will fail to be traded due to a price mismatch)
#    e.place_add_order('T1', 'S1', buy_sell='buy',qty=100, price=10)   
#    e.place_add_order('T1', 'S2', buy_sell='buy',qty=50, price=19)    

    # T2 : Stock name / Quantity(Q) / Price(P) / Cost(at bought price) / Total Cost
    #  1) S1 /   0 / 10 /  9 / 0      (Will fail to be traded due to a price mismatch)
    #  2) S2 /  50 / 20 / 20 / 1,000  (Will be traded)
#    e.place_add_order('T2', 'S1', buy_sell='buy', qty=100, price=9) 
#    e.place_add_order('T2', 'S2', buy_sell='buy', qty=50, price=20) 
    ################################################################################


    ################################################################################
    ### Test case #2 : T1's portfolio is initialized to have an equal weight of 20% to each stock.
    ###                T2's portfolio is empty. T2 wanted to have the same portfolio as T1's,
    #                   but his buy orders do not match to any IPO price.
    # result: The maximum quantity for each stock that agents can trade is limited to
    # the quantity that T1 is holing at this initialization.
    #       	S1		S2		S3		S4		S5	
#    e.place_add_order('T1', 'S1', buy_sell='buy',qty=200, price=10)   
#    e.place_add_order('T1', 'S2', buy_sell='buy',qty=100, price=20)
#    e.place_add_order('T1', 'S3', buy_sell='buy',qty=80, price=25)
#    e.place_add_order('T1', 'S4', buy_sell='buy',qty=50, price=40)
#    e.place_add_order('T1', 'S5', buy_sell='buy',qty=40, price=50)
#
#    e.place_add_order('T2', 'S1', buy_sell='buy',qty=200, price=9)   
#    e.place_add_order('T2', 'S2', buy_sell='buy',qty=100, price=19)
#    e.place_add_order('T2', 'S3', buy_sell='buy',qty=80, price=24)
#    e.place_add_order('T2', 'S4', buy_sell='buy',qty=50, price=39)
#    e.place_add_order('T2', 'S5', buy_sell='buy',qty=40, price=49)
    ################################################################################



    ################################################################################
    ### Test case #3 : T1's portfolio is initialized to have an equal weight of 20% to each stock.
    ###                T2's portfolio is initialized to have the half of T1's quantity of each stock at the same price
    #	S1		S2		S3		S4		S5			
    #Agent	Q	P	Q	P	Q	P	Q	P	Q	P	Total	Stock	Cash	Total
    #   T1	200	10	100	20	80	25	50	40	40	50	10,000	10,000	0	    10,000
    #   T2	100	10	50	20	40	25	25	40	20	50	10,000	5,000	5,000	10,000
#    e.place_add_order('T1', 'S1', buy_sell='buy',qty=200, price=10)   
#    e.place_add_order('T1', 'S2', buy_sell='buy',qty=100, price=20)
#    e.place_add_order('T1', 'S3', buy_sell='buy',qty=80, price=25)
#    e.place_add_order('T1', 'S4', buy_sell='buy',qty=50, price=40)
#    e.place_add_order('T1', 'S5', buy_sell='buy',qty=40, price=50)
#
#    e.place_add_order('T2', 'S1', buy_sell='buy',qty=100, price=10)   
#    e.place_add_order('T2', 'S2', buy_sell='buy',qty=50, price=20)
#    e.place_add_order('T2', 'S3', buy_sell='buy',qty=40, price=25)
#    e.place_add_order('T2', 'S4', buy_sell='buy',qty=25, price=40)
#    e.place_add_order('T2', 'S5', buy_sell='buy',qty=20, price=50)

    ################################################################################



   ################################################################################
    ### Test case #4 : T1's portfolio is initialized to have an equal weight of 16.7% to each stock.
    ###                T2's portfolio is initialized to have an equal weight of 10% to each stock.
    #       	S1		S2		S3		S4		S5	

#    e.place_add_order('T1', 'S1', buy_sell='buy',qty=166, price=10)   
#    e.place_add_order('T1', 'S2', buy_sell='buy',qty=83, price=20)
#    e.place_add_order('T1', 'S3', buy_sell='buy',qty=66, price=25)
#    e.place_add_order('T1', 'S4', buy_sell='buy',qty=41, price=40)
#    e.place_add_order('T1', 'S5', buy_sell='buy',qty=33, price=50)
#
#    e.place_add_order('T2', 'S1', buy_sell='buy',qty=100, price=11)   
#    e.place_add_order('T2', 'S2', buy_sell='buy',qty=50, price=21)
#    e.place_add_order('T2', 'S3', buy_sell='buy',qty=40, price=26)
#    e.place_add_order('T2', 'S4', buy_sell='buy',qty=25, price=41)
#    e.place_add_order('T2', 'S5', buy_sell='buy',qty=20, price=51)
    ################################################################################


    ################################################################################
    ### Test case #5 : T1's portfolio is initialized to have an equal weight of 20% to each stock.
    ###                T2's portfolio is initialized to have the half of T1's quantity of each stock at the same price

#    e.place_add_order('T1', 'S1', buy_sell='buy',qty=200, price=10)   
#    e.place_add_order('T1', 'S2', buy_sell='buy',qty=100, price=20)
#    e.place_add_order('T1', 'S3', buy_sell='buy',qty=80, price=25)
#    e.place_add_order('T1', 'S4', buy_sell='buy',qty=50, price=40)
#    e.place_add_order('T1', 'S5', buy_sell='buy',qty=40, price=50)
#
#    e.place_add_order('T2', 'S1', buy_sell='buy',qty=100, price=10)   
#    e.place_add_order('T2', 'S2', buy_sell='buy',qty=50, price=20)
#    e.place_add_order('T2', 'S3', buy_sell='buy',qty=40, price=25)
#    e.place_add_order('T2', 'S4', buy_sell='buy',qty=25, price=40)
#    e.place_add_order('T2', 'S5', buy_sell='buy',qty=20, price=50)
#    
#    e.place_add_order('T3', 'S1', buy_sell='buy',qty=10, price=10)   
#    e.place_add_order('T3', 'S2', buy_sell='buy',qty=5, price=20)
#    e.place_add_order('T3', 'S3', buy_sell='buy',qty=4, price=25)
#    e.place_add_order('T3', 'S4', buy_sell='buy',qty=2, price=40)
#    e.place_add_order('T3', 'S5', buy_sell='buy',qty=2, price=50)

#    e.place_add_order('T4', 'S1', buy_sell='buy',qty=500, price=10)   
#    e.place_add_order('T4', 'S5', buy_sell='buy',qty=100, price=50)
#    
#    e.place_add_order('T5', 'S2', buy_sell='buy',qty=200, price=20)   
#    e.place_add_order('T5', 'S3', buy_sell='buy',qty=80, price=25)
#    e.place_add_order('T5', 'S4', buy_sell='buy',qty=100, price=40)
    ################################################################################


    ################################################################################
    ### Test case #6 : T1's portfolio is initialized to have an equal weight of 20% to each stock.
    ###                T2's portfolio is initialized to have the half of T1's quantity of each stock at the same price
    #Agent	Q	P	Q	P	Q	P	Q	P	Q	P	Total	Stock	Cash	Total
    #   T1	200	10	100	20	80	25	50	40	40	50	10,000	10,000	0	10,000
    #   T2	100	10	50	20	40	25	25	40	20	50	10,000	5,000	5,000	10,000
    #   T3	10	10	5	20	4	25	2	40	2	50	10,000	480	9,520	10,000
#    e.place_add_order('T1', 'S1', buy_sell='buy',qty=200, price=10)   
#    e.place_add_order('T1', 'S2', buy_sell='buy',qty=100, price=20)
#    e.place_add_order('T1', 'S3', buy_sell='buy',qty=80, price=25)
#    e.place_add_order('T1', 'S4', buy_sell='buy',qty=50, price=40)
#    e.place_add_order('T1', 'S5', buy_sell='buy',qty=40, price=50)
#
#    e.place_add_order('T2', 'S1', buy_sell='buy',qty=100, price=10)   
#    e.place_add_order('T2', 'S2', buy_sell='buy',qty=50, price=20)
#    e.place_add_order('T2', 'S3', buy_sell='buy',qty=40, price=25)
#    e.place_add_order('T2', 'S4', buy_sell='buy',qty=25, price=40)
#    e.place_add_order('T2', 'S5', buy_sell='buy',qty=20, price=50)
#    
#    e.place_add_order('T3', 'S1', buy_sell='buy',qty=100, price=10)   
#    e.place_add_order('T3', 'S2', buy_sell='buy',qty=50, price=20)
#    e.place_add_order('T3', 'S3', buy_sell='buy',qty=40, price=25)
#    e.place_add_order('T3', 'S4', buy_sell='buy',qty=25, price=40)
#    e.place_add_order('T3', 'S5', buy_sell='buy',qty=20, price=50)
#    
#    e.place_add_order('T3', 'S1', buy_sell='buy',qty=50, price=10)   
#    e.place_add_order('T3', 'S2', buy_sell='buy',qty=90, price=20)
#    e.place_add_order('T3', 'S3', buy_sell='buy',qty=20, price=25)
#    e.place_add_order('T3', 'S4', buy_sell='buy',qty=10, price=40)
#    e.place_add_order('T3', 'S5', buy_sell='buy',qty=10, price=50)

    ################################################################################
    
    
    

    ################################################################################
    ### Test case #7 : T1's portfolio is initialized to have an equal weight of 20% to each stock.
    ###                T2's portfolio is initialized to have the half of T1's quantity of each stock at the same price
    #Agent	Q	P	Q	P	Q	P	Q	P	Q	P	Total	Stock	Cash	Total
    #   T1	200	10	100	20	80	25	50	40	40	50	10,000	10,000	0	10,000
    #   T2	100	10	50	20	40	25	25	40	20	50	10,000	5,000	5,000	10,000
    #   T3	10	10	5	20	4	25	2	40	2	50	10,000	480	9,520	10,000
#    e.place_add_order('T1', 'S1', buy_sell='buy',qty=100, price=10)   
#    e.place_add_order('T1', 'S2', buy_sell='buy',qty=50, price=20)
#    e.place_add_order('T1', 'S3', buy_sell='buy',qty=40, price=25)
#    e.place_add_order('T1', 'S4', buy_sell='buy',qty=25, price=40)
#    e.place_add_order('T1', 'S5', buy_sell='buy',qty=20, price=50)
#
#    e.place_add_order('T2', 'S1', buy_sell='buy',qty=100, price=10)   
#    e.place_add_order('T2', 'S2', buy_sell='buy',qty=50, price=20)
#    e.place_add_order('T2', 'S3', buy_sell='buy',qty=40, price=25)
#    e.place_add_order('T2', 'S4', buy_sell='buy',qty=25, price=40)
#    e.place_add_order('T2', 'S5', buy_sell='buy',qty=20, price=50)
#    
#    e.place_add_order('T3', 'S1', buy_sell='buy',qty=100, price=10)   
#    e.place_add_order('T3', 'S2', buy_sell='buy',qty=50, price=20)
#    e.place_add_order('T3', 'S3', buy_sell='buy',qty=40, price=25)
#    e.place_add_order('T3', 'S4', buy_sell='buy',qty=25, price=40)
#    e.place_add_order('T3', 'S5', buy_sell='buy',qty=20, price=50)
    
    
    ################################################################################
    ### Test case #8 : T1's portfolio is initialized to have an equal weight of 20% to each stock.
    ###                T2's portfolio is initialized to have the half of T1's quantity of each stock at the same price

#    e.place_add_order('T1', 'S1', buy_sell='buy',qty=80, price=40)   
#    e.place_add_order('T1', 'S2', buy_sell='buy',qty=60, price=45)
#    e.place_add_order('T1', 'S3', buy_sell='buy',qty=40, price=50)
#    e.place_add_order('T1', 'S4', buy_sell='buy',qty=20, price=55)
#    e.place_add_order('T1', 'S5', buy_sell='buy',qty=10, price=60)
#
#    e.place_add_order('T2', 'S1', buy_sell='buy',qty=70, price=40)   
#    e.place_add_order('T2', 'S2', buy_sell='buy',qty=50, price=45)
#    e.place_add_order('T2', 'S3', buy_sell='buy',qty=30, price=50)
#    e.place_add_order('T2', 'S4', buy_sell='buy',qty=10, price=55)
#    e.place_add_order('T2', 'S5', buy_sell='buy',qty=0, price=60)
#    
#    e.place_add_order('T3', 'S1', buy_sell='buy',qty=8, price=40)   
#    e.place_add_order('T3', 'S2', buy_sell='buy',qty=6, price=45)
#    e.place_add_order('T3', 'S3', buy_sell='buy',qty=4, price=50)
#    e.place_add_order('T3', 'S4', buy_sell='buy',qty=2, price=55)
#    e.place_add_order('T3', 'S5', buy_sell='buy',qty=0, price=60)
#    
#    # Random agent
#    e.place_add_order('T4', 'S1', buy_sell='buy',qty=8, price=40)   
#    e.place_add_order('T4', 'S2', buy_sell='buy',qty=6, price=45)
#    e.place_add_order('T4', 'S3', buy_sell='buy',qty=4, price=50)
#    e.place_add_order('T4', 'S4', buy_sell='buy',qty=2, price=55)
#    e.place_add_order('T4', 'S5', buy_sell='buy',qty=0, price=60)
#    
#    # Mean-reversion agent
#    e.place_add_order('T5', 'S1', buy_sell='buy',qty=8, price=40)   
#    e.place_add_order('T5', 'S2', buy_sell='buy',qty=6, price=45)
#    e.place_add_order('T5', 'S3', buy_sell='buy',qty=4, price=50)
#    e.place_add_order('T5', 'S4', buy_sell='buy',qty=2, price=55)
#    e.place_add_order('T5', 'S5', buy_sell='buy',qty=0, price=60)
    
   ################################################################################
    ### Test case #9 : T1's portfolio is initialized to have an equal weight of 20% to each stock.
    ###                T2's portfolio is initialized to have the half of T1's quantity of each stock at the same price

#    e.place_add_order('T1', 'S1', buy_sell='buy',qty=1, price=40)   
#    e.place_add_order('T1', 'S2', buy_sell='buy',qty=1, price=45)
#    e.place_add_order('T1', 'S3', buy_sell='buy',qty=1, price=50)
#    e.place_add_order('T1', 'S4', buy_sell='buy',qty=1, price=55)
#    e.place_add_order('T1', 'S5', buy_sell='buy',qty=1, price=60)
#
#    e.place_add_order('T2', 'S1', buy_sell='buy',qty=1, price=40)   
#    e.place_add_order('T2', 'S2', buy_sell='buy',qty=1, price=45)
#    e.place_add_order('T2', 'S3', buy_sell='buy',qty=1, price=50)
#    e.place_add_order('T2', 'S4', buy_sell='buy',qty=1, price=55)
#    e.place_add_order('T2', 'S5', buy_sell='buy',qty=0, price=60)
#    
#    e.place_add_order('T3', 'S1', buy_sell='buy',qty=8, price=40)   
#    e.place_add_order('T3', 'S2', buy_sell='buy',qty=6, price=45)
#    e.place_add_order('T3', 'S3', buy_sell='buy',qty=4, price=50)
#    e.place_add_order('T3', 'S4', buy_sell='buy',qty=2, price=55)
#    e.place_add_order('T3', 'S5', buy_sell='buy',qty=0, price=60)
#    
#    # Random agent
#    e.place_add_order('T4', 'S1', buy_sell='buy',qty=8, price=40)   
#    e.place_add_order('T4', 'S2', buy_sell='buy',qty=6, price=45)
#    e.place_add_order('T4', 'S3', buy_sell='buy',qty=4, price=50)
#    e.place_add_order('T4', 'S4', buy_sell='buy',qty=2, price=55)
#    e.place_add_order('T4', 'S5', buy_sell='buy',qty=0, price=60)
#    
#    # Mean-reversion agent
    e.place_add_order('T5', 'S1', buy_sell='buy',qty=8, price=40)   
    e.place_add_order('T5', 'S2', buy_sell='buy',qty=6, price=45)
    e.place_add_order('T5', 'S3', buy_sell='buy',qty=4, price=50)
    e.place_add_order('T5', 'S4', buy_sell='buy',qty=2, price=55)
    e.place_add_order('T5', 'S5', buy_sell='buy',qty=0, price=60)
    
    print("T1:", e.agents["T1"].get_portfolio())
    print("T2:", e.agents["T2"].get_portfolio())
    print("T3:", e.agents["T3"].get_portfolio())
    print("T4:", e.agents["T4"].get_portfolio())
    print("T5:", e.agents["T5"].get_portfolio())
    

    codes = [key for key in IPO.keys()]
    env = Environment(e, codes, init_stock_price, num_agents)
    
    init_state = get_init_state(env, e, init_stock_info, num_stocks, len(features))
    init_state = init_state.reshape(1, env.M, env.L, len(features))

    #global M
    M=len(codes)+1
#
#    if framework == 'DDPG':
#        print("*-----------------Loading DDPG Agent---------------------*")
#        from agents.ddpg import DDPG
#        agent = DDPG(predictor, len(codes) + 1, int(window_length), len(features), '-'.join(agent_config)+str(i), reload_flag,trainable)
#    
#    elif framework == 'PPO':
#        print("*-----------------Loading PPO Agent---------------------*")
#        from agents.ppo import PPO
#        agent = PPO(predictor, len(codes) + 1, int(window_length), len(features), '-'.join(agent_config)+str(i), reload_flag,trainable)
    
    

    stocktrader = []
    agent = []
    
    if framework == 'DDPG':
        print("*-----------------Loading DDPG Agent---------------------*")
        from agents.ddpg import DDPG
        for i in range(num_agents):
            agent.append( DDPG(predictor, len(codes) + 1, int(window_length), len(features), '-'.join(agent_config)+str(i), reload_flag,trainable) )
    
    elif framework == 'PPO':
        print("*-----------------Loading PPO Agent---------------------*")
        from agents.ppo import PPO
        for i in range(num_agents):
            agent.append( PPO(predictor, len(codes) + 1, int(window_length), len(features), '-'.join(agent_config)+str(i), reload_flag,trainable) )


    ### for Sourabh: 
    for i in range(num_agents):
        stocktrader.append(StockTrader())



    PATH_prefix = "result/PG/" + str(args['num']) + '/'

    portfolio_values = [[] for i in range(num_agents)]
    portfolio_weights_cash = [[] for i in range(num_agents)]
    portfolio_weights_s1 = [[] for i in range(num_agents)]
    portfolio_weights_s2 = [[] for i in range(num_agents)]
    portfolio_weights_s3 = [[] for i in range(num_agents)]
    portfolio_weights_s4 = [[] for i in range(num_agents)]
    portfolio_weights_s5 = [[] for i in range(num_agents)]
    stock_prices = [[] for i in range(num_stocks)]

    if args['mode']=='train':

            if framework == 'PG':
                print("*-----------------Loading PG Agent---------------------*")
                for i in range(num_agents):
                    agent.append(PG(len(codes) + 1, int(window_length), len(features), '-'.join(agent_config)+str(i), reload_flag,
                           trainable,noise_flag,args['num']))
                    print("Loaded Agent "+str(i))

            print("Training with {:d}".format(epochs))
            for epoch in range(epochs):
                print("Now we are at epoch", epoch)
                portfolio_value, portfolio_weights = traversal(stocktrader,agent,env,epoch,noise_flag,framework,method,trainable,e, num_rl_agents,num_random_agents,num_mean_agents, num_stocks)
                portfolio_values = np.hstack((portfolio_values,portfolio_value))
                portfolio_weights_cash = portfolio_weights[0]
                portfolio_weights_s1 = portfolio_weights[1]
                portfolio_weights_s2 = portfolio_weights[2]
                portfolio_weights_s3 = portfolio_weights[3]
                portfolio_weights_s4 = portfolio_weights[4]
                portfolio_weights_s5 = portfolio_weights[5]
                print(portfolio_values)

                if record_flag=='True':
                    stocktrader.write(epoch,framework)

                if plot_flag=='True':
                    stocktrader.plot_result()
                for iii in range(len(agent)):
                    agent[iii].reset_buffer()
                    stocktrader[iii].print_result(epoch,agent,noise_flag)
                    stocktrader[iii].reset()

            print ("portofolio values shape: ", portfolio_values.shape)


            for i in range(len(portfolio_values)):
                plt.plot( list(range(len(portfolio_values[0]))), portfolio_values[i] , label='$Agent {i}: P/F value$'.format(i=i+1))

            plt.legend(loc='best')
            plt.savefig('./portfolio_value.png')



#            plt_cash.figure(figsize=(8, 6), dpi=100)
#            plt_s1.figure(figsize=(8, 6), dpi=100)
#            plt_s2.figure(figsize=(8, 6), dpi=100)
#            plt_s3.figure(figsize=(8, 6), dpi=100)
#            plt_s4.figure(figsize=(8, 6), dpi=100)
#            plt_s5.figure(figsize=(8, 6), dpi=100)
#            
#            
#            for i in range(len(portfolio_weights_cash)):
#                plt_cash.plot( list(range(len(portfolio_weights_cash[0]))), portfolio_weights_cash[i] , label='$Agent {i}: Weight-S1$'.format(i=i+1))
#
#            plt_cash.legend(loc='best')
#            plt_cash.savefig('./portfolio_weights_s1.png')            
#            
#            for i in range(len(portfolio_weights_s1)):
#                plt_s1.plot( list(range(len(portfolio_weights_s1[0]))), portfolio_weights_s1[i] , label='$Agent {i}: Weight-S1$'.format(i=i+1))
#
#            plt_s1.legend(loc='best')
#            plt_s1.savefig('./portfolio_weights_s1.png')
#
#            
#            for i in range(len(portfolio_weights_s2)):
#                plt_s2.plot( list(range(len(portfolio_weights_s2[0]))), portfolio_weights_s2[i] , label='$Agent {i}: Weight-S2$'.format(i=i+1))
#
#            plt_s2.legend(loc='best')
#            plt_s2.savefig('./portfolio_weights_s2.png')
#        
#
#            
#            for i in range(len(portfolio_weights_s3)):
#                plt_s3.plot( list(range(len(portfolio_weights_s3[0]))), portfolio_weights_s3[i] , label='$Agent {i}: Weight-S3$'.format(i=i+1))
#
#            plt_s3.legend(loc='best')
#            plt_s3.savefig('./portfolio_weights_s3.png')
#
#            
#            for i in range(len(portfolio_weights_s4)):
#                plt_s4.plot( list(range(len(portfolio_weights_s4[0]))), portfolio_weights_s4[i] , label='$Agent {i}: Weight-S4$'.format(i=i+1))
#
#            plt_s4.legend(loc='best')
#            plt_s4.savefig('./portfolio_weights_s4.png')
#
#            
#            for i in range(len(portfolio_weights_s5)):
#                plt_s5.plot( list(range(len(portfolio_weights_s5[0]))), portfolio_weights_s5[i] , label='$Agent {i}: Weight-S5$'.format(i=i+1))
#
#            plt_s5.legend(loc='best')
#            plt_s5.savefig('./portfolio_weights_s5.png')

            
            for iii in range(len(agent)):
                agent[iii].close()
            del agent

    elif args['mode']=='test':
        with open("result/PG/" + str(args['num']) + '/config.json', 'r') as f:
            dict_data=json.load(f)
        backtest([PG(len(codes) + 1, int(window_length), len(features), '-'.join(agent_config)+str(i), 'True','False','True',args['num'])],
                 env)
        
def take_mean_rev_action(w1, num_stocks,e,mean_t):
    
    long_window_length = 20
    delta = 0.05

    w2 = w1
    # print('w2_shape',w2.shape)

    if(mean_t>=long_window_length):
        for i in range(num_stocks): 

            price_history = e.stocks['S'+str(i+1)].get_price_history()

            stock_price = e.stocks['S'+str(i+1)].get_price()
        
            recent_price_history = price_history[-long_window_length:]

            mean = np.mean(recent_price_history)
            std = np.std(recent_price_history)
            
            z_score = (stock_price - mean)/std
            
            
            if(z_score>1):

                w2[0][0]+=delta
                w2[0][i+1] -=delta
               

            elif(-1<z_score<=1):
                w2[0] = w2[0]
            else:
                
                w2[0][0]-= delta
                w2[0][i+1]+=delta
        
    return w2

def take_random_action(w1):

    w2=w1
    a = np.random.uniform(0,1,size=(1,w1.shape[1]))
    a /= a.sum()
                # a -= 1/(len(a))
                # print('a',a)
    w2 =  a

    return w2
        
def traversal(stocktrader,agent,env,epoch,noise_flag,framework,method,trainable,e,num_rl_agents,num_random_agents,num_mean_agents, num_stocks):
    info = env.step(None,None,noise_flag, 0)   # Initialization

#        return info['reward'],info['continue'],info[ 'next state'],info['weight vector'],info['price'],info['risk']
    r,contin, init_state, w1_for_all,p,risk=parse_info(info)
    p = info['price']
    contin=1
    t=0
    mean_t=1
    w1 = [[] for k in range(len(agent))]
    w2 = [[] for k in range(len(agent))]
    target_weight = [[] for k in range(len(agent))]
    state = [init_state for k in range(len(agent))]
    portfolio_values = [[] for i in range(len(agent))]
    portfolio_weights_cash = [[] for i in range(len(agent))]
    portfolio_weights_s1 = [[] for i in range(len(agent))]
    portfolio_weights_s2 = [[] for i in range(len(agent))]
    portfolio_weights_s3 = [[] for i in range(len(agent))]
    portfolio_weights_s4 = [[] for i in range(len(agent))]
    portfolio_weights_s5 = [[] for i in range(len(agent))]
    stock_prices = [[] for i in range(len(agent))]
    count_contin = 1
    while contin:

        print('count_contin', count_contin)
        count_contin+=1
        for agent_idx in randomly(range(len(agent))):
            #print ("state shape to agent: ", s.shape)
            agent_code = "T" + str(agent_idx+1)            
            current_portfolio = e.get_portfolio(agent_code)
            stock_weight = []
            
            for stock, price_qty in current_portfolio.items():
                ### ms: How to do place order using 'different weights' -
                ### ms: First,
                ### ms:   we calculate a difference between two weights, one stock from new portfolio
                ### ms:   and the same stock from the current portfolio.
                ### ms:
                ###### ms' thoughts: well, I have not found any unfavorable case without cash weight in our portfolio.
                ######               However, something inconvenient always occurs without cash weight in the right place to use.
                
                ### ms: Get a current portfolio value.
                cur_pf_value = e.get_portfolio_value(agent_code)
                cur_stock_price = e.stocks[stock]._last_settled_price
                cur_stock_qty = price_qty[0]     # [0] = total quantity.   [1] = effective quantity
                cur_stock_amount = cur_stock_price * cur_stock_qty
                cur_stock_wt = cur_stock_amount/cur_pf_value
                stock_weight.append(cur_stock_wt)

            w1[agent_idx] = np.array([[e.get_total_funds(agent_code)/cur_pf_value] + stock_weight])

            # original w1's format
            # 1st: array([[1, 0, 0, 0, 0, 0]]), array([[1, 0, 0, 0, 0, 0]])]
            # 2nd: [array([[0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
            #       0.16666667]], dtype=float32), array([[1, 0, 0, 0, 0, 0]])]
            # 3rd [array([[0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
            #        0.16666667]], dtype=float32), array([[0.16668212, 0.16668855, 0.16663074, 0.16666862, 0.16669858,
            #        0.16663139]], dtype=float32)]

            if(agent_idx<=(num_rl_agents-1)):

                w2[agent_idx] = agent[agent_idx].predict(state[agent_idx],w1[agent_idx])
            elif(agent_idx<=(num_rl_agents+num_random_agents-1)):
                # print('agent_idx',agent_idx)
                print('random agent is ', agent_idx)
                # a = np.random.uniform(0,1,size=(1,w1[agent_idx].shape[1]))
                # a /= a.sum()
                # a -= 1/(len(a))
                # print('a',a)
                w2[agent_idx] =  take_random_action(w1[agent_idx]) #+ w1[agent_idx] 

            elif(agent_idx<=(num_rl_agents+num_random_agents+num_mean_agents-1)):

                w2[agent_idx] = take_mean_rev_action(w1[agent_idx], num_stocks,e,mean_t) 



            # elif(agent_idx<=(num_rl_agents+num_random_agents+num_mean_agents-1)):

            #     delta = 0.1

            #     w_2_bin = 


                
            #print (w1.shape, w2.shape) (1,6) (1,6)

            ##########
            # env.step returns:
            # info = {'reward': reward, 'continue': not_terminal, 'next state': next_state,
            #        'weight vector': w2, 'price': price,'risk':risk}
            # 'continue' is zero if it's reset.
            env_info = env.step(w1[agent_idx], w2[agent_idx],noise_flag, agent_idx)
            
            #########
            # r <- info['reward']
            # contin <- info['continue']
            # s_next <- info[ 'next state']
            # w1 <- info['weight vector']
            # p <- info['price']
            #risk <- info['risk']
            
            # target_weight[agent_idx] is just equal to w2[agent_odx]. It's a useless vector.
            # s_next is a vector of asset(=cash and stocks) prices relative to their initial price.
            r, contin, s_next, target_weight[agent_idx], p ,risk = parse_info(env_info)

            if framework=='PG': 
                agent[agent_idx].save_transition(s_next,p,w2[agent_idx],w1[agent_idx])
            else:
                agent[agent_idx].save_transition(init_state, w2[agent_idx], r-risk, contin, s_next, w1[agent_idx])
            loss, q_value,actor_loss=0,0,0

            if framework=='DDPG':
                if not contin and trainable=="True":
                    agent_info= agent[agent_idx].train(method,epoch)
                    loss, q_value=agent_info["critic_loss"],agent_info["q_value"]
                    if method=='model_based':
                        actor_loss=agent_info["actor_loss"]

            elif framework=='PPO':
                if not contin and trainable=="True":
                    agent_info = agent[agent_idx].train(method, epoch)
                    loss, q_value = agent_info["critic_loss"], agent_info["q_value"]
                    if method=='model_based':
                        actor_loss=agent_info["actor_loss"]

            elif framework=='PG':
                if not contin and trainable=="True":
                    agent[agent_idx].train()
            #p = np.asarray(p,dtype=np.float32)
            #print ("p before update_sum: ", p)
            
            stocktrader[agent_idx].update_summary(loss,r,q_value,actor_loss,w2[agent_idx],p)
            state[agent_idx] = s_next
            t=t+1
       
            if not contin:
                break
        mean_t+=1

        for p_idx in range(len(agent)):
            portfolio_values[p_idx].append(e.get_portfolio_value("T" + str(p_idx+1)))
            print("w1[p_idx]", w1[p_idx])
            portfolio_weights_cash[p_idx].append(w1[p_idx][0][0])
            portfolio_weights_s1[p_idx].append(w1[p_idx][0][1])
            portfolio_weights_s2[p_idx].append(w1[p_idx][0][2])
            portfolio_weights_s3[p_idx].append(w1[p_idx][0][3])
            portfolio_weights_s4[p_idx].append(w1[p_idx][0][4])
            portfolio_weights_s5[p_idx].append(w1[p_idx][0][5])

            
    
    portfolio_values = np.array(portfolio_values)
    portfolio_weights = [portfolio_weights_cash, portfolio_weights_s1, portfolio_weights_s2, portfolio_weights_s3, portfolio_weights_s4, portfolio_weights_s5]
    
#    print ("portofolio values shape: ", portfolio_values.shape)
#    for i in range(len(portfolio_values)):
#        plt.plot( list(range(len(portfolio_values[0]))), portfolio_values[i] , label='$Agent {i}$'.format(i=i))
#    plt.legend(loc='best')
#    plt.savefig('./image.png')
    
    return portfolio_values, portfolio_weights

import numpy as np
import pandas as pd
from math import log
from datetime import datetime
import time
import random

eps=10e-8

def fill_zeros(x):
    return '0'*(6-len(x))+x


class Environment:
    def __init__(self, e, codes, init_stock_price, num_agents):
        self.cost=0.0025
        self.e = e
        self.FLAG = False
        self.state=[]
        self.codes = codes
        self.M=len(codes)+1
        self.t = 0
        self.L = 10
        self.init_stock_price = init_stock_price
        self.num_agents = num_agents
        self.old_portfolio_value = [0.] * num_agents

    def get_rel_price(self, price):
        rel_price = []
        rel_price.append(1.0)   # the relative price of cash is always equal to 1
        for i in range(len(self.init_stock_price)) :
#            rel_price.append((float)(price[i+1]/self.init_stock_price[i]))
            rel_price.append((float)(price[i+1]/price[i]))
        
        return rel_price
    
    
    def get_price_history_for_all(self, L) :
        price_history = []
        for i, stock in enumerate(self.e.stocks.keys()):
            temp_price = self.e.stocks[stock].get_price_history()
            if len(temp_price) >= L:
                price_history.append(temp_price)
            else:
                padding_price = []
                for xxx in range(L - len(temp_price)):
                    padding_price.append(temp_price[0])
                price_history.append(temp_price + padding_price)
  
        return price_history

    def get_market_return(self, price_history) :
        market_cum_return_t_minus_1 = 0
        market_cum_return_t = 0
        for k in range(len(price_history)) :
            market_cum_return_t_minus_1 += 10000 * (price_history[k][-2]/self.init_stock_price[k]-1)
            market_cum_return_t += 10000 * (price_history[k][-1]/self.init_stock_price[k]-1)
        market_cum_return_t_minus_1 = market_cum_return_t_minus_1 / len(price_history)
        market_cum_return_t = market_cum_return_t / len(price_history)
        
        market_return_t = market_cum_return_t - market_cum_return_t_minus_1
        
        return market_return_t

    def step(self,w1,w2,noise, agent_idx):
        N  = 2
        if self.FLAG:
            not_terminal = 1
            price = [[1]]
            price += ([[self.e.stocks[stock]._last_settled_price] for stock in self.e.stocks.keys()])
            #print ("Price: ", price)

#            if noise=='True':
#                price=price+np.stack(np.random.normal(0,0.002,(1,len(price))),axis=1)

            #print ("Price after noise: ", price)
            mu = self.cost * (np.abs(w2[0][1:] - w1[0][1:])).sum()
            #self.e.place_delta_add_order(self.e.agents["T"+str(agent_idx+1)], w2)
            print ("At t=", self.t, "[T"+str(agent_idx+1), "]  w1 and w2: ", np.around(w1,3), np.around(w2,3))
            self.e.place_delta_add_order("T"+str(agent_idx+1), w2)
            # std = self.states[self.t - 1][0].std(axis=0, ddof=0)
            # w2_std = (w2[0]* std).sum()

            # #adding risk
            # gamma=0.00
            # risk=gamma*w2_std
            current_portfolio_value = self.e.get_portfolio_value("T"+str(agent_idx+1))
            
            

            risk=0
            ########################
            ### A reward choice #1 : current portfolio value - old portfolio value
            # result: no trade happens after only one trade.
#            r = current_portfolio_value - self.old_portfolio_value[agent_idx]
            ########################
            
            
            ########################
            ### A reward choice #2 : current portfolio value - old portfolio value - market return - management fee
            # rationale : An agent should be able to outperform the market after deduction of management fee.
            ########################           
            price_history = self.get_price_history_for_all(self.L)
            market_return_t = self.get_market_return(price_history)

            mgmt_fee = 10
            
            r = current_portfolio_value - self.old_portfolio_value[agent_idx] - market_return_t - mgmt_fee
            
            self.old_portfolio_value[agent_idx] = current_portfolio_value
            #(np.dot(w2, price)[0] - mu)[0]

            sign = 1 if r > 0 else -1
            reward = np.log(np.abs(r) + eps)
            reward *= sign
            print ("At t=", self.t, "[T"+str(agent_idx+1), "]  reward: ",reward, "  Total P/F value: ", current_portfolio_value, "price:", np.around(price,1))
#            w2 = w2 / (np.dot(w2, price) + eps)
            self.t += 1
            if self.t == 30:   #len(self.states):
                not_terminal = 0    # 0 means that 'continue' variable is equal to zero.
                self.reset()

            
            
                
            price = np.squeeze(price)
            price = self.get_rel_price(price)
            
#            price_history = self.get_price_history_for_all(self.L)

            
            ### ms: 2 -> len(features)   - this change is needed.
            state = get_state(self, self.init_stock_price, price_history, N)
            state = state.reshape(1, self.M, self.L, N)
    #def get_state(env, init_stock_price, price_hisotry, N):
            state = np.array(state)
            
            next_state = state
            
            price = np.array(price)
            #print ("State stape in step: ", next_state.shape)
            info = {'reward': reward, 'continue': not_terminal, 'next state': next_state, #self.states[self.t],
                    'weight vector': w2, 'price': price,'risk':risk}
            return info
        
        else:
            M =len(self.codes)+1
#            for stock in self.e.stocks.keys():
#                self.e.stocks[stock]._last_settled_price = 100

            price = [[1]]
            price += ([[self.e.stocks[stock]._last_settled_price] for stock in self.e.stocks.keys()])
            price = np.squeeze(price)
            price = self.get_rel_price(price)
            
            price_history = self.get_price_history_for_all(self.L)
            ### ms: I commented it (below)
            #state = np.random.uniform(-1,1, (1,6,10,2))#1, M, int(window_length), len(features)))
            state = get_state(self, self.init_stock_price, price_history, N)
            state = state.reshape(1, self.M, self.L, N)
            
            
            #print ("State shape in step initial: ", state.shape)
            price = np.array(price)
            info = {'reward': 0, 'continue': 1, 'next state': state,#self.states[self.t],
                        'weight vector': np.array([[1] + [0 for i in range(self.M-1)]]),
                        'price': price,'risk':0}
            
            self.old_portfolio_value = [self.e.get_portfolio_value("T1")] * self.num_agents

            self.FLAG=True
            return info

    def reset(self):
#        self.t=self.L+1
        self.t=0
        self.FLAG = False

    def get_codes(self):
        return self.codes

    def get_all_last_settled_price(self):
        return [self.e.stocks[stock]._last_settled_price for stock in self.e.stocks.keys()]

def build_parser():
    parser = ArgumentParser(description='Provide arguments for training different DDPG or PPO models in Portfolio Management')
    parser.add_argument("--mode",choices=['train','test','download'])
    parser.add_argument("--num",type=int)
    return parser



args = {}
args['mode'] = 'train'
args['num'] = 3


config = {'data': {'start_date': '2015-01-01', 'end_date': '2018-01-01', \
                   'market_types': ['stock'], 'ktype': 'D'}, \
          'session': {'start_date': '2007-01-01', 'end_date': '2018-12-30', \
                      'market_types': 'China', 'codes': 2, 'features': ['close', 'high'], \
#                      'agents': ['CNN', 'DDPG', '10'], 'epochs': '100', 'noise_flag': 'False', \
                      'agents': ['CNN', 'PG', '10'], 'epochs': '100', 'noise_flag': 'False', \
                      'record_flag': 'False', 'plot_flag': 'False', 'reload_flag': 'False', \
                      'trainable': 'True', 'method': 'model_free'}}
# print (config)
# session(config,args)



    

def main():    
    session(config,args)

if __name__=="__main__":
    main()
