#!/usr/bin/env python3
import pandas as pd
import numpy as np

name = "BTC Intraday Advanced Spot PRO V6"
timeframe = "5m"
leverage = 1

def calculate_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def calculate_rsi(series, length):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def generate_signals(prices):
    df = prices.copy()
    n = len(df)
    signals = np.zeros(n, dtype=int)
    
    ema_fast = calculate_ema(df['close'], 21)
    ema_slow = calculate_ema(df['close'], 50)
    rsi = calculate_rsi(df['close'], 14)
    
    position = 0
    entry_price = 0.0
    sl_price = 0.0
    tp1_price = 0.0
    tp2_price = 0.0
    be_active = False
    
    sl_perc = 0.005
    tp1_perc = 0.007
    tp2_perc = 0.015
    rsi_buy = 55
    rsi_sell = 45
    
    for i in range(1, n):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        close = df['close'].iloc[i]
        
        if position != 0:
            if not be_active:
                if position == 1 and high >= tp1_price:
                    sl_price = entry_price
                    be_active = True
                elif position == -1 and low <= tp1_price:
                    sl_price = entry_price
                    be_active = True
            
            if position == 1:
                if low <= sl_price or high >= tp2_price:
                    position = 0
            elif position == -1:
                if high >= sl_price or low <= tp2_price:
                    position = 0
        
        if position == 0:
            if not np.isnan(ema_fast.iloc[i]) and not np.isnan(ema_slow.iloc[i]) and not np.isnan(rsi.iloc[i]):
                cross_over = ema_fast.iloc[i] > ema_slow.iloc[i] and ema_fast.iloc[i-1] <= ema_slow.iloc[i-1]
                cross_under = ema_fast.iloc[i] < ema_slow.iloc[i] and ema_fast.iloc[i-1] >= ema_slow.iloc[i-1]
                
                if cross_over and rsi.iloc[i] > rsi_buy:
                    position = 1
                    entry_price = close
                    sl_price = entry_price * (1 - sl_perc)
                    tp1_price = entry_price * (1 + tp1_perc)
                    tp2_price = entry_price * (1 + tp2_perc)
                    be_active = False
                elif cross_under and rsi.iloc[i] < rsi_sell:
                    position = -1
                    entry_price = close
                    sl_price = entry_price * (1 + sl_perc)
                    tp1_price = entry_price * (1 - tp1_perc)
                    tp2_price = entry_price * (1 - tp2_perc)
                    be_active = False
        
        signals[i] = position
    
    return signals