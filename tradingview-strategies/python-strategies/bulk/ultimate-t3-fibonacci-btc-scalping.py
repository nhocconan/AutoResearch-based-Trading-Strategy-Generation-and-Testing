#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "Ultimate T3 Fibonacci BTC Scalping"
timeframe = "30m"
leverage = 1

def ema(series, length):
    alpha = 2.0 / (length + 1.0)
    result = np.zeros_like(series)
    if len(series) > 0:
        result[0] = series[0]
        for i in range(1, len(series)):
            result[i] = alpha * series[i] + (1.0 - alpha) * result[i-1]
    return result

def t3_indicator(high, low, close, length, vf):
    src = (high + low + 2.0 * close) / 4.0
    e1 = ema(src, length)
    e2 = ema(e1, length)
    e3 = ema(e2, length)
    e4 = ema(e3, length)
    e5 = ema(e4, length)
    e6 = ema(e5, length)
    
    c1 = -vf**3
    c2 = 3.0 * vf**2 + 3.0 * vf**3
    c3 = -6.0 * vf**2 - 3.0 * vf - 3.0 * vf**3
    c4 = 1.0 + 3.0 * vf + vf**3 + 3.0 * vf**2
    
    t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    return t3

def generate_signals(prices):
    high = prices['high'].to_numpy()
    low = prices['low'].to_numpy()
    close = prices['close'].to_numpy()
    
    t3_len = 33
    t3_vf = 0.6
    t3_fibo_len = 19
    t3_fibo_vf = 0.418
    tp_pct = 0.15
    sl_pct = 0.02
    
    t3 = t3_indicator(high, low, close, t3_len, t3_vf)
    t32 = t3_indicator(high, low, close, t3_fibo_len, t3_fibo_vf)
    
    n = len(prices)
    signals = np.zeros(n)
    
    position = 0
    entry_price = 0.0
    
    for i in range(n):
        if i == 0:
            signals[i] = 0
            continue
            
        long_signal = (t32[i] > t3[i]) and (t32[i-1] <= t3[i-1])
        short_signal = (t32[i] < t3[i]) and (t32[i-1] >= t3[i-1])
        
        if position == 1:
            tp_level = entry_price * (1.0 + tp_pct)
            sl_level = entry_price * (1.0 - sl_pct)
            if high[i] >= tp_level or low[i] <= sl_level:
                position = 0
                entry_price = 0.0
            elif short_signal:
                position = 0
                entry_price = 0.0
        elif position == -1:
            tp_level = entry_price * (1.0 - tp_pct)
            sl_level = entry_price * (1.0 + sl_pct)
            if low[i] <= tp_level or high[i] >= sl_level:
                position = 0
                entry_price = 0.0
            elif long_signal:
                position = 0
                entry_price = 0.0
        
        if position == 0:
            if long_signal:
                position = 1
                entry_price = close[i]
            elif short_signal:
                position = -1
                entry_price = close[i]
        
        signals[i] = position
        
    return signals

if __name__ == "__main__":
    pass
