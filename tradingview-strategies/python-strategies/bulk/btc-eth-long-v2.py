#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "BTC_ETH_Long_V2"
timeframe = "4h"
leverage = 1

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def sma(series, length):
    return series.rolling(window=length).mean()

def macd(series, fast, slow, signal):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def is_rising(series, lookback):
    if len(series) < lookback + 1:
        return np.zeros(len(series), dtype=bool)
    diffs = series.diff() >= 0
    rising = diffs.rolling(window=lookback).sum() == lookback
    return rising.fillna(False).values

def crossunder(series1, series2):
    s1_prev = series1.shift(1)
    s2_prev = series2.shift(1)
    cond = (s1_prev >= s2_prev) & (series1 < series2)
    return cond.fillna(False).values

def generate_signals(prices):
    df = prices.copy()
    n = len(df)
    signals = np.zeros(n, dtype=int)
    
    if n < 200:
        return signals
        
    close = df['close']
    
    ema20 = ema(close, 20)
    sma100 = sma(close, 100)
    sma200 = sma(close, 200)
    macd_line, signal_line = macd(close, 12, 26, 7)
    
    ema20_v = ema20.values
    sma100_v = sma100.values
    sma200_v = sma200.values
    macd_line_v = macd_line.values
    close_v = close.values
    
    sma200_rising = is_rising(sma200, 10)
    macd_rising = is_rising(macd_line, 3)
    ema20_rising = is_rising(ema20, 2)
    sma100_rising = is_rising(sma100, 3)
    crossunder_ema_sma = crossunder(ema20, sma100)
    
    in_position = False
    entry_price = 0.0
    stop_loss_price = 0.0
    stop_loss_pct = 0.05
    
    start_idx = 200
    
    for i in range(start_idx, n):
        prev = i - 1
        
        if np.isnan(ema20_v[prev]) or np.isnan(sma100_v[prev]) or np.isnan(sma200_v[prev]):
            continue
            
        if in_position:
            if close_v[prev] <= stop_loss_price:
                in_position = False
                signals[i] = 0
                continue
            
            if crossunder_ema_sma[prev]:
                in_position = False
                signals[i] = 0
                continue
            
            signals[i] = 1
        else:
            cond_slow_rising = sma200_rising[prev]
            cond_macd_rising = macd_rising[prev]
            cond_ema_rising = ema20_rising[prev]
            cond_sma_rising = sma100_rising[prev]
            cond_ema_gt_sma = ema20_v[prev] > sma100_v[prev]
            cond_sma_lt_close = sma100_v[prev] < close_v[prev]
            
            if (cond_slow_rising and cond_macd_rising and cond_ema_rising and 
                cond_sma_rising and cond_ema_gt_sma and cond_sma_lt_close):
                
                in_position = True
                signals[i] = 1
                entry_price = close_v[prev]
                stop_loss_price = entry_price * (1.0 - stop_loss_pct)
            else:
                signals[i] = 0
                
    return signals
