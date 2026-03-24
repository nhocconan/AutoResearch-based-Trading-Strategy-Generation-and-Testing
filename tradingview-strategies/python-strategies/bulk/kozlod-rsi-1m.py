#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "Kozlod - RSI Strategy - 1 minute - ETHUSD"
timeframe = "1m"
leverage = 1

def _calculate_rsi(close, length):
    close = np.array(close, dtype=float)
    n = len(close)
    rsi = np.zeros(n)
    if n < length + 1:
        return rsi
    
    diff = np.diff(close)
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    # First Simple Moving Average for the first 'length' periods
    # gains[:length] corresponds to changes ending at index 'length'
    avg_gain[length] = np.mean(gains[:length])
    avg_loss[length] = np.mean(losses[:length])
    
    # Wilder's Smoothing for subsequent periods
    for i in range(length + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (length - 1) + gains[i-1]) / length
        avg_loss[i] = (avg_loss[i-1] * (length - 1) + losses[i-1]) / length
        
    rs = np.zeros(n)
    mask = avg_loss != 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    close = prices['close'].values
    length = 23
    over_sold = 54.0
    over_bought = 65.0
    
    rsi = _calculate_rsi(close, length)
    
    n = len(close)
    decisions = np.zeros(n, dtype=int)
    current_pos = 0
    
    # Iterate starting from where RSI is fully valid
    for i in range(length + 1, n):
        if np.isnan(rsi[i]) or np.isnan(rsi[i-1]):
            decisions[i] = current_pos
            continue
            
        # Crossover above overSold (54) -> Long
        if rsi[i] > over_sold and rsi[i-1] <= over_sold:
            current_pos = 1
        # Crossunder below overBought (65) -> Short
        elif rsi[i] < over_bought and rsi[i-1] >= over_bought:
            current_pos = -1
            
        decisions[i] = current_pos
    
    # Shift signals by 1 to prevent lookahead bias (execute on next bar open)
    signals = np.roll(decisions, 1)
    signals[0] = 0
    
    return signals
