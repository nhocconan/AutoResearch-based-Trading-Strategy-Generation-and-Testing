#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "ETH Signal 15m"
timeframe = "15m"
leverage = 1

def _rma(series, length):
    """Calculate Rolling Moving Average (Wilder's Smoothing)"""
    alpha = 1.0 / length
    result = np.zeros_like(series)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1.0 - alpha) * result[i-1]
    return result

def _calculate_atr(high, low, close, length):
    """Calculate ATR using RMA"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = _rma(tr, length)
    return atr

def _calculate_rsi(close, length):
    """Calculate RSI using RMA"""
    diff = np.diff(close)
    diff = np.insert(diff, 0, 0)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    avg_gain = _rma(gain, length)
    avg_loss = _rma(loss, length)
    rs = np.where(avg_loss == 0, 100.0, avg_gain / avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def _calculate_supertrend(high, low, close, atr, factor):
    """Calculate Supertrend direction (1 or -1)"""
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + factor * atr
    basic_lower = hl2 - factor * atr
    
    final_upper = np.zeros_like(close)
    final_lower = np.zeros_like(close)
    direction = np.zeros_like(close)
    
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    direction[0] = 1
    
    for i in range(1, len(close)):
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]
            
        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]
            
        if close[i] > final_upper[i]:
            direction[i] = 1
        elif close[i] < final_lower[i]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
    return direction

def generate_signals(prices):
    """
    Generates trading signals based on Supertrend, RSI, and ATR logic.
    Returns a numpy array of positions (1=Long, -1=Short, 0=Flat).
    """
    n = len(prices)
    signals = np.zeros(n, dtype=int)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Constants
    atr_period = 12
    supertrend_factor = 2.76
    rsi_length = 12
    rsi_overbought = 70
    rsi_oversold = 30
    # Indicators
    atr = _calculate_atr(high, low, close, atr_period)
    rsi = _calculate_rsi(close, rsi_length)
    direction = _calculate_supertrend(high, low, close, atr, supertrend_factor)
    
    # State
    position = 0
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    
    for i in range(n):
        # Check Exits first (using current bar High/Low)
        if position == 1:
            if low[i] <= stop_loss or high[i] >= take_profit:
                position = 0
        elif position == -1:
            if high[i] >= stop_loss or low[i] <= take_profit:
                position = 0
        
        # Check Entries (if flat, using previous bar indicators to avoid lookahead)
        if position == 0 and i >= 1:
            dir_change = direction[i-1] - direction[i-2] if i >= 2 else 0
            rsi_val = rsi[i-1]
            
            # Long Condition
            if dir_change < 0 and rsi_val < rsi_overbought:
                position = 1
                entry_price = close[i-1]
                atr_val = atr[i-1]
                stop_loss = entry_price - 4.0 * atr_val
                take_profit = entry_price + 2.0 * atr_val
            
            # Short Condition
            elif dir_change > 0 and rsi_val > rsi_oversold:
                position = -1
                entry_price = close[i-1]
                atr_val = atr[i-1]
                stop_loss = entry_price + 4.0 * atr_val
                take_profit = entry_price - 2.237 * atr_val
        
        signals[i] = position
    
    return signals
