#!/usr/bin/env python3
"""
SuperTrend Strategy with RSI filter for BTCUSD 4H
Converted from TradingView Pine Script
"""

import numpy as np
import pandas as pd

name = "SuperTrend RSI Strategy BTCUSD 4H"
timeframe = "4h"
leverage = 10


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    atr = np.zeros_like(close, dtype=np.float64)
    atr[0] = tr[0]
    for i in range(1, len(close)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_rsi(close, period=6):
    """Calculate RSI"""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.zeros_like(close, dtype=np.float64)
    avg_loss = np.zeros_like(close, dtype=np.float64)
    
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_supertrend(high, low, close, atr_period=9, factor=2.5):
    """Calculate SuperTrend indicator"""
    atr = calculate_atr(high, low, close, atr_period)
    
    hl2 = (high + low) / 2
    
    upper_band = hl2 + factor * atr
    lower_band = hl2 - factor * atr
    
    supertrend = np.zeros_like(close, dtype=np.float64)
    direction = np.zeros_like(close, dtype=np.float64)
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close)):
        if close[i-1] <= supertrend[i-1]:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
            if close[i] > supertrend[i]:
                direction[i] = -1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = 1
        else:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
            if close[i] < supertrend[i]:
                direction[i] = 1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = -1
    
    return supertrend, direction


def generate_signals(prices):
    """
    Generate trading signals based on SuperTrend and RSI
    
    Args:
        prices: pandas DataFrame with columns: open_time, open, high, low, close, volume
    
    Returns:
        numpy array with signals: 1=long, -1=short, 0=neutral
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    if n < 50:
        return signals
    
    close = prices['close'].values.astype(np.float64)
    high = prices['high'].values.astype(np.float64)
    low = prices['low'].values.astype(np.float64)
    
    supertrend, direction = calculate_supertrend(high, low, close, atr_period=9, factor=2.5)
    rsi = calculate_rsi(close, period=6)
    atr = calculate_atr(high, low, close, period=14)
    
    oversold = 30
    overbought = 70
    atr_multiplier = 1.5
    rr_breakeven = 0.75
    rr_takeprofit = 0.75
    
    long_supertrend = (np.roll(direction, 1) > 0) & (direction < 0)
    short_supertrend = (np.roll(direction, 1) < 0) & (direction > 0)
    
    long_rsi = (rsi > oversold) & (np.roll(rsi, 1) <= oversold) & (direction < 0)
    short_rsi = (rsi < overbought) & (np.roll(rsi, 1) >= overbought) & (direction > 0)
    
    long_entry = long_supertrend | long_rsi
    short_entry = short_supertrend | short_rsi
    
    in_long = False
    in_short = False
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    breakeven = 0.0
    
    for i in range(1, n):
        if not in_long and not in_short:
            if long_entry[i]:
                in_long = True
                entry_price = close[i]
                sl_distance = atr[i] * atr_multiplier
                stop_loss = entry_price - sl_distance
                tp_distance = sl_distance * rr_takeprofit
                take_profit = entry_price + tp_distance
                breakeven = entry_price + sl_distance * rr_breakeven
                signals[i] = 1
            elif short_entry[i]:
                in_short = True
                entry_price = close[i]
                sl_distance = atr[i] * atr_multiplier
                stop_loss = entry_price + sl_distance
                tp_distance = sl_distance * rr_takeprofit
                take_profit = entry_price - tp_distance
                breakeven = entry_price - sl_distance * rr_breakeven
                signals[i] = -1
        
        elif in_long:
            if high[i] >= breakeven:
                stop_loss = entry_price
            if low[i] <= stop_loss or high[i] >= take_profit or short_supertrend[i]:
                in_long = False
                signals[i] = 0
        
        elif in_short:
            if low[i] <= breakeven:
                stop_loss = entry_price
            if high[i] >= stop_loss or low[i] <= take_profit or long_supertrend[i]:
                in_short = False
                signals[i] = 0
    
    return signals


if __name__ == "__main__":
    print(f"Strategy: {name}")
    print(f"Timeframe: {timeframe}")
    print(f"Leverage: {leverage}")
